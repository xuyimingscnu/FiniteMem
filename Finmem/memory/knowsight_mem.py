# med_memory/memory/insight_mem.py

import sys
import os
import ast
import json
from typing import List, Dict, Optional, Union, Any


from tool.API import generate
from memory.prompt import insight_prompt, knowledge_prompt
from memory.knowsight_node import KnowsightNode

from memory.retrieve import knowsight_retrieve, knowsight_mem_retrieve
from memory.forget_replay import mem_replay


class KnowsightLibrary:

    def __init__(self, lib_type: str):
        if lib_type not in ("insight", "knowledge"):
            raise ValueError("lib_type 必须为 'insight' 或 'knowledge'")

        self.lib_type = lib_type
        self.max_nodes = 100 if lib_type == "insight" else 500
        self.prompt = insight_prompt if lib_type == "insight" else knowledge_prompt

        self.nodes: List[KnowsightNode] = []          
        self.text_to_id: Dict[str, str] = {}          
        self._next_id = 1                             
        self.dialogue_round: int = 0                  
        self.overflow_buffer: List[KnowsightNode] = []  

    def _generate_node_id(self) -> str:
        node_id = f"{self.lib_type}_{self._next_id}"
        self._next_id += 1
        return node_id

    def increment_round(self) -> None:
        self.dialogue_round += 1

    def _create_knowsight_text(self, question: str, steps: List[str]) -> str:
        steps_str = "\n".join(steps)
        if self.lib_type == "insight":
            user_prompt = self.prompt +f"请根据执行记录生成经验总结\n执行记录：\n{steps_str}"
        else:
            user_prompt = self.prompt + f"\n执行记录：{steps_str}"
        result = generate(prompt=user_prompt)
        
        return result.strip() if isinstance(result, str) else str(result)

    def _normalize_quotes_for_list(self, s: str) -> str:
        quote_map = {
            '\u201c': '"',   # 左双引号 “
            '\u201d': '"',   # 右双引号 ”
            '\u300c': '"',   # 左直角引号 「
            '\u300d': '"',   # 右直角引号 」
            '\uff02': '"',   # 全角双引号 ＂
        }
        for k, v in quote_map.items():
            s = s.replace(k, v)
        return s

    def _parse_possible_list_string(self, s: str) -> List[str]:
        s = s.strip()
        
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, TypeError):
            pass

        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (SyntaxError, ValueError):
            pass

        normalized = self._normalize_quotes_for_list(s)
        try:
            parsed = json.loads(normalized)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (json.JSONDecodeError, TypeError):
            pass

        return [s]

    def _create_nodes_from_texts(
        self, text_source: Union[str, List[str]], question: str
    ) -> List[KnowsightNode]:
        # 如果已经是列表，直接使用
        if self.lib_type == "knowledge":
            text_list = self._parse_possible_list_string(text_source)
        else:
            text_list = [text_source]


        nodes_created = []
        for text in text_list:
            text = text.strip()
            if not text:
                continue
            if knowsight_retrieve(self.nodes, text):
                continue
            node_id = self._generate_node_id()
            new_node = KnowsightNode(
                node_id=node_id,
                knowsight_text=text,
                question=question,
                creation_round=self.dialogue_round
            )
            nodes_created.append(new_node)
        return nodes_created

    def add_from_input(self, input_data: Dict):
        question = input_data.get("question", "")
        steps = input_data.get("step", [])

        if not steps:
            print("警告：输入数据中缺少 'step' 字段，无法生成节点。")
            return None

        knowsight_result = self._create_knowsight_text(question, steps)
        if not knowsight_result:
            print("错误：生成文本失败，节点添加终止。")
            return None

        new_nodes = self._create_nodes_from_texts(knowsight_result, question)
        if not new_nodes:
            return None

        for node in new_nodes:
            if len(self.nodes) < self.max_nodes:
                self.nodes.append(node)
                self.text_to_id[node.get_knowsight_text()] = node.get_id()
            else:
                self.overflow_buffer.append(node)

        if self.overflow_buffer:
            self._trigger_replay()

        return None

    def _trigger_replay(self) -> None:
        texts_to_delete = mem_replay(self.nodes, self.dialogue_round, self.overflow_buffer)

        if not isinstance(texts_to_delete, list):
            print(f"警告：mem_replay 应返回列表，实际返回类型为 {type(texts_to_delete)}，尝试强制转换。")
            texts_to_delete = [texts_to_delete] if texts_to_delete else []

        delete_set = set(texts_to_delete)

        all_nodes = self.nodes + self.overflow_buffer

        retained_nodes = [node for node in all_nodes if node.get_knowsight_text() not in delete_set]

        self._rebuild_from_nodes_list(retained_nodes)

        self.overflow_buffer.clear()


    def _rebuild_from_nodes_list(self, node_list: List[KnowsightNode]) -> None:
        self.nodes = node_list.copy()
        self.text_to_id.clear()
        for node in self.nodes:
            self.text_to_id[node.get_knowsight_text()] = node.get_id()

    def get_node_by_text(self, text: str) -> Optional[KnowsightNode]:
        node_id = self.text_to_id.get(text)
        if node_id:
            for node in self.nodes:
                if node.get_id() == node_id:
                    return node
        return None

    def get_node_by_id(self, node_id: str) -> Optional[KnowsightNode]:
        for node in self.nodes:
            if node.get_id() == node_id:
                return node
        return None

    def delete_node_by_id(self, node_id: str) -> bool:
        for i, node in enumerate(self.nodes):
            if node.get_id() == node_id:
                if node.get_knowsight_text() in self.text_to_id:
                    del self.text_to_id[node.get_knowsight_text()]
                self.nodes.pop(i)
                return True
        return False

    def update_importance_by_text(self, knowsight_text: str, delta: int) -> bool:
        node = self.get_node_by_text(knowsight_text)
        if not node:
            print(f"警告：未找到文本为 '{knowsight_text[:30]}...' 的节点。")
            return False

        if delta > 0:
            node.increase_importance(delta)
        else:
            node.decrease_importance(abs(delta))

        if node.get_importance_score() <= 0:
            self.delete_node_by_id(node.get_id())
            print(f"信息：节点 {node.get_id()} 重要性分数降为0，已删除。")

        return True

    def get_all_nodes(self) -> List[KnowsightNode]:
        return self.nodes.copy()

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"<{self.lib_type.capitalize()}Library with {len(self.nodes)} nodes>"

    def memory_retrieve(self, text: str):
        self.increment_round()
        k = 1 if self.lib_type == "insight" else 5
        similar_texts = knowsight_mem_retrieve(self.nodes, text, k)
        for similar_text in similar_texts:
            node = self.get_node_by_text(similar_text)
            if node is not None:
                node.add_retrieval_round(self.dialogue_round)
        return similar_texts

    def importance_score_modify(self, knowsight_text_list, effect: bool):
        for text in knowsight_text_list:
            node = self.get_node_by_text(text)
            if node is None:
                continue
            
            if effect:
                node.increase_importance(1)
            else:
                node.decrease_importance(1)
                if node.get_importance_score() == 0:
                    self.delete_node_by_id(node.get_id())
    def save_to_json(self, filepath: str) -> None:
        library_data = {
            "lib_type": self.lib_type,
            "dialogue_round": self.dialogue_round,
            "max_nodes": self.max_nodes,
            "nodes": [node.to_dict() for node in self.nodes]
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(library_data, f, ensure_ascii=False, indent=2)

