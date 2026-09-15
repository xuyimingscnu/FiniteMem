import json
from typing import List, Dict, Any, Optional, Set
from memory.track_node import TrajectoryNode
from memory.retrieve import action_retrieve, trajectory_mem_retrieve


class ActionLibrary:

    def __init__(self, action_type: str):
        self.action_type = action_type
        self.nodes: List[TrajectoryNode] = []          
        self.text_to_id: Dict[str, int] = {}           
        self.cluster_buffer: List[List[str]] = []      
        self.overflow_buffer: List[List[str]] = []     
        self.max_cluster_buffer_num: int = 500

    @property
    def count(self) -> int:
        return len(self.nodes)

    def add_node(self, node: TrajectoryNode) -> None:
        if node.action != self.action_type:
            raise ValueError(f"节点动作类型 '{node.action}' 与动作库类型 '{self.action_type}' 不匹配")

        self.nodes.append(node)
        self.text_to_id[node.trajectory_text] = node.id

    def remove_node_by_text(self, text: str) -> bool:
        if text not in self.text_to_id:
            return False

        node_id = self.text_to_id[text]
        for i, node in enumerate(self.nodes):
            if node.id == node_id:
                self.nodes.pop(i)
                break
        del self.text_to_id[text]
        return True

    def clear_relations_for_node(self, node_id: int) -> None:
        pass

    def __repr__(self):
        return f"ActionLibrary(type={self.action_type}, count={self.count})"


class TrajectoryLibrary:
    ACTION_TYPES = ['思考', '搜索', '总结']

    def __init__(self):
        # 全局节点存储，key为node_id
        self._nodes: Dict[int, TrajectoryNode] = {}
        # 下一个可用的节点ID
        self._next_node_id = 1

        # 三个动作库
        self.think_lib = ActionLibrary('思考')
        self.search_lib = ActionLibrary('搜索')
        self.summary_lib = ActionLibrary('总结')

        # 动作类型到库的映射，便于批量操作
        self._lib_map = {
            '思考': self.think_lib,
            '搜索': self.search_lib,
            '总结': self.summary_lib
        }

    def _generate_node_id(self) -> int:
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _extract_action_and_text(self, step_str: str) -> tuple:
        bracket_start = step_str.find('[')
        if bracket_start == -1:
            raise ValueError(f"无法解析步骤字符串，缺少 '[': {step_str}")

        action = step_str[:bracket_start].strip()
        text = step_str[bracket_start + 1:].rstrip(']')
        return action, text

    def build_from_input(self, input_data: Dict[str, Any]) -> None:
        question = input_data['question']
        steps = input_data['step']

        if not steps:
            return

        created_nodes: List[TrajectoryNode] = []
        trajectory_text_list = []
        for step_str in steps:
            action, text = self._extract_action_and_text(step_str)
            if text not in trajectory_text_list and action in self.ACTION_TYPES: 
                trajectory_text_list.append(text)
                node_id = self._generate_node_id()
                node = TrajectoryNode(node_id, question, action, text)
                created_nodes.append(node)
                self._nodes[node_id] = node

        for i, node in enumerate(created_nodes):
            if i > 0:
                pred_node = created_nodes[i - 1]
                node.add_predecessor(pred_node.id, score=5)
            if i < len(created_nodes) - 1:
                succ_node = created_nodes[i + 1]
                node.add_successor(succ_node.id, score=5)

        for node in created_nodes:
            if node.action in self.ACTION_TYPES:
                lib = self._lib_map[node.action]
                lib.add_node(node)
                self._add_to_buffer_on_create(lib, node)

        for node in created_nodes:
            if node.action not in self.ACTION_TYPES:
                continue  

            lib = self._lib_map[node.action]
            similar_texts = action_retrieve(lib, node)

            for sim_text in similar_texts:
                sim_node_id = lib.text_to_id.get(sim_text)
                if sim_node_id is None:
                    continue  
                sim_node = self._nodes.get(sim_node_id)
                if sim_node is None or sim_node.id == node.id:
                    continue

                for pred_info in sim_node.get_predecessors():
                    pred_id = pred_info['node_id']
                    if pred_id == node.id:
                        continue
                    node.add_predecessor(pred_id, score=5)
                    pred_node = self._nodes.get(pred_id)
                    if pred_node:
                        pred_node.add_successor(node.id, score=5)

                for succ_info in sim_node.get_successors():
                    succ_id = succ_info['node_id']
                    if succ_id == node.id:
                        continue
                    node.add_successor(succ_id, score=5)
                    succ_node = self._nodes.get(succ_id)
                    if succ_node:
                        succ_node.add_predecessor(node.id, score=5)
                        
        for node in self._nodes.values():
            node.compute_importance_score(self._nodes)
            
        self.replay()

    def _add_to_buffer_on_create(self, lib: ActionLibrary, node: TrajectoryNode) -> None:
        if lib.count <= lib.max_cluster_buffer_num:
            if len(lib.cluster_buffer) == 0:
                lib.cluster_buffer.append([node.trajectory_text])
            else:
                lib.cluster_buffer[0].append(node.trajectory_text)
        else:
            lib.overflow_buffer.append([node.trajectory_text])

    def add_node(self, node: TrajectoryNode) -> None:
        if node.id in self._nodes:
            raise ValueError(f"节点ID {node.id} 已存在")
        self._nodes[node.id] = node

        if node.action not in self.ACTION_TYPES:
            return

        lib = self._lib_map[node.action]
        lib.add_node(node)
        self._add_to_buffer_on_create(lib, node)

    def delete_node_by_text(self, action_type: str, text: str) -> bool:
        if action_type not in self._lib_map:
            return False

        lib = self._lib_map[action_type]
        if text not in lib.text_to_id:
            return False

        node_id = lib.text_to_id[text]
        node = self._nodes.get(node_id)
        if not node:
            return False

        self._clean_relations(node_id)
        lib.remove_node_by_text(text)

        del self._nodes[node_id]

        self._remove_text_from_buffers(lib, text)

        return True

    def _clean_relations(self, node_id: int) -> None:
        for node in self._nodes.values():
            node.remove_predecessor(node_id)
            node.remove_successor(node_id)

    def _remove_text_from_buffers(self, lib: ActionLibrary, text: str) -> None:
        new_cluster = []
        for group in lib.cluster_buffer:
            filtered = [t for t in group if t != text]
            if filtered:
                new_cluster.append(filtered)
        lib.cluster_buffer = new_cluster

        new_overflow = []
        for group in lib.overflow_buffer:
            filtered = [t for t in group if t != text]
            if filtered:
                new_overflow.append(filtered)
        lib.overflow_buffer = new_overflow

    def modify_node(self, action_type: str, old_text: str, new_node: TrajectoryNode) -> bool:
        if not self.delete_node_by_text(action_type, old_text):
            return False

        self.add_node(new_node)
        return True

    def replay(self) -> None:
        try:
            from memory.replay import memreplay
        except ImportError:
            def memreplay(action_lib):
                return [], action_lib.cluster_buffer
        else:
            pass

        for lib in [self.think_lib, self.search_lib, self.summary_lib]:
            if not lib.overflow_buffer:
                continue

            to_delete_texts, new_cluster_buffer = memreplay(lib)

            for text in to_delete_texts:
                self.delete_node_by_text(lib.action_type, text)

            lib.cluster_buffer = new_cluster_buffer

            lib.overflow_buffer.clear()

    def get_library_summary(self) -> Dict[str, Any]:
        return {
            'think': {
                'count': self.think_lib.count,
                'cluster_buffer_size': len(self.think_lib.cluster_buffer),
                'overflow_buffer_size': len(self.think_lib.overflow_buffer)
            },
            'search': {
                'count': self.search_lib.count,
                'cluster_buffer_size': len(self.search_lib.cluster_buffer),
                'overflow_buffer_size': len(self.search_lib.overflow_buffer)
            },
            'summary': {
                'count': self.summary_lib.count,
                'cluster_buffer_size': len(self.summary_lib.cluster_buffer),
                'overflow_buffer_size': len(self.summary_lib.overflow_buffer)
            },
            'total_nodes': len(self._nodes)
        }


    def memory_retrieve(self, question: str, previous_step: str = "") -> List[str]:
        if not previous_step:
            target_lib = self.think_lib
            text = ""
        else:
            action, text = self._extract_action_and_text(previous_step)
            next_action_map = {
                '思考': '搜索',
                '搜索': '总结',
                '总结': '思考'
            }
            if action not in next_action_map:
                return []
            target_lib = self._lib_map[next_action_map[action]]
        similar_texts = trajectory_mem_retrieve(target_lib, question, text)
        output = []
        for text in similar_texts:
            text = target_lib.action_type + "[" + text +"]"
            output.append(text)
        
        return output
    
    def importance_score_modify(self, trajectory_text_list, effect: bool):
        if len(trajectory_text_list) == 0:
            return
        
        res = []
        for text in trajectory_text_list:
            action, step = self._extract_action_and_text(text)
            res.append(step)
        trajectory_text_list = res
        
        nodes_to_process = []
        for text in trajectory_text_list:
            node = None
            for lib in (self.think_lib, self.search_lib, self.summary_lib):
                nid = lib.text_to_id.get(text)
                if nid is not None:
                    node = self._nodes.get(nid)
                    break
            if node:
                nodes_to_process.append(node)

        for cur_node in nodes_to_process:
            affected_nodes = set()  

            for pred_info in list(cur_node.predecessors):
                pred_id = pred_info['node_id']
                delta = 1 if effect else -1
                new_score = pred_info['score'] + delta
                pred_node = self._nodes.get(pred_id)

                if new_score == 0:
                    cur_node.predecessors.remove(pred_info)
                    if pred_node:
                        for succ_info in list(pred_node.successors):
                            if succ_info['node_id'] == cur_node.id:
                                pred_node.successors.remove(succ_info)
                                break
                        affected_nodes.add(pred_node)
                else:
                    pred_info['score'] = new_score
                    if pred_node:
                        for succ_info in pred_node.successors:
                            if succ_info['node_id'] == cur_node.id:
                                succ_info['score'] = new_score
                                break
                        affected_nodes.add(pred_node)

            for succ_info in list(cur_node.successors):
                succ_id = succ_info['node_id']
                delta = 1 if effect else -1
                new_score = succ_info['score'] + delta
                succ_node = self._nodes.get(succ_id)

                if new_score == 0:
                    cur_node.successors.remove(succ_info)
                    if succ_node:
                        for pred_info in list(succ_node.predecessors):
                            if pred_info['node_id'] == cur_node.id:
                                succ_node.predecessors.remove(pred_info)
                                break
                        affected_nodes.add(succ_node)
                else:
                    succ_info['score'] = new_score
                    if succ_node:
                        for pred_info in succ_node.predecessors:
                            if pred_info['node_id'] == cur_node.id:
                                pred_info['score'] = new_score
                                break
                        affected_nodes.add(succ_node)

            cur_node.compute_importance_score(self._nodes)
            for node in affected_nodes:
                node.compute_importance_score(self._nodes)

            if cur_node.importance_score == 0:
                self.delete_node_by_text(cur_node.action, cur_node.trajectory_text)
                lib = self._lib_map[cur_node.action]
                if lib.count <= lib.max_cluster_buffer_num:
                    all_texts = []
                    for group in lib.cluster_buffer:
                        all_texts.extend(group)
                    lib.cluster_buffer = [all_texts] if all_texts else []
    
    def save_to_json(self, directory: str) -> None:
        import os
        import json

        os.makedirs(directory, exist_ok=True)

        for action_type in self.ACTION_TYPES:
            lib = self._lib_map[action_type]

            data = {
                'action_type': lib.action_type,
                'max_cluster_buffer_num': lib.max_cluster_buffer_num,
                'nodes': [node.to_dict() for node in lib.nodes],
                'cluster_buffer': lib.cluster_buffer,
                'overflow_buffer': lib.overflow_buffer
            }

            filename = os.path.join(directory, f"{action_type}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)