from typing import Optional, List


class KnowsightNode:
    """
    洞察力节点类，用于存放由轨迹信息总结而来的经验结论。

    Attributes:
        node_id (str): 洞察力节点的唯一编号。
        knowsight_text (str): 由大模型对轨迹信息总结而来的经验结论。
        question (str): 生成该洞察力信息的轨迹的问题。
        importance_score (int): 重要性分数，初始为5，后续可调整。
        creation_round (int): 创建该节点的轮次编号。
        retrieval_history (List[int]): 记录该节点被检索到的所有轮次列表。
    """

    def __init__(
        self,
        node_id: str,
        knowsight_text: str,
        question: str,
        creation_round: int = 0,
    ) -> None:
        """
        初始化洞察力节点。

        Args:
            node_id (str): 节点的唯一编号。
            knowsight_text (str): 洞察力文本内容。
            question (str): 对应的问题描述。
            creation_round (int): 创建该节点的轮次，默认为0。
        """
        self.node_id = node_id
        self.knowsight_text = knowsight_text
        self.question = question
        self.importance_score = 5  # 初始重要性分数为5
        self.creation_round = creation_round
        self.retrieval_history: List[int] = []  # 初始为空列表

    # ---------- 增删改查方法 ----------
    # 1. 编号 (node_id)
    def get_id(self) -> str:
        """获取节点编号"""
        return self.node_id

    def set_id(self, new_id: str) -> None:
        """修改节点编号（慎用，应确保唯一性）"""
        self.node_id = new_id

    def delete_id(self) -> None:
        """删除编号（置为空字符串，不推荐）"""
        self.node_id = ""

    # 2. 洞察力文本 (knowsight_text)
    def get_knowsight_text(self) -> str:
        """获取洞察力文本"""
        return self.knowsight_text

    def set_knowsight_text(self, new_text: str) -> None:
        """更新洞察力文本"""
        self.knowsight_text = new_text

    def delete_knowsight_text(self) -> None:
        """清空洞察力文本"""
        self.knowsight_text = ""

    # 3. 问题 (question)
    def get_question(self) -> str:
        """获取问题描述"""
        return self.question

    def set_question(self, new_question: str) -> None:
        """更新问题描述"""
        self.question = new_question

    def delete_question(self) -> None:
        """清空问题描述"""
        self.question = ""

    # 4. 重要性分数 (importance_score)
    def get_importance_score(self) -> int:
        """获取当前重要性分数"""
        return self.importance_score

    def set_importance_score(self, new_score: int) -> None:
        """
        设置重要性分数（覆盖原值）。

        Args:
            new_score (int): 新的重要性分数。
        """
        self.importance_score = new_score

    def increase_importance(self, delta: int = 1) -> None:
        """
        增加重要性分数。

        Args:
            delta (int): 增加量，默认为1。
        """
        self.importance_score += delta

    def decrease_importance(self, delta: int = 1) -> None:
        """
        减少重要性分数。

        Args:
            delta (int): 减少量，默认为1。
        """
        self.importance_score -= delta

    def delete_importance_score(self) -> None:
        """重置重要性分数为默认值5（模拟删除操作）"""
        self.importance_score = 5

    # 5. 创建轮次 (creation_round)
    def get_creation_round(self) -> int:
        """获取创建轮次"""
        return self.creation_round

    def set_creation_round(self, new_round: int) -> None:
        """设置创建轮次"""
        self.creation_round = new_round

    def delete_creation_round(self) -> None:
        """重置创建轮次为0（模拟删除）"""
        self.creation_round = 0

    # 6. 历史检索轮次 (retrieval_history)
    def get_retrieval_history(self) -> List[int]:
        """获取历史检索轮次列表的副本"""
        return self.retrieval_history.copy()

    def add_retrieval_round(self, round_num: int) -> None:
        """
        添加一个检索轮次记录。

        Args:
            round_num (int): 被检索到的轮次编号。
        """
        self.retrieval_history.append(round_num)

    def set_retrieval_history(self, history: List[int]) -> None:
        """
        覆盖设置整个历史检索轮次列表。

        Args:
            history (List[int]): 新的历史检索轮次列表。
        """
        self.retrieval_history = history.copy()

    def clear_retrieval_history(self) -> None:
        """清空历史检索轮次列表"""
        self.retrieval_history.clear()

    def delete_retrieval_history(self) -> None:
        """清空历史检索轮次列表（与 clear 等效）"""
        self.clear_retrieval_history()

    # ---------- 辅助方法 ----------
    def to_dict(self) -> dict:
        """
        将节点数据转换为字典，便于序列化或传输。

        Returns:
            dict: 包含所有字段的字典。
        """
        return {
            "node_id": self.node_id,
            "knowsight_text": self.knowsight_text,
            "question": self.question,
            "importance_score": self.importance_score,
            "creation_round": self.creation_round,
            "retrieval_history": self.retrieval_history.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowsightNode":
        """
        从字典创建knowsightNode实例。

        Args:
            data (dict): 包含node_id, knowsight_text, question, creation_round, 
                         importance_score, retrieval_history的字典。

        Returns:
            knowsightNode: 新构建的节点对象。
        """
        node = cls(
            node_id=data["node_id"],
            knowsight_text=data["knowsight_text"],
            question=data["question"],
            creation_round=data.get("creation_round", 0),
        )
        node.set_importance_score(data.get("importance_score", 5))
        node.set_retrieval_history(data.get("retrieval_history", []))
        return node

    def __repr__(self) -> str:
        return (
            f"knowsightNode(id={self.node_id!r}, "
            f"importance={self.importance_score}, "
            f"creation_round={self.creation_round}, "
            f"retrievals={len(self.retrieval_history)}, "
            f"question={self.question[:20]!r}...)"
        )

    