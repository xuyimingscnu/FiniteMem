from typing import Optional, List
import numpy as np


class KnowsightNode:
    MAX_RETRIEVAL_HISTORY = 10

    def __init__(
        self,
        node_id: str,
        knowsight_text: str,
        question: str,
        creation_round: int = 0,
    ) -> None:

        self.node_id = node_id
        self.knowsight_text = knowsight_text
        self.question = question
        self.importance_score = 5  
        self.creation_round = creation_round
        self.retrieval_history: List[int] = []  
        self.knowsight_text_embedding: Optional[np.ndarray] = None
        self.question_embedding: Optional[np.ndarray] = None

    def get_id(self) -> str:
        return self.node_id

    def set_id(self, new_id: str) -> None:
        self.node_id = new_id

    def delete_id(self) -> None:
        self.node_id = ""

    def get_knowsight_text(self) -> str:
        return self.knowsight_text

    def set_knowsight_text(self, new_text: str) -> None:
        self.knowsight_text = new_text
        self.knowsight_text_embedding = None   

    def delete_knowsight_text(self) -> None:
        self.knowsight_text = ""
        self.knowsight_text_embedding = None

    def get_question(self) -> str:
        return self.question

    def set_question(self, new_question: str) -> None:
        self.question = new_question
        self.question_embedding = None   

    def delete_question(self) -> None:
        self.question = ""
        self.question_embedding = None

    def get_importance_score(self) -> int:
        return self.importance_score

    def set_importance_score(self, new_score: int) -> None:
        self.importance_score = new_score

    def increase_importance(self, delta: int = 1) -> None:
        self.importance_score += delta

    def decrease_importance(self, delta: int = 1) -> None:
        self.importance_score -= delta

    def delete_importance_score(self) -> None:
        self.importance_score = 5

    def get_creation_round(self) -> int:
        return self.creation_round

    def set_creation_round(self, new_round: int) -> None:
        self.creation_round = new_round

    def delete_creation_round(self) -> None:
        self.creation_round = 0

    def get_retrieval_history(self) -> List[int]:
        return self.retrieval_history.copy()

    def add_retrieval_round(self, round_num: int) -> None:
        self.retrieval_history.append(round_num)
        while len(self.retrieval_history) > self.MAX_RETRIEVAL_HISTORY:
            self.retrieval_history.pop(0)

    def set_retrieval_history(self, history: List[int]) -> None:
        if history:
            self.retrieval_history = history[-self.MAX_RETRIEVAL_HISTORY:].copy()
        else:
            self.retrieval_history = []

    def clear_retrieval_history(self) -> None:
        self.retrieval_history.clear()

    def delete_retrieval_history(self) -> None:
        self.clear_retrieval_history()

    def to_dict(self) -> dict:
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
        node = cls(
            node_id=data["node_id"],
            knowsight_text=data["knowsight_text"],
            question=data["question"],
            creation_round=data.get("creation_round", 0),
        )
        node.set_importance_score(data.get("importance_score", 5))
        node.set_retrieval_history(data.get("retrieval_history", []))  # 自动应用容量限制
        return node

    def __repr__(self) -> str:
        return (
            f"KnowsightNode(id={self.node_id!r}, "
            f"importance={self.importance_score}, "
            f"creation_round={self.creation_round}, "
            f"retrievals={len(self.retrieval_history)}, "
            f"question={self.question[:20]!r}...)"
        )