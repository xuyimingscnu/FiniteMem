class TrajectoryNode:

    def __init__(self, node_id: int, question: str, action: str, trajectory_text: str):
        self.id = node_id
        self.question = question
        self.action = action  
        self.trajectory_text = trajectory_text

        self.question_embedding = None      
        self.trajectory_embedding = None    

        self.predecessors = []

        self.successors = []

        self.importance_score = 0

    def set_question(self, new_question: str):
        self.question = new_question
        self.question_embedding = None      

    def set_action(self, new_action: str):
        allowed_actions = ['思考', '搜索', '总结', '观察']
        if new_action not in allowed_actions:
            raise ValueError(f"动作必须为 {allowed_actions} 之一")
        self.action = new_action

    def set_trajectory_text(self, new_text: str):
        self.trajectory_text = new_text
        self.trajectory_embedding = None    

    def add_predecessor(self, pred_id: int, score: int):
        self.remove_predecessor(pred_id)
        self.predecessors.append({'node_id': pred_id, 'score': score})

    def remove_predecessor(self, pred_id: int) -> bool:
        for pred in self.predecessors:
            if pred['node_id'] == pred_id:
                self.predecessors.remove(pred)
                return True
        return False

    def update_predecessor_score(self, pred_id: int, new_score: int) -> bool:
        for pred in self.predecessors:
            if pred['node_id'] == pred_id:
                pred['score'] = new_score
                return True
        return False

    def get_predecessors(self):
        return self.predecessors.copy()  

    def clear_predecessors(self):
        self.predecessors.clear()

    def add_successor(self, succ_id: int, score: int):
        self.remove_successor(succ_id)
        self.successors.append({'node_id': succ_id, 'score': score})

    def remove_successor(self, succ_id: int) -> bool:
        for succ in self.successors:
            if succ['node_id'] == succ_id:
                self.successors.remove(succ)
                return True
        return False

    def update_successor_score(self, succ_id: int, new_score: int) -> bool:
        for succ in self.successors:
            if succ['node_id'] == succ_id:
                succ['score'] = new_score
                return True
        return False

    def get_successors(self):
        return self.successors.copy()

    def clear_successors(self):
        self.successors.clear()

    def compute_importance_score(self, node_dict: dict) -> int:
        total = 0
        for pred_info in self.predecessors:
            total += pred_info['score']
        for succ_info in self.successors:
            total += succ_info['score']
        self.importance_score = total
        return total

    def __repr__(self):
        return (f"TrajectoryNode(id={self.id}, action='{self.action}', "
                f"pred_count={len(self.predecessors)}, succ_count={len(self.successors)}, "
                f"importance={self.importance_score})")

    def detailed_info(self):
        return {
            'id': self.id,
            'question': self.question,
            'action': self.action,
            'trajectory_text': self.trajectory_text,
            'predecessors': self.predecessors.copy(),
            'successors': self.successors.copy(),
            'importance_score': self.importance_score
        }
        
    def to_dict(self):
        return {
            'id': self.id,
            'question': self.question,
            'action': self.action,
            'trajectory_text': self.trajectory_text,
            'predecessors': self.predecessors.copy(),
            'successors': self.successors.copy(),
            'importance_score': self.importance_score
        }
