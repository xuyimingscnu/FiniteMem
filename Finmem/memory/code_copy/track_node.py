class TrajectoryNode:
    """
    轨迹库节点类

    每个节点表示推理过程中的一个状态，包含问题、动作、轨迹文本以及前后关系。
    前驱和后继存储为列表中每个元素为字典：{'node_id': int, 'score': int}
    """

    def __init__(self, node_id: int, question: str, action: str, trajectory_text: str):
        """
        初始化轨迹库节点

        :param node_id: 节点唯一编号
        :param question: 节点对应的问题文本
        :param action: 节点动作类型，只能是 '思考'、'搜索'、'总结'、'观察'
        :param trajectory_text: 轨迹文本内容
        """
        # 基本属性
        self.id = node_id
        self.question = question
        self.action = action  # 允许的动作：思考、搜索、总结、观察
        self.trajectory_text = trajectory_text

        # 前驱关系：列表，每个元素为 {'node_id': int, 'score': int}
        self.predecessors = []

        # 后继关系：列表，每个元素为 {'node_id': int, 'score': int}
        self.successors = []

        # 重要性得分：所有前驱与后继节点重要性得分之和，初始为0
        self.importance_score = 0

    # --------------------- 基本属性的修改（改） ---------------------
    def set_question(self, new_question: str):
        """修改问题文本"""
        self.question = new_question

    def set_action(self, new_action: str):
        """
        修改动作类型，需检查是否为允许的值
        """
        allowed_actions = ['思考', '搜索', '总结', '观察']
        if new_action not in allowed_actions:
            raise ValueError(f"动作必须为 {allowed_actions} 之一")
        self.action = new_action

    def set_trajectory_text(self, new_text: str):
        """修改轨迹文本"""
        self.trajectory_text = new_text

    # --------------------- 前驱的增删改查 ---------------------
    def add_predecessor(self, pred_id: int, score: int):
        """
        增加一个前驱节点及其可信度分数

        :param pred_id: 前驱节点编号
        :param score: 前驱到当前节点的可信度分数（整数）
        """
        # 防止重复添加同一个前驱节点（若已存在则更新分数）
        self.remove_predecessor(pred_id)
        self.predecessors.append({'node_id': pred_id, 'score': score})

    def remove_predecessor(self, pred_id: int) -> bool:
        """
        删除指定编号的前驱节点

        :param pred_id: 要删除的前驱节点编号
        :return: 是否成功删除（找到并删除返回 True，否则 False）
        """
        for pred in self.predecessors:
            if pred['node_id'] == pred_id:
                self.predecessors.remove(pred)
                return True
        return False

    def update_predecessor_score(self, pred_id: int, new_score: int) -> bool:
        """
        修改指定前驱节点的可信度分数

        :param pred_id: 前驱节点编号
        :param new_score: 新的分数
        :return: 是否成功更新
        """
        for pred in self.predecessors:
            if pred['node_id'] == pred_id:
                pred['score'] = new_score
                return True
        return False

    def get_predecessors(self):
        """
        查询所有前驱节点信息

        :return: 包含前驱节点编号和分数的列表
        """
        return self.predecessors.copy()  # 返回副本，避免外部直接修改内部列表

    def clear_predecessors(self):
        """清空所有前驱关系"""
        self.predecessors.clear()

    # --------------------- 后继的增删改查 ---------------------
    def add_successor(self, succ_id: int, score: int):
        """
        增加一个后继节点及其可信度分数

        :param succ_id: 后继节点编号
        :param score: 当前节点到后继节点的可信度分数
        """
        self.remove_successor(succ_id)
        self.successors.append({'node_id': succ_id, 'score': score})

    def remove_successor(self, succ_id: int) -> bool:
        """
        删除指定编号的后继节点

        :param succ_id: 要删除的后继节点编号
        :return: 是否成功删除
        """
        for succ in self.successors:
            if succ['node_id'] == succ_id:
                self.successors.remove(succ)
                return True
        return False

    def update_successor_score(self, succ_id: int, new_score: int) -> bool:
        """
        修改指定后继节点的可信度分数

        :param succ_id: 后继节点编号
        :param new_score: 新的分数
        :return: 是否成功更新
        """
        for succ in self.successors:
            if succ['node_id'] == succ_id:
                succ['score'] = new_score
                return True
        return False

    def get_successors(self):
        """
        查询所有后继节点信息

        :return: 包含后继节点编号和分数的列表
        """
        return self.successors.copy()

    def clear_successors(self):
        """清空所有后继关系"""
        self.successors.clear()

    # --------------------- 重要性得分计算与更新 ---------------------
    def compute_importance_score(self, node_dict: dict) -> int:
        """
        根据当前节点所有前驱边和后继边的可信度分数（score），计算并更新当前节点的重要性得分。
        重要性得分 = 所有前驱边分数之和 + 所有后继边分数之和。

        :param node_dict: （已弃用）全局节点字典，保留仅为兼容旧版调用，不再使用
        :return: 计算后的重要性得分
        """
        total = 0
        # 累加所有入边的分数
        for pred_info in self.predecessors:
            total += pred_info['score']
        # 累加所有出边的分数
        for succ_info in self.successors:
            total += succ_info['score']
        self.importance_score = total
        return total

    # --------------------- 便捷的展示方法 ---------------------
    def __repr__(self):
        """返回节点的简要字符串表示"""
        return (f"TrajectoryNode(id={self.id}, action='{self.action}', "
                f"pred_count={len(self.predecessors)}, succ_count={len(self.successors)}, "
                f"importance={self.importance_score})")

    def detailed_info(self):
        """返回节点详细信息的字典"""
        return {
            'id': self.id,
            'question': self.question,
            'action': self.action,
            'trajectory_text': self.trajectory_text,
            'predecessors': self.predecessors.copy(),
            'successors': self.successors.copy(),
            'importance_score': self.importance_score
        }
        
    # track_node.py
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

# # ===================== 示例用法 =====================
# if __name__ == "__main__":
#     # 创建节点
#     node1 = TrajectoryNode(1, "初始问题", "思考", "开始分析用户输入...")
#     node2 = TrajectoryNode(2, "搜索步骤", "搜索", "检索相关文档...")
#     node3 = TrajectoryNode(3, "总结步骤", "总结", "综合信息生成答案...")

#     # 建立前后关系： node1 -> node2 (可信度85) -> node3 (可信度90)
#     node2.add_predecessor(1, 85)   # node2的前驱是node1
#     node3.add_predecessor(2, 90)   # node3的前驱是node2

#     node1.add_successor(2, 85)     # node1的后继是node2
#     node2.add_successor(3, 90)     # node2的后继是node3

#     # 查询节点信息
#     print("节点2的前驱：", node2.get_predecessors())   # [{'node_id': 1, 'score': 85}]
#     print("节点2的后继：", node2.get_successors())     # [{'node_id': 3, 'score': 90}]

#     # 修改前驱分数
#     node2.update_predecessor_score(1, 95)
#     print("更新后节点2的前驱分数：", node2.get_predecessors())  # score变为95

#     # 删除后继关系
#     node2.remove_successor(3)
#     print("删除后节点2的后继：", node2.get_successors())       # []

#     # 查看节点详细内容
#     print(node2.detailed_info())