import math
from typing import List

from memory.knowsight_node import KnowsightNode


def mem_replay(
    nodes: List[KnowsightNode],
    current_round: int,
    overflow_buffer: List[KnowsightNode]
) -> List[str]:
    """
    基于遗忘的重放算法，结合溢出缓存区，选择重要性得分最低的多个节点，
    返回其文本内容列表用于删除。

    Args:
        nodes: 当前洞察库中的所有节点列表。
        current_round: 当前对话轮次数。
        overflow_buffer: 溢出缓存区中的节点列表。

    Returns:
        需要删除的节点的 knowsight_text 列表，数量等于溢出缓存区节点数量。
        若总节点数为0，则返回空列表。
    """
    # 合并所有待评估节点（原有节点 + 溢出缓存区节点）
    all_nodes = nodes + overflow_buffer
    if not all_nodes:
        return []

    # 超参数
    alpha = 0.1
    beta = 0.5
    gamma = 0.4
    epsilon = 1e-6

    n_nodes = len(nodes)
    n_overflow = len(overflow_buffer)
    total_n = len(all_nodes)

    # ---------- 预处理：计算每个节点的第二项（检索强化项）原始值 ----------
    retrieval_sums = []  # 与 all_nodes 顺序对应
    # 分别记录 nodes 中非空检索历史的 term2 值，用于计算平均值
    non_empty_retrieval_sums_in_nodes = []

    # 先计算 nodes 中的节点
    for node in nodes:
        r_c = current_round
        retrieval_history = node.get_retrieval_history()
        if retrieval_history:
            sum_r = sum(1.0 / (r_c - r + epsilon) for r in retrieval_history)
            retrieval_sums.append(sum_r)
            non_empty_retrieval_sums_in_nodes.append(sum_r)
        else:
            retrieval_sums.append(0.0)

    # 统计 nodes 中检索历史为空的节点数量
    empty_count_in_nodes = n_nodes - len(non_empty_retrieval_sums_in_nodes)

    # 计算非空节点的平均值（若没有非空节点，则平均值为 0.0）
    if non_empty_retrieval_sums_in_nodes:
        avg_non_empty_sum = sum(non_empty_retrieval_sums_in_nodes) / n_nodes
    else:
        avg_non_empty_sum = 0.0

    # 处理溢出缓存区节点
    for node in overflow_buffer:
        r_c = current_round
        retrieval_history = node.get_retrieval_history()
        if retrieval_history:
            sum_r = sum(1.0 / (r_c - r + epsilon) for r in retrieval_history)
        else:
            sum_r = 0.0

        # 特殊处理：当 nodes 中空检索历史节点数量少于溢出缓存区节点数量时，
        # 将溢出缓存区节点的检索强化项替换为平均值
        if empty_count_in_nodes < n_overflow:
            retrieval_sums.append(avg_non_empty_sum)
        else:
            retrieval_sums.append(sum_r)

    # ---------- 计算重要性分数的归一化比重（超过的比例）----------
    importance_scores = [node.get_importance_score() for node in all_nodes]

    important_ratios = []
    for score in importance_scores:
        less_count = sum(1 for s in importance_scores if s < score)
        ratio = less_count / total_n
        important_ratios.append(ratio)

    # ---------- 计算最终得分 S ----------
    scores = []
    for i, node in enumerate(all_nodes):
        r_c = current_round
        b = node.get_creation_round()

        # 第一项：时间衰减项
        term1 = 1.0 / (math.exp(r_c - b) + (1.0 - epsilon))

        # 第二项：检索强化项（已预处理）
        term2 = retrieval_sums[i]

        # 第三项：重要性分数比重项
        term3 = important_ratios[i]

        S = alpha * term1 + beta * term2 + gamma * term3
        scores.append(S)

    # ---------- 选择得分最低的 n 个节点 ----------
    # 按得分升序排序，取前 n_overflow 个
    sorted_indices = sorted(range(total_n), key=lambda i: scores[i])
    delete_indices = sorted_indices[:n_overflow]

    # 提取对应节点的 knowsight_text
    texts_to_delete = [all_nodes[i].get_knowsight_text() for i in delete_indices]
    return texts_to_delete