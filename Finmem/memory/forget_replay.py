import math
from typing import List

from memory.knowsight_node import KnowsightNode


def mem_replay(
    nodes: List[KnowsightNode],
    current_round: int,
    overflow_buffer: List[KnowsightNode]
) -> List[str]:
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

    retrieval_sums = []  
    non_empty_retrieval_sums_in_nodes = []

    for node in nodes:
        r_c = current_round
        retrieval_history = node.get_retrieval_history()
        if retrieval_history:
            sum_r = sum(1.0 / (r_c - r + epsilon) for r in retrieval_history)
            retrieval_sums.append(sum_r)
            non_empty_retrieval_sums_in_nodes.append(sum_r)
        else:
            retrieval_sums.append(0.0)

    empty_count_in_nodes = n_nodes - len(non_empty_retrieval_sums_in_nodes)

    if non_empty_retrieval_sums_in_nodes:
        avg_non_empty_sum = sum(non_empty_retrieval_sums_in_nodes) / n_nodes
    else:
        avg_non_empty_sum = 0.0

    for node in overflow_buffer:
        r_c = current_round
        retrieval_history = node.get_retrieval_history()
        if retrieval_history:
            sum_r = sum(1.0 / (r_c - r + epsilon) for r in retrieval_history)
        else:
            sum_r = 0.0

        if empty_count_in_nodes < n_overflow:
            retrieval_sums.append(avg_non_empty_sum)
        else:
            retrieval_sums.append(sum_r)

    importance_scores = [node.get_importance_score() for node in all_nodes]

    important_ratios = []
    for score in importance_scores:
        less_count = sum(1 for s in importance_scores if s < score)
        ratio = less_count / total_n
        important_ratios.append(ratio)

    scores = []
    for i, node in enumerate(all_nodes):
        r_c = current_round
        b = node.get_creation_round()
        term1 = 1.0 / (math.exp(r_c - b) + (1.0 - epsilon))
        term2 = retrieval_sums[i]
        term3 = important_ratios[i]

        S = alpha * term1 + beta * term2 + gamma * term3
        scores.append(S)
    sorted_indices = sorted(range(total_n), key=lambda i: scores[i])
    delete_indices = sorted_indices[:n_overflow]

    texts_to_delete = [all_nodes[i].get_knowsight_text() for i in delete_indices]
    return texts_to_delete