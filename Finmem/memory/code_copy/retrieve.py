"""
轨迹检索模块

本模块提供基于语义相似度的轨迹节点检索功能，用于扩展推理图中的前驱/后继关系。

检索时需要注意：
1.只能检索相同动作库的文本，思考只能检索思考，总结只能检索总结
2.不能检索同问题下的动作，同一个问题的文本肯定具有高相似性，检索无意义

"""

import os
import warnings
from typing import List, Dict, Optional, Set
import numpy as np
from collections import deque
from memory.knowsight_node import KnowsightNode   # 确保引入节点类型

# 尝试导入 sentence-transformers，若未安装则给出友好提示
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("请先安装 sentence-transformers 库：pip install sentence-transformers")

# --------------------- 全局上下文管理 ---------------------
# 为了在保持 action_retrieve 签名不变的前提下获取全局节点字典，
# 使用一个全局变量来暂存当前 TrajectoryLibrary 实例的引用。

# 相似度阈值，仅当语义相似度 >= 该值时，候选节点才会被纳入结果列表
_SIMILARITY_THRESHOLD = 0.8
# 洞察-知识库的相似度阈值（可配置，默认0.8）
_KNOWSIGHT_SIMILARITY_THRESHOLD = 0.8

_current_library = None

def set_current_library(traj_lib):
    """
    设置当前轨迹库实例，供检索时获取全局节点图。

    应在 TrajectoryLibrary.build_from_input 开始时调用。
    """
    global _current_library
    _current_library = traj_lib

# --------------------- 设备配置 ---------------------
_DEVICE = "cuda:1"

def set_device(device: str):
    """
    设置编码模型使用的计算设备。

    :param device: 设备字符串，如 'cuda', 'cuda:0', 'cpu'
    """
    global _DEVICE
    _DEVICE = device

def _get_device():
    """获取当前设备，若未设置则自动检测 cuda 可用性，默认优先使用 GPU"""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            _DEVICE = 'cuda'
        else:
            _DEVICE = 'cpu'
            warnings.warn("CUDA 不可用，将使用 CPU 进行编码，速度可能较慢。")
    except ImportError:
        _DEVICE = 'cpu'
        warnings.warn("未安装 torch，将使用 CPU 进行编码。")
    return _DEVICE

# --------------------- 模型加载与缓存 ---------------------
_MODEL = None
_MODEL_PATH = "/home/yiming/hzk/baseModel/bge-large-zh-v1.5/"

def _get_model():
    """加载并返回 BGE 模型（单例模式），模型将移动到指定设备"""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"模型路径不存在: {_MODEL_PATH}")
        device = _get_device()
        _MODEL = SentenceTransformer(_MODEL_PATH, device=device)
    return _MODEL

# --------------------- 图环路检测辅助函数 ---------------------
def _has_path(graph: Dict[int, any], start_id: int, target_id: int) -> bool:
    """
    检测有向图中是否存在从 start_id 到 target_id 的路径（通过后继关系）。

    :param graph: 全局节点字典 {node_id: TrajectoryNode}
    :param start_id: 起始节点ID
    :param target_id: 目标节点ID
    :return: 若可达返回 True，否则 False
    """
    if start_id == target_id:
        return True
    visited: Set[int] = set()
    queue = deque([start_id])
    visited.add(start_id)

    while queue:
        cur_id = queue.popleft()
        cur_node = graph.get(cur_id)
        if not cur_node:
            continue
        for succ_info in cur_node.get_successors():
            succ_id = succ_info['node_id']
            if succ_id == target_id:
                return True
            if succ_id not in visited:
                visited.add(succ_id)
                queue.append(succ_id)
    return False

# --------------------- 核心检索函数 ---------------------

def action_retrieve(lib, node) -> List[str]:
    """
    在指定动作库中检索与给定节点语义相似度最高的三个轨迹文本，
    同时排除可能导致推理环路的候选节点，并且仅保留相似度不低于 0.8 的结果。

    :param lib: ActionLibrary 实例
    :param node: TrajectoryNode 实例（查询节点）
    :return: 相似轨迹文本列表，最多 3 个元素
    """
    # 1. 获取全局节点字典（用于环路检测）
    global _current_library
    if _current_library is None:
        raise RuntimeError("未设置当前轨迹库，请先调用 set_current_library()")
    nodes_dict = _current_library._nodes

    # 2. 筛选候选节点：同一动作库内、不同问题、非自身
    candidate_nodes = [
        n for n in lib.nodes
        if n.question != node.question and n.id != node.id
    ]
    if not candidate_nodes:
        return []

    # 3. 计算查询节点向量（已归一化）
    model = _get_model()
    query_vec = model.encode(node.trajectory_text, normalize_embeddings=True)

    # 4. 批量计算候选节点向量并计算余弦相似度（归一化后点积即为余弦相似度）
    candidate_texts = [n.trajectory_text for n in candidate_nodes]
    candidate_vecs = model.encode(candidate_texts, normalize_embeddings=True)

    # 计算相似度分数
    similarities = (candidate_vecs @ query_vec).tolist()  # 矩阵乘法得到余弦相似度列表

    # 5. 按相似度降序排序，并附加节点索引
    scored_candidates = sorted(
        zip(candidate_nodes, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    # 6. 依次选取满足环路约束且相似度达标的节点，直至凑满 3 个
    selected_texts = []
    for cand_node, score in scored_candidates:
        # 相似度不足 0.8 时，后续候选分数更低，无需继续检查
        if score < _SIMILARITY_THRESHOLD:
            break

        # 检查是否会形成循环：
        # 若从 node 可到达 cand_node，或从 cand_node 可到达 node，
        # 则添加关系可能导致推理环路，予以排除。
        if _has_path(nodes_dict, node.id, cand_node.id) or \
           _has_path(nodes_dict, cand_node.id, node.id):
            continue

        selected_texts.append(cand_node.trajectory_text)
        if len(selected_texts) >= 3:
            break

    return selected_texts




def knowsight_retrieve(nodes: List[KnowsightNode], query_text: str) -> bool:
    """
    在洞察/知识节点列表中检索是否存在与查询文本语义相似（余弦相似度 >= 0.8）的节点。

    :param nodes: KnowsightNode 实例列表
    :param query_text: 待检索的文本
    :return: 若存在相似节点返回 True，否则返回 False
    """
    if not nodes:
        return False

    model = _get_model()
    query_vec = model.encode(query_text, normalize_embeddings=True)

    # 提取所有节点的文本并批量编码
    texts = [node.get_knowsight_text() for node in nodes]
    node_vecs = model.encode(texts, normalize_embeddings=True)

    # 计算余弦相似度（归一化后点积即余弦相似度）
    similarities = (node_vecs @ query_vec).tolist()

    return any(sim >= _KNOWSIGHT_SIMILARITY_THRESHOLD for sim in similarities)


def trajectory_mem_retrieve(lib, question: str, text: str):
    """
    在指定动作库中检索与给定问题和待检索文本最相似的轨迹文本。

    相似度综合得分公式：
        S = α * sim(traj_text, question) + β * sim(traj_text, text) + γ * importance_ratio
    其中：
        α = 0.2, β = 0.4, γ = 0.4
        importance_ratio = 节点重要性得分超过动作库中其它节点的比例（值域 [0, 1)）

    :param lib: ActionLibrary 实例
    :param question: 当前问题文本
    :param text: 待检索的文本，可能为空字符串
    :return: 按综合得分降序排列的轨迹文本列表，最多返回 5 个
    """
    nodes = lib.nodes
    if not nodes:
        return []

    model = _get_model()

    # 1. 批量编码所有节点的 trajectory_text
    traj_texts = [node.trajectory_text for node in nodes]
    traj_embs = model.encode(traj_texts, normalize_embeddings=True)

    # 2. 编码问题并计算与各节点的相似度
    q_emb = model.encode(question, normalize_embeddings=True)
    sim_q = traj_embs @ q_emb  # 归一化后点积即余弦相似度

    # 3. 编码待检索文本（若为空则相似度置零，避免无效编码）
    if text and text.strip():
        t_emb = model.encode(text, normalize_embeddings=True)
        sim_t = traj_embs @ t_emb
    else:
        sim_t = np.zeros(len(nodes))

    # 4. 计算重要性得分比例：该节点重要性超过动作库中多少比例的节点
    scores = np.array([node.importance_score for node in nodes])
    sorted_scores = np.sort(scores)
    # searchsorted(side='left') 返回严格小于该分数的节点个数
    ratios = np.searchsorted(sorted_scores, scores, side='left').astype(float) / len(nodes)

    # 5. 综合得分
    alpha, beta, gamma = 0.3, 0.5, 0.2
    final_scores = alpha * sim_q + beta * sim_t + gamma * ratios

    # 6. 取前5个最高分节点
    top_indices = np.argsort(-final_scores)[:5]
    return [nodes[i].trajectory_text for i in top_indices]


def knowsight_mem_retrieve(nodes: List[KnowsightNode], query_text: str, k: int) -> List[str]:
    """
    在洞察/知识节点列表中检索与查询文本最相似的 knowsight_text。

    综合得分 S = α·sim(knowsight_text, query_text) + β·importance_ratio
    其中 importance_ratio 是该节点重要性分数超过列表中所有节点重要性分数的比例。
    α = 0.6, β = 0.4

    :param nodes: KnowsightNode 列表
    :param query_text: 待检索的文本
    :param k: 需要返回的最相似文本数量
    :return: 按综合得分降序排列的 knowsight_text 列表，不足 k 个时返回全部
    """
    if not nodes:
        return []

    model = _get_model()

    # 提取所有节点的语义文本并编码（归一化，便于点积直接得到余弦相似度）
    texts = [node.get_knowsight_text() for node in nodes]
    text_embs = model.encode(texts, normalize_embeddings=True)
    query_emb = model.encode(query_text, normalize_embeddings=True)

    # 余弦相似度（归一化后点积）
    sim_scores = text_embs @ query_emb

    # 重要性得分比重：严格小于该节点分数的节点数占总数的比例
    scores = np.array([node.get_importance_score() for node in nodes])
    sorted_scores = np.sort(scores)
    ratios = np.searchsorted(sorted_scores, scores, side='left').astype(float) / len(nodes)

    # 加权综合得分
    alpha, beta = 0.8, 0.2
    final_scores = alpha * sim_scores + beta * ratios

    # 取 top‑k（不足 k 个时全部保留）
    top_indices = np.argsort(-final_scores)[:k]
    return [nodes[i].get_knowsight_text() for i in top_indices]