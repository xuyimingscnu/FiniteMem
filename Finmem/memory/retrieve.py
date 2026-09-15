import os
import warnings
from typing import List, Dict, Optional, Set
import numpy as np
from collections import deque
from memory.knowsight_node import KnowsightNode   

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("请先安装 sentence-transformers 库：pip install sentence-transformers")

_SIMILARITY_THRESHOLD = 0.8
_KNOWSIGHT_SIMILARITY_THRESHOLD = 0.8

_current_library = None

def set_current_library(traj_lib):
    global _current_library
    _current_library = traj_lib

_DEVICE = ""

def set_device(device: str):
    global _DEVICE
    _DEVICE = device

def _get_device():
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

_MODEL = None
_MODEL_PATH = ""

def _get_model():
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"模型路径不存在: {_MODEL_PATH}")
        device = _get_device()
        _MODEL = SentenceTransformer(_MODEL_PATH, device=device)
    return _MODEL

# --------------------- 图环路检测辅助函数 ---------------------
def _has_path(graph: Dict[int, any], start_id: int, target_id: int) -> bool:
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
    global _current_library
    if _current_library is None:
        raise RuntimeError("未设置当前轨迹库，请先调用 set_current_library()")
    nodes_dict = _current_library._nodes

    candidate_nodes = [
        n for n in lib.nodes
        if n.question != node.question and n.id != node.id
    ]
    if not candidate_nodes:
        return []

    model = _get_model()
    query_vec = model.encode(node.trajectory_text, normalize_embeddings=True)

    candidate_texts = [n.trajectory_text for n in candidate_nodes]
    candidate_vecs = model.encode(candidate_texts, normalize_embeddings=True)

    similarities = (candidate_vecs @ query_vec).tolist() 

    scored_candidates = sorted(
        zip(candidate_nodes, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    selected_texts = []
    for cand_node, score in scored_candidates:
        if score < _SIMILARITY_THRESHOLD:
            break
        if _has_path(nodes_dict, node.id, cand_node.id) or \
           _has_path(nodes_dict, cand_node.id, node.id):
            continue

        selected_texts.append(cand_node.trajectory_text)
        if len(selected_texts) >= 3:
            break

    return selected_texts




def knowsight_retrieve(nodes: List[KnowsightNode], query_text: str) -> bool:
    if not nodes:
        return False

    model = _get_model()
    query_vec = model.encode(query_text, normalize_embeddings=True)

    texts = [node.get_knowsight_text() for node in nodes]
    node_vecs = model.encode(texts, normalize_embeddings=True)

    similarities = (node_vecs @ query_vec).tolist()

    return any(sim >= _KNOWSIGHT_SIMILARITY_THRESHOLD for sim in similarities)


def trajectory_mem_retrieve(lib, question: str, text: str):
    nodes = lib.nodes
    if not nodes:
        return []

    model = _get_model()

    traj_embs = [None] * len(nodes)          
    to_encode_texts = []                     
    to_encode_indices = []                   
    to_encode_nodes = []                     

    for i, node in enumerate(nodes):
        if node.trajectory_embedding is not None:
            traj_embs[i] = node.trajectory_embedding
        else:
            to_encode_texts.append(node.trajectory_text)
            to_encode_indices.append(i)
            to_encode_nodes.append(node)

    if to_encode_texts:
        new_embs = model.encode(to_encode_texts, normalize_embeddings=True)
        for idx, node, emb in zip(to_encode_indices, to_encode_nodes, new_embs):
            node.trajectory_embedding = emb   
            traj_embs[idx] = emb

    traj_embs = np.array(traj_embs)
    q_emb = model.encode(question, normalize_embeddings=True)
    sim_q = traj_embs @ q_emb  

    if text and text.strip():
        t_emb = model.encode(text, normalize_embeddings=True)
        sim_t = traj_embs @ t_emb
    else:
        sim_t = np.zeros(len(nodes))

    scores = np.array([node.importance_score for node in nodes])
    sorted_scores = np.sort(scores)
    ratios = np.searchsorted(sorted_scores, scores, side='left').astype(float) / len(nodes)

    alpha, beta, gamma = 0.2, 0.8, 0.0
    final_scores = alpha * sim_q + beta * sim_t + gamma * ratios

    
    
    top_indices = np.argsort(-final_scores)[:5]
    return [nodes[i].trajectory_text for i in top_indices]


def knowsight_mem_retrieve(nodes: List[KnowsightNode], query_text: str, k: int) -> List[str]:
    if not nodes:
        return []

    model = _get_model()

    to_encode_texts = []          
    to_encode_indices = []        
    to_encode_nodes_text = []     
    text_embs = [None] * len(nodes)

    for i, node in enumerate(nodes):
        if node.knowsight_text_embedding is not None:
            text_embs[i] = node.knowsight_text_embedding
        else:
            to_encode_texts.append(node.get_knowsight_text())
            to_encode_indices.append(i)
            to_encode_nodes_text.append(node)

    if to_encode_texts:
        new_embs = model.encode(to_encode_texts, normalize_embeddings=True)
        for idx, node, emb in zip(to_encode_indices, to_encode_nodes_text, new_embs):
            node.knowsight_text_embedding = emb
            text_embs[idx] = emb

    text_embs = np.array(text_embs)

    to_encode_q_texts = []        
    to_encode_q_indices = []      
    to_encode_q_nodes = []        
    q_embs = [None] * len(nodes)

    for i, node in enumerate(nodes):
        if node.question_embedding is not None:
            q_embs[i] = node.question_embedding
        else:
            to_encode_q_texts.append(node.get_question())
            to_encode_q_indices.append(i)
            to_encode_q_nodes.append(node)

    if to_encode_q_texts:
        new_q_embs = model.encode(to_encode_q_texts, normalize_embeddings=True)
        for idx, node, emb in zip(to_encode_q_indices, to_encode_q_nodes, new_q_embs):
            node.question_embedding = emb
            q_embs[idx] = emb

    q_embs = np.array(q_embs)
    query_emb = model.encode(query_text, normalize_embeddings=True)

    sim_knowsight = text_embs @ query_emb
    sim_question = q_embs @ query_emb

    scores = np.array([node.get_importance_score() for node in nodes])
    sorted_scores = np.sort(scores)
    ratios = np.searchsorted(sorted_scores, scores, side='left').astype(float) / len(nodes)

    alpha, beta, gamma = 0.2, 0.8, 0.0
    final_scores = alpha * sim_knowsight + beta * sim_question + gamma * ratios

    top_indices = np.argsort(-final_scores)[:k]
    return [nodes[i].get_knowsight_text() for i in top_indices]

