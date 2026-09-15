import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Dict, Any
import warnings
import os
from sklearn.cluster import KMeans
from track_node import TrajectoryNode

_DEFAULT_MODEL_PATH = os.environ.get("BGE_MODEL_PATH", "")
AVE_CLUSTER_BUFFER_NUM = 100
CLUSTERS_NUM = 5
_selector_instance = None

def get_selector(model_path: str = None, device: str = 'cuda') -> 'MemoryReplayBufferSelector':
    global _selector_instance
    if _selector_instance is None:
        path = model_path or _DEFAULT_MODEL_PATH
        _selector_instance = MemoryReplayBufferSelector(model_path=path, device=device)
    return _selector_instance


# ==================== 缓存编码工具函数 ====================
def encode_texts_cached(texts: List[str],
                        model: SentenceTransformer,
                        node_map: Optional[Dict[str, TrajectoryNode]] = None) -> np.ndarray:
    if not texts:
        return np.array([])

    if node_map is None:
        return model.encode(texts, convert_to_numpy=True)

    # 用于存放最终结果
    vectors = [None] * len(texts)
    missing_indices = []
    missing_texts = []

    for i, text in enumerate(texts):
        node = node_map.get(text)
        if node is not None and node.trajectory_embedding is not None:
            vectors[i] = node.trajectory_embedding
        else:
            missing_indices.append(i)
            missing_texts.append(text)

    if missing_texts:
        encoded = model.encode(missing_texts, convert_to_numpy=True)
        for j, i in enumerate(missing_indices):
            vec = encoded[j]
            vectors[i] = vec
            # 回存至节点（如果存在该节点）
            node = node_map.get(texts[i])
            if node is not None:
                node.trajectory_embedding = vec

    return np.array(vectors)


class MemoryReplayBufferSelector:

    def __init__(self, model_path: str, device: str = 'cuda', importance_weight: float = 10.0):
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"模型目录不存在: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        self.device = device
        self.importance_weight = importance_weight

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    @staticmethod
    def l2_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.linalg.norm(vec1 - vec2)

    @staticmethod
    def compute_pairwise_distances(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
        diff = vecs1[:, np.newaxis, :] - vecs2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def compute_A(self, v_vec: np.ndarray, buffer_vecs: np.ndarray,
                  exclude_self: bool = False, self_idx: Optional[int] = None) -> float:
        if buffer_vecs.shape[0] == 0:
            return 1e9
        if exclude_self and self_idx is not None:
            other_vecs = np.delete(buffer_vecs, self_idx, axis=0)
            if other_vecs.shape[0] == 0:
                return 0.0
            distances = np.linalg.norm(v_vec - other_vecs, axis=1)
        else:
            distances = np.linalg.norm(v_vec - buffer_vecs, axis=1)
        return np.min(distances)

    def compute_D(self, buffer_vecs: np.ndarray,
                  precomputed_dist_to_old: np.ndarray,
                  old_buffer_count: int) -> float:
        b = buffer_vecs.shape[0]
        if b == 0:
            return 0.0
        intra_sum = 0.0
        for i, v_vec in enumerate(buffer_vecs):
            intra_sum += self.compute_A(v_vec, buffer_vecs, exclude_self=True, self_idx=i)
        inter_weight = 1.0 / (old_buffer_count - 1) if old_buffer_count > 1 else 0.0
        inter_sum = 0.0
        if old_buffer_count > 0:
            inter_sum = np.sum(np.mean(precomputed_dist_to_old, axis=1))
        return intra_sum + inter_weight * inter_sum

    def init_first_sample(self, class_vecs: np.ndarray,
                          old_buffer_centers: List[np.ndarray]) -> int:
        if len(old_buffer_centers) == 0:
            return 0
        min_dists = []
        for v_vec in class_vecs:
            dists = [np.linalg.norm(v_vec - center) for center in old_buffer_centers]
            min_dists.append(np.min(dists))
        return int(np.argmax(min_dists))

    def greedy_fill(self, class_vecs: np.ndarray,
                    precomputed_dist_to_old: np.ndarray,
                    old_buffer_count: int,
                    buffer_size: int,
                    initial_idx: int,
                    importance_scores: Optional[np.ndarray] = None) -> List[int]:
        n_samples = class_vecs.shape[0]
        selected = [initial_idx]
        selected_set = set(selected)
        current_vecs = [class_vecs[initial_idx]]
        current_imp_sum = importance_scores[initial_idx] if importance_scores is not None else 0.0

        while len(selected) < buffer_size:
            best_gain = -np.inf
            best_idx = -1

            for i in range(n_samples):
                if i in selected_set:
                    continue

                candidate_vecs = current_vecs + [class_vecs[i]]
                candidate_vecs_array = np.array(candidate_vecs)
                candidate_indices = selected + [i]
                candidate_dist_to_old = precomputed_dist_to_old[candidate_indices, :]

                D_new = self.compute_D(candidate_vecs_array, candidate_dist_to_old, old_buffer_count)

                if importance_scores is not None:
                    imp_new = current_imp_sum + importance_scores[i]
                    imp_old = current_imp_sum
                    gain = (D_new + self.importance_weight * imp_new) - (self.compute_D(np.array(current_vecs),
                                                                                       precomputed_dist_to_old[selected, :],
                                                                                       old_buffer_count) + self.importance_weight * imp_old)
                else:
                    D_old = self.compute_D(np.array(current_vecs),
                                           precomputed_dist_to_old[selected, :],
                                           old_buffer_count)
                    gain = D_new - D_old

                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1:
                warnings.warn("无法找到能增加增益的样本，提前终止填充")
                break

            selected.append(best_idx)
            selected_set.add(best_idx)
            current_vecs.append(class_vecs[best_idx])
            if importance_scores is not None:
                current_imp_sum += importance_scores[best_idx]

        return selected

    def select_buffer_for_new_class(self,
                                    new_class_texts: List[str],
                                    old_buffers: List[List[str]],
                                    buffer_size: int,
                                    importance_scores: Optional[List[float]] = None,
                                    node_map: Optional[Dict[str, TrajectoryNode]] = None) -> Tuple[List[str], List[int]]:
        all_texts = new_class_texts
        all_vecs = encode_texts_cached(all_texts, self.model, node_map)
        n_samples = all_vecs.shape[0]

        old_buffer_vecs_list = []
        old_buffer_centers = []
        for buf in old_buffers:
            if len(buf) == 0:
                vecs = np.empty((0, all_vecs.shape[1]))
                center = np.zeros(all_vecs.shape[1])
            else:
                vecs = encode_texts_cached(buf, self.model, node_map)
                center = np.mean(vecs, axis=0)
            old_buffer_vecs_list.append(vecs)
            old_buffer_centers.append(center)
        K = len(old_buffers)

        precomputed_dist_to_old = np.zeros((n_samples, K))
        for j, buf_vecs in enumerate(old_buffer_vecs_list):
            if buf_vecs.shape[0] == 0:
                precomputed_dist_to_old[:, j] = 1e9
            else:
                dist_mat = self.compute_pairwise_distances(all_vecs, buf_vecs)
                precomputed_dist_to_old[:, j] = np.min(dist_mat, axis=1)

        init_idx = self.init_first_sample(all_vecs, old_buffer_centers)

        imp_arr = np.array(importance_scores) if importance_scores is not None else None
        selected_indices = self.greedy_fill(all_vecs, precomputed_dist_to_old, K,
                                            buffer_size, init_idx, imp_arr)

        selected_texts = [all_texts[i] for i in selected_indices]
        return selected_texts, selected_indices

def compute_importance_normalization(action_lib) -> Dict[str, float]:
    if not action_lib.nodes:
        return {}
    scores = [node.importance_score for node in action_lib.nodes]
    sorted_scores = sorted(scores)
    total = len(scores)
    text_to_score = {}
    for node in action_lib.nodes:
        rank = sum(1 for s in sorted_scores if s < node.importance_score)
        percentile = rank / total if total > 0 else 0.0
        text_to_score[node.trajectory_text] = percentile
    return text_to_score


def kmeans_split_texts(texts: List[str],
                       k: int = CLUSTERS_NUM,
                       max_per_class: int = AVE_CLUSTER_BUFFER_NUM,
                       model: Optional[SentenceTransformer] = None,
                       node_map: Optional[Dict[str, TrajectoryNode]] = None) -> List[List[str]]:
    if not texts:
        return []

    if model is None:
        model = get_selector().model

    total_required = k * max_per_class
    if len(texts) < total_required:
        raise ValueError(
            f"文本数量({len(texts)})不足以形成 {k} 个大小为 {max_per_class} 的类，"
            f"需要至少 {total_required} 条。请调整 k 或 max_per_class。"
        )

    vecs = encode_texts_cached(texts, model, node_map)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecs)
    centers = kmeans.cluster_centers_

    cluster_indices = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        cluster_indices[label].append(idx)

    final_clusters_texts = [[] for _ in range(k)]   
    overflow_pool = []                               

    for i in range(k):
        indices = cluster_indices[i]
        cluster_vecs = vecs[indices]
        dists = np.linalg.norm(cluster_vecs - centers[i], axis=1)
        sorted_order = np.argsort(dists)
        sorted_indices = [indices[j] for j in sorted_order]

        keep = sorted_indices[:max_per_class]
        final_clusters_texts[i] = [texts[idx] for idx in keep]
        if len(sorted_indices) > max_per_class:
            overflow_pool.extend(sorted_indices[max_per_class:])

    unfilled = []
    for i in range(k):
        deficit = max_per_class - len(final_clusters_texts[i])
        if deficit > 0:
            unfilled.append((i, deficit))

    if overflow_pool and unfilled:
        unfilled_indices = [uf[0] for uf in unfilled]
        unfilled_centers = centers[unfilled_indices]
        pool_vecs = vecs[overflow_pool]

        dist_mat = np.linalg.norm(
            pool_vecs[:, np.newaxis, :] - unfilled_centers[np.newaxis, :, :], axis=2
        )

        candidates = []
        for j, pool_idx in enumerate(overflow_pool):
            for m, uf_idx in enumerate(unfilled_indices):
                candidates.append((dist_mat[j, m], pool_idx, uf_idx))
        candidates.sort(key=lambda x: x[0])

        remaining = {uf_idx: deficit for uf_idx, deficit in unfilled}
        assigned_pool = set()
        for dist, pool_idx, uf_idx in candidates:
            if pool_idx in assigned_pool:
                continue
            if uf_idx in remaining and remaining[uf_idx] > 0:
                final_clusters_texts[uf_idx].append(texts[pool_idx])
                remaining[uf_idx] -= 1
                assigned_pool.add(pool_idx)
                if remaining[uf_idx] == 0:
                    del remaining[uf_idx]
            if not remaining:   
                break

    for i in range(k):
        if len(final_clusters_texts[i]) != max_per_class:
            warnings.warn(f"类 {i} 最终大小为 {len(final_clusters_texts[i])}，期望 {max_per_class}，可能存在数据不足或分配异常。")

    return final_clusters_texts


def memreplay(action_lib) -> Tuple[List[str], List[List[str]]]:
    text_to_node: Dict[str, TrajectoryNode] = {}
    for node in action_lib.nodes:
        if node.trajectory_text:   
            text_to_node[node.trajectory_text] = node

    selector = get_selector()
    
    embedding_dim = None
    for cls_texts in action_lib.cluster_buffer:
        if cls_texts:
            sample_text = cls_texts[0]
            sample_vec = encode_texts_cached([sample_text], selector.model, text_to_node)
            embedding_dim = sample_vec.shape[1]
            break
    if embedding_dim is None:
        for group in action_lib.overflow_buffer:
            if group:
                sample_text = group[0]
                sample_vec = encode_texts_cached([sample_text], selector.model, text_to_node)
                embedding_dim = sample_vec.shape[1]
                break
    if embedding_dim is None:
        return [], []

    cluster_buffer = action_lib.cluster_buffer
    if len(cluster_buffer) == 1 and len(cluster_buffer[0]) > 0:
        all_texts = cluster_buffer[0]
        new_clusters = kmeans_split_texts(
            all_texts, k=CLUSTERS_NUM, max_per_class=AVE_CLUSTER_BUFFER_NUM,
            model=selector.model, node_map=text_to_node
        )
        cluster_buffer = new_clusters
    else:
        cluster_buffer = [list(cls) for cls in cluster_buffer]

    overflow_texts = [text for group in action_lib.overflow_buffer for text in group]
    if not overflow_texts:
        return [], cluster_buffer

    class_centers = []
    for cls_texts in cluster_buffer:
        if cls_texts:
            vecs = encode_texts_cached(cls_texts, selector.model, text_to_node)
            center = np.mean(vecs, axis=0)
            class_centers.append(center)
        else:
            class_centers.append(np.zeros(embedding_dim))
    class_centers = np.array(class_centers)

    overflow_vecs = encode_texts_cached(overflow_texts, selector.model, text_to_node)
    total_distances = []
    for center in class_centers:
        dists = np.linalg.norm(overflow_vecs - center, axis=1)
        total_distances.append(np.sum(dists))
    nearest_class_idx = int(np.argmin(total_distances))

    moved_class_texts = cluster_buffer.pop(nearest_class_idx)
    new_buffer_texts = moved_class_texts + overflow_texts

    imp_map = compute_importance_normalization(action_lib)
    importance_scores = [imp_map.get(text, 0.0) for text in new_buffer_texts]

    selected_texts, _ = selector.select_buffer_for_new_class(
        new_class_texts=new_buffer_texts,
        old_buffers=cluster_buffer,          
        buffer_size=AVE_CLUSTER_BUFFER_NUM,
        importance_scores=importance_scores,
        node_map=text_to_node
    )

    to_delete = [text for text in new_buffer_texts if text not in selected_texts]

    cluster_buffer.append(selected_texts)

    return to_delete, cluster_buffer
