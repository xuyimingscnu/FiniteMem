import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Dict, Any
import warnings
import os
from sklearn.cluster import KMeans

"""
# 模拟数据：一些中文文本（实际使用时替换为真实数据）
# 旧类别缓冲区（已存在）
old_buffers = [
    ["苹果很好吃", "我喜欢吃苹果"],          # 类别1
    ["今天天气不错", "阳光明媚"]             # 类别2
]

# 新增类别样本集
new_classes_texts = [
    ["机器学习很有趣", "深度学习是AI的分支", "神经网络模拟人脑"],   # 新类别3
    ["北京是首都", "上海是金融中心", "广州有美食"]               # 新类别4
]
"""


# ==================== 全局模型管理 ====================
_DEFAULT_MODEL_PATH = os.environ.get("BGE_MODEL_PATH", "/home/yiming/hzk/baseModel/bge-large-zh-v1.5/")
AVE_CLUSTER_BUFFER_NUM = 100
_selector_instance = None

def get_selector(model_path: str = None, device: str = 'cuda') -> 'MemoryReplayBufferSelector':
    """获取全局单例选择器（避免重复加载模型）"""
    global _selector_instance
    if _selector_instance is None:
        path = model_path or _DEFAULT_MODEL_PATH
        _selector_instance = MemoryReplayBufferSelector(model_path=path, device=device)
    return _selector_instance

# ==================== 原有 MemoryReplayBufferSelector 扩展 ====================
class MemoryReplayBufferSelector:
    """
    基于记忆重放算法的缓冲区选择器（支持重要性加权）
    """

    def __init__(self, model_path: str, device: str = 'cuda', importance_weight: float = 10.0):
        """
        :param model_path: 本地 BGE 模型目录路径
        :param device: 运行设备，如 'cpu' 或 'cuda'
        :param importance_weight: 重要性得分的权重系数（用于平衡多样性与重要性）
        """
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"模型目录不存在: {model_path}")
        self.model = SentenceTransformer(model_path, device=device)
        self.device = device
        self.importance_weight = importance_weight

    def encode(self, texts: List[str]) -> np.ndarray:
        """将文本列表转换为向量（numpy 数组）"""
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
        """
        贪心填充缓冲区，支持重要性加权
        :param importance_scores: shape (n_samples,) 每个样本的重要性得分（0~1），None 表示不使用重要性
        """
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

                # 模拟候选缓冲区
                candidate_vecs = current_vecs + [class_vecs[i]]
                candidate_vecs_array = np.array(candidate_vecs)
                candidate_indices = selected + [i]
                candidate_dist_to_old = precomputed_dist_to_old[candidate_indices, :]

                D_new = self.compute_D(candidate_vecs_array, candidate_dist_to_old, old_buffer_count)

                # 计算重要性增益
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
                                    importance_scores: Optional[List[float]] = None) -> Tuple[List[str], List[int]]:
        """
        为单个新类别选择缓冲区（支持重要性加权）
        :param importance_scores: 与 new_class_texts 顺序对应的重要性得分列表（0~1），None 表示不使用
        """
        all_texts = new_class_texts
        all_vecs = self.encode(all_texts)
        n_samples = all_vecs.shape[0]

        # 编码旧缓冲区并计算中心
        old_buffer_vecs_list = []
        old_buffer_centers = []
        for buf in old_buffers:
            if len(buf) == 0:
                vecs = np.empty((0, all_vecs.shape[1]))
                center = np.zeros(all_vecs.shape[1])
            else:
                vecs = self.encode(buf)
                center = np.mean(vecs, axis=0)
            old_buffer_vecs_list.append(vecs)
            old_buffer_centers.append(center)
        K = len(old_buffers)

        # 预计算每个新样本到每个旧缓冲区（最近邻）的距离矩阵
        precomputed_dist_to_old = np.zeros((n_samples, K))
        for j, buf_vecs in enumerate(old_buffer_vecs_list):
            if buf_vecs.shape[0] == 0:
                precomputed_dist_to_old[:, j] = 1e9
            else:
                dist_mat = self.compute_pairwise_distances(all_vecs, buf_vecs)
                precomputed_dist_to_old[:, j] = np.min(dist_mat, axis=1)

        # 初始化第一个样本
        init_idx = self.init_first_sample(all_vecs, old_buffer_centers)

        # 贪心填充（传入重要性得分数组）
        imp_arr = np.array(importance_scores) if importance_scores is not None else None
        selected_indices = self.greedy_fill(all_vecs, precomputed_dist_to_old, K,
                                            buffer_size, init_idx, imp_arr)

        selected_texts = [all_texts[i] for i in selected_indices]
        return selected_texts, selected_indices

# ==================== 辅助函数 ====================
def compute_importance_normalization(action_lib) -> Dict[str, float]:
    """
    计算动作库中每个节点文本的重要性归一化得分（百分位数）
    返回：{ trajectory_text: normalized_score (0~1) }
    """
    if not action_lib.nodes:
        return {}
    scores = [node.importance_score for node in action_lib.nodes]
    sorted_scores = sorted(scores)
    total = len(scores)
    # 计算百分位数：分数严格小于当前值的比例
    text_to_score = {}
    for node in action_lib.nodes:
        rank = sum(1 for s in sorted_scores if s < node.importance_score)
        percentile = rank / total if total > 0 else 0.0
        text_to_score[node.trajectory_text] = percentile
    return text_to_score

import numpy as np
from sklearn.cluster import KMeans
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import warnings

# 假设此函数位于模块内，能访问 AVE_CLUSTER_BUFFER_NUM 和 get_selector
AVE_CLUSTER_BUFFER_NUM = 100   # 示例，实际应与模块内一致

def kmeans_split_texts(texts: List[str],
                       k: int = 5,
                       max_per_class: int = AVE_CLUSTER_BUFFER_NUM,
                       model: Optional[SentenceTransformer] = None) -> List[List[str]]:
    """
    使用 k-means 将文本分成 k 个类，每类强制包含 max_per_class 个样本。
    数据不足时抛出异常；不好聚类的数据（距离中心较远）会被重新分配到距离最近的未满类。

    :param texts: 输入文本列表
    :param k: 聚类数
    :param max_per_class: 每类固定样本数
    :param model: SentenceTransformer 模型实例，若为 None 则使用全局单例
    :return: 长度为 k 的列表，每个元素为长度等于 max_per_class 的文本列表
    """
    if not texts:
        return []

    # if model is None:
    #     # 依赖模块内 get_selector 单例
    #     from . import get_selector   # 注意：实际路径需根据项目结构调整
    #     model = get_selector().model

    total_required = k * max_per_class
    if len(texts) < total_required:
        raise ValueError(
            f"文本数量({len(texts)})不足以形成 {k} 个大小为 {max_per_class} 的类，"
            f"需要至少 {total_required} 条。请调整 k 或 max_per_class。"
        )

    vecs = model.encode(texts, convert_to_numpy=True)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vecs)
    centers = kmeans.cluster_centers_

    # ---------- 1. 初始簇整理 ----------
    cluster_indices = [[] for _ in range(k)]
    for idx, label in enumerate(labels):
        cluster_indices[label].append(idx)

    final_clusters_texts = [[] for _ in range(k)]   # 最终各类的文本
    overflow_pool = []                               # 存放被移出的样本索引

    for i in range(k):
        indices = cluster_indices[i]
        # 类内按到本簇中心的距离排序
        cluster_vecs = vecs[indices]
        dists = np.linalg.norm(cluster_vecs - centers[i], axis=1)
        sorted_order = np.argsort(dists)
        sorted_indices = [indices[j] for j in sorted_order]

        # 保留最近的 max_per_class 个
        keep = sorted_indices[:max_per_class]
        final_clusters_texts[i] = [texts[idx] for idx in keep]
        # 多余样本进入溢出池
        if len(sorted_indices) > max_per_class:
            overflow_pool.extend(sorted_indices[max_per_class:])

    # ---------- 2. 用溢出池填补未满的类 ----------
    # 找出未满的类及其缺额
    unfilled = []
    for i in range(k):
        deficit = max_per_class - len(final_clusters_texts[i])
        if deficit > 0:
            unfilled.append((i, deficit))

    if overflow_pool and unfilled:
        unfilled_indices = [uf[0] for uf in unfilled]
        unfilled_centers = centers[unfilled_indices]
        pool_vecs = vecs[overflow_pool]

        # 距离矩阵：池中每个点到每个未满类中心的距离 (pool_size × num_unfilled)
        dist_mat = np.linalg.norm(
            pool_vecs[:, np.newaxis, :] - unfilled_centers[np.newaxis, :, :], axis=2
        )

        # 构建候选列表并按距离升序排序
        candidates = []
        for j, pool_idx in enumerate(overflow_pool):
            for m, uf_idx in enumerate(unfilled_indices):
                candidates.append((dist_mat[j, m], pool_idx, uf_idx))
        candidates.sort(key=lambda x: x[0])

        # 贪心分配：每个点分配给距离最近且仍有空位的类
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
            if not remaining:   # 所有类已满
                break

    # ---------- 3. 最终验证 ----------
    for i in range(k):
        if len(final_clusters_texts[i]) != max_per_class:
            warnings.warn(f"类 {i} 最终大小为 {len(final_clusters_texts[i])}，期望 {max_per_class}，可能存在数据不足或分配异常。")

    return final_clusters_texts

# ==================== 主接口 memreplay ====================
def memreplay(action_lib) -> Tuple[List[str], List[List[str]]]:
    """
    轨迹库重放算法主函数
    :param action_lib: ActionLibrary 对象（包含 cluster_buffer, overflow_buffer, nodes）
    :return: (待删除文本列表, 新的聚类缓冲区)
    """
    # 1. 处理聚类缓冲区只有一个类的情况
    cluster_buffer = action_lib.cluster_buffer
    if len(cluster_buffer) == 1 and len(cluster_buffer[0]) > 0:
        # 将该类的所有文本提取出来
        all_texts = cluster_buffer[0]
        # k-means 分成 5 类（每类最多20）
        new_clusters = kmeans_split_texts(all_texts, k=5, max_per_class=AVE_CLUSTER_BUFFER_NUM, model=get_selector().model)
        cluster_buffer = new_clusters
        # 更新动作库的聚类缓冲区（后续会赋值）
    else:
        # 深拷贝一份，避免直接修改原引用
        cluster_buffer = [list(cls) for cls in cluster_buffer]

    # 2. 处理溢出缓冲区
    overflow_texts = [text for group in action_lib.overflow_buffer for text in group]
    if not overflow_texts:
        # 无溢出数据，直接返回
        return [], cluster_buffer

    # 3. 计算当前聚类缓冲区每个类的中心
    selector = get_selector()
    class_centers = []
    for cls_texts in cluster_buffer:
        if cls_texts:
            vecs = selector.encode(cls_texts)
            center = np.mean(vecs, axis=0)
            class_centers.append(center)
        else:
            class_centers.append(np.zeros(selector.model.get_sentence_embedding_dimension()))
    class_centers = np.array(class_centers)

    # 4. 计算溢出文本到每个类中心的距离总和，选择最近的类
    overflow_vecs = selector.encode(overflow_texts)
    total_distances = []
    for center in class_centers:
        dists = np.linalg.norm(overflow_vecs - center, axis=1)
        total_distances.append(np.sum(dists))
    nearest_class_idx = int(np.argmin(total_distances))

    # 5. 将该类移出聚类缓冲区，作为新缓冲区
    moved_class_texts = cluster_buffer.pop(nearest_class_idx)
    new_buffer_texts = moved_class_texts + overflow_texts

    # 6. 计算重要性分数（用于重放）
    imp_map = compute_importance_normalization(action_lib)
    # 为新缓冲区中的每个文本获取重要性（缺失则给0）
    importance_scores = [imp_map.get(text, 0.0) for text in new_buffer_texts]

    # 7. 调用重放方法：old_buffers 是剩余的聚类缓冲区，new_class 是新缓冲区
    # MAX_CLASS_SIZE = 20
    selected_texts, _ = selector.select_buffer_for_new_class(
        new_class_texts=new_buffer_texts,
        old_buffers=cluster_buffer,          # 剩余类作为旧缓冲区
        buffer_size=AVE_CLUSTER_BUFFER_NUM,
        importance_scores=importance_scores
    )

    # 8. 待删除文本 = 新缓冲区中未被选中的
    to_delete = [text for text in new_buffer_texts if text not in selected_texts]

    # 9. 更新聚类缓冲区：将选中的文本作为一个新类加入
    cluster_buffer.append(selected_texts)

    return to_delete, cluster_buffer

# ==================== 保留原有示例代码（可选） ====================
# if __name__ == "__main__":
#     # 原有示例可保留，但需要适配新的接口
#     pass