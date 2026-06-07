"""
混合检索模块

实现「BM25 关键词检索 + Chroma 向量检索 → RRF 融合 → CrossEncoder 重排序」三级流水线。

整体设计思路：
- 第一路（BM25）：基于关键词的**稀疏检索**，擅长精确匹配专有名词（如"知母""六味地黄丸"）
- 第二路（Chroma 向量）：基于语义的**密集检索**，擅长理解意图相近但表达不同的查询
- 两路结果通过 RRF（Reciprocal Rank Fusion）按排名加权合并
- 最后用 CrossEncoder 深度模型对合并结果逐对打分，做精细重排序

"""

import os
import warnings

# 必须在所有 HuggingFace/transformers 导入之前设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_TOKEN"] = ""
warnings.filterwarnings("ignore", message="The Transformer.*cache_dir.*deprecated")

from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

# ──────────────────────────────────────────────
# 全局重排序模型（延迟加载）

_RERANKER = None

def get_reranker() -> CrossEncoder:
    """获取重排序模型（延迟初始化，首次调用时加载）"""
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            cache_folder=get_abs_path("model"),
        )
    return _RERANKER


class HybridRetriever:
    """
    混合检索器：融合 BM25 关键词检索 + Chroma 向量检索 + CrossEncoder 重排序。

    使用方式：
        retriever = HybridRetriever(vector_store=chroma_instance, source_type="herb")
        docs = retriever.retrieve("知母的功效是什么？")
    """

    def __init__(
        self,
        vector_store: Chroma,
        source_type: Optional[str] = None,
        top_k: int = 3,
    ):
        """
        :param vector_store: Chroma 向量库实例（由 VectorStoreService 暴露的 .vector_store 属性传入）
        :param source_type: 来源类型过滤，可选 "herb"/"prescription"/"symptom"，None 表示检索全部
        :param top_k: 最终返回给 LLM 的文档数量
        """
        self.vector_store = vector_store
        self.source_type = source_type
        self.top_k = top_k

        # 候选数量 = 最终 K 值的 3 倍
        # 为什么要多召回？
        #   BM25 和向量检索各自召回 top-9，两路合并后约有 12-15 篇不重复文档，
        #   给 CrossEncoder 提供足够多的候选进行精细排序，避免好文档被早截断。
        self._candidate_k = top_k * 3

        # ──────────── 1. 从 Chroma 提取所有文档块 ────────────
        # BM25 需要完整的文档文本才能构建倒排索引，不能直接使用 Chroma 的向量检索结果。
        # 所以额外调用 vector_store.get() 提取所有已入库的文本块。
        all_docs = self._load_all_docs()

        # ──────────── 2. BM25 检索器（关键词匹配）────────────
        # BM25 是一种基于词频（TF）和逆文档频率（IDF）的经典检索算法。
        # 优点：对专有名词（药材名、方剂名）的精确匹配效果极好。
        # 缺点：对同义词、语义相近的表达不敏感。
        self.bm25 = BM25Retriever.from_documents(all_docs, k=self._candidate_k) if all_docs else None

        # ──────────── 3. 向量检索器（语义匹配）────────────
        # 将查询转为 embedding 向量，在 Chroma 中做余弦相似度搜索。
        # 优点：能理解"口干舌燥"和"阴液不足"之间的语义关联。
        # 缺点：对冷门专有名词的匹配不如 BM25 精确。
        search_kwargs = {"k": self._candidate_k}
        if source_type:
            # 按来源类型过滤，例如只检索"herb"类型的文档
            search_kwargs["filter"] = {"source_type": source_type}
        self.vector = vector_store.as_retriever(search_kwargs=search_kwargs)

    # ──────────────────────────────────────────────
    # 内部方法：从 Chroma 提取文本
    # ──────────────────────────────────────────────

    def _load_all_docs(self) -> List[Document]:
        """
        从 Chroma 数据库中提取所有已入库的文本块，用于构建 BM25 检索器。

        Chroma 的 get() 方法返回结构：
        {
            "documents": ["文本1", "文本2", ...],
            "metadatas": [{"source_type": "herb", ...}, ...],
            "ids": ["id1", "id2", ...],
        }

        :return: Document 对象列表
        """
        raw = self.vector_store.get()  # 一次性获取所有文档
        docs = []
        for i, text in enumerate(raw.get("documents", [])):
            # 安全性处理：如果某个文档没有 metadata，用空字典兜底
            meta = raw.get("metadatas", [{}])[i] or {}

            # 如果指定了 source_type 过滤，跳过不匹配的类型
            # 例如只检索"herb"时，跳过"prescription"的文档
            if self.source_type and meta.get("source_type") != self.source_type:
                continue

            docs.append(Document(page_content=text, metadata=meta))
        return docs

    # ──────────────────────────────────────────────
    # RRF 融合方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(
        bm25_results: List[Document],
        vector_results: List[Document],
        k: int = 60,
        weight_bm25: float = 0.3,
        weight_vector: float = 0.7,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion（倒数排名融合）

        核心思想：不依赖检索器的原始得分（不同检索器的得分不可直接比较），
        而是基于文档在两路检索结果中的**排名**来计算融合分数。

        计算公式：
            score(d) = weight_bm25 / (k + rank_bm25(d))
                     + weight_vector / (k + rank_vector(d))

        参数说明：
            k：平滑常数，防止排名为 0 时除零，默认 60（业界常用值）
            rank_bm25(d)：文档 d 在 BM25 结果中的排名（0-based）
            rank_vector(d)：文档 d 在向量检索结果中的排名（0-based）

        举例：
            文档 D 在 BM25 排第 1，在向量排第 3：
            score = 0.3/(60+1) + 0.7/(60+3) = 0.0049 + 0.0111 = 0.0160

            文档 E 只在向量排第 1：
            score = 0 + 0.7/(60+1) = 0.0115

            → 文档 D 排到前面（两路都命中，置信度更高）

        :param bm25_results: BM25 检索结果（按相关性降序）
        :param vector_results: 向量检索结果（按相关性降序）
        :param k: RRF 平滑常数
        :param weight_bm25: BM25 路权重（默认 0.3）
        :param weight_vector: 向量路权重（默认 0.7）
        :return: RRF 融合后按分降序排列的文档列表（去重）
        """
        # scores：以文档内容为 key 的得分字典
        scores: dict[str, float] = {}
        # doc_map：以文档内容为 key 的 Document 映射（用于去重）
        doc_map: dict[str, Document] = {}

        # 处理 BM25 路结果：对每个文档按排名加分
        for rank, doc in enumerate(bm25_results):
            doc_map[doc.page_content] = doc
            # 排名越靠前，rank 值越小，1/(k+rank) 越大
            scores[doc.page_content] = scores.get(doc.page_content, 0.0) + (
                weight_bm25 / (k + rank)
            )

        # 处理向量路结果：对每个文档按排名加分
        for rank, doc in enumerate(vector_results):
            doc_map[doc.page_content] = doc
            scores[doc.page_content] = scores.get(doc.page_content, 0.0) + (
                weight_vector / (k + rank)
            )
            # 如果一个文档在两路中都出现，分数会累加 → 排名更靠前

        # 按融合得分降序排列，返回 Document 列表
        # scores.items() → [(content, score), ...]
        # sorted(..., reverse=True) → 高分在前
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, _ in ranked]

    # ──────────────────────────────────────────────
    # 核心检索方法
    # ──────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """
        执行三级检索流程：两路召回 → RRF 融合 → CrossEncoder 重排序 → 取 Top-K

        完整链路：
            用户查询
                │
                ├──→ BM25 检索（关键词匹配，召回 top-9）
                ├──→ 向量检索（语义匹配，召回 top-9）
                │
                └──→ RRF 融合（按排名加权，去重合并，约 12-15 篇）
                        │
                        ↓
                  CrossEncoder 重排序
                  （将 query 与每篇文档拼接，逐对打分）
                        │
                        ↓
                  取分数最高的 top-3 返回

        :param query: 用户输入的查询文本
        :return: 重排序后最相关的 top-k 篇文档
        """
        # Step 1: 分别执行两路检索
        # BM25 擅长精确匹配，向量检索擅长语义理解
        bm25_results = self.bm25.invoke(query) if self.bm25 else []
        vector_results = self.vector.invoke(query)

        # 如果两路都未召回任何文档，提前返回空列表
        if not bm25_results and not vector_results:
            logger.warning(f"[混合检索] 两路检索均未召回结果，query={query[:30]}")
            return []

        # Step 2: RRF 融合
        # 将两路结果按排名倒数加权合并，去重后按分降序排列
        candidates = self._rrf_merge(bm25_results, vector_results)

        # Step 3: CrossEncoder 重排序
        # 将查询与每篇候选文档拼接成对，输入 CrossEncoder 做深度语义匹配打分
        # 示例：["知母的功效是什么？", "知母性寒，味苦，归肺胃肾经..."]
        pairs = [[query, doc.page_content] for doc in candidates]

        try:
            # predict 返回每个 pair 的相关性得分（float 数组）
            scores = get_reranker().predict(pairs)
        except Exception as e:
            # 重排序失败时降级：直接返回 RRF 融合结果的前 top_k 篇
            logger.warning(f"[混合检索] CrossEncoder 重排序失败，退回 RRF 结果：{e}")
            return candidates[: self.top_k]

        # Step 4: 按 CrossEncoder 得分降序排列，取 top_k
        # zip(candidates, scores) → [(doc1, 0.92), (doc2, 0.85), ...]
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = [doc for doc, _ in ranked[: self.top_k]]

        logger.info(
            f"[混合检索] query={query[:20]}... "
            f"BM25={len(bm25_results)} + 向量={len(vector_results)} "
            f"→ RRF={len(candidates)} → 精排={len(result)}"
        )
        return result


if __name__ == "__main__":
    print("正在验证模型...")
    _ = get_reranker()
    print("模型加载成功！")
