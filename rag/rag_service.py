"""
RAG 检索增强生成服务模块

实现检索增强生成（Retrieval-Augmented Generation）核心流程：
1. 接收用户查询 → 嵌入向量化
2. Chroma 向量库相似度检索 → 获取相关文档片段
3. 将检索结果注入提示词 → 调用 LLM 生成基于知识的回答

采用 LCEL（LangChain Expression Language）构建处理链，
支持流式输出和灵活扩展。
"""

# 必须在所有 HuggingFace/transformers 导入之前设置离线模式
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import tiktoken
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from rag.hybrid_retriever import HybridRetriever
from utils.prompt_loader import load_rag_prompts, load_system_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
import tiktoken
from utils.config_handler import rag_conf

# token 计数器（全局单例）
_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer

class RagSummarizerService(object):
    """
    RAG 摘要服务

    封装了"检索 → 拼接上下文 → LLM 生成"的完整 RAG 流程。
    所有使用 RAG 的工具（如 rag_summarize）都通过此服务执行。
    """

    def __init__(self, source_type: str = None):
        """
        初始化 RAG 服务组件链：
        VectorStore → Retriever → PromptTemplate → ChatModel → StrOutputParser

        :param source_type: 来源类型（'herb'/'prescription'/'symptom'），None 表示检索全部
        """
        # 1. 向量存储服务（连接 Chroma 数据库）
        self.vector_store = VectorStoreService()
        # 2. 混合检索器（BM25 + 向量检索 + CrossEncoder 重排序）
        self.retriever = HybridRetriever(
            vector_store=self.vector_store.vector_store,
            source_type=source_type,
            top_k=3,
        )
        # 3. RAG 提示词模板（定义如何结合用户输入和检索结果）
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        # 4. 大语言模型（用于生成最终回答）
        self.model = chat_model
        # 5. 组装完整链
        self.chain = self._init_chain()

    def _init_chain(self):
        """
        初始化 LCEL 处理链

        链结构：PromptTemplate | LLM | StrOutputParser
        - PromptTemplate: 将用户输入和检索上下文填入模板
        - LLM: 调用大模型生成回答
        - StrOutputParser: 将 LLM 输出解析为纯文本字符串

        :return: 可调用的 LCEL 链对象
        """
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        """
        执行混合检索（BM25 + 向量 + 重排序），获取与查询相关的文档片段

        :param query: 用户查询文本
        :return: 重排序后的 Document 对象列表
        """
        return self.retriever.retrieve(query)

    def rag_summarizer(self, query: str) -> str:
        """
        RAG 核心方法：检索 + 生成

        完整执行流程：
        1. retriever_docs: 从向量库检索相关文档
        2. 拼接文档为 context 字符串（含元数据引用）
        3. chain.invoke: 将 query + context 注入提示词 → LLM 生成回答

        :param query: 用户的问题或检索关键词
        :return: 基于知识库生成的回答文本
        """
        # Step 1: 检索相关文档
        context_docs = self.retriever_docs(query)
        context = ''
        counter = 0
        enc = get_tokenizer()
        system_tokens = len(enc.encode(load_system_prompts()))
        query_tokens = len(enc.encode(query))
        reserved_output = rag_conf['reserved_output']

        max_context = 128000 - system_tokens - query_tokens - reserved_output

        for doc in context_docs:
            candidate = (
                f'[参考资料{counter + 1}]：{doc.page_content}'
                f' | 元数据：{doc.metadata}\n'
            )
            # 估算 token（中文字符 ≈ 1 token，英文字符 ≈ 0.25 token）
            estimated_tokens = len(candidate) * 1.2
            # 简易估算：len(text) 对中文偏小，乘以 1.2 近似 token 数
            # 更精确可用 tiktoken

            if estimated_tokens > max_context:
                break
            context += candidate
            max_context -= estimated_tokens
            counter += 1

        # Step 3: 执行 LCEL 链，生成基于知识的回答
        return self.chain.invoke(
            {
                'input': query,
                'context': context
            }
        )


if __name__ == '__main__':
    # 本地测试：查询小户型扫地机器人推荐
    rag = RagSummarizerService()
    print(rag.rag_summarizer('知母适合治疗什么疾病？'))
