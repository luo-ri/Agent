"""
向量存储服务模块 - Chroma 数据库管理

负责整个 RAG 系统的数据持久化层：
1. 连接本地 Chroma 向量数据库
2. 管理知识库文档的增量加载（含 MD5 去重）
3. 提供文本分片和检索器接口

支持的数据源类型：txt、pdf
文本分片策略：RecursiveCharacterTextSplitter（递归字符分割）
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_core.documents import Document
from utils.file_handler import (get_file_md5_hex, check_md5_hex, save_md5_hex,get_file_document,
                                listdir_with_allowed_type,copy_file,infer_source_type)
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import emded_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from rag.knowledge_manager import KnowledgeManager


class VectorStoreService:
    """
    向量存储服务

    封装 Chroma 向量数据库的初始化、文档加载和检索器创建。
    知识库数据存储在 chroma_db/ 目录下（本地持久化），
    支持重复文件检测（MD5 去重）。
    """

    def __init__(self):
        """
        初始化向量存储：
        1. 连接本地 Chroma 数据库（自动加载已有数据）
        2. 配置文本分割器（chunk 大小和重叠量由 chroma.yml 控制）
        """
        # ======== Chroma 向量数据库连接 ========
        # 使用指定的 collection 名称 + embedding 模型 + 持久化目录
        self.vector_store = Chroma(
            collection_name=chroma_conf['collection_name'],
            embedding_function=emded_model,
            persist_directory=get_abs_path(chroma_conf['persist_directory'])
        )

    def spliters(self,source_type):
        # ======== 文本分割器 ========
        # RecursiveCharacterTextSplitter 递归地按分隔符切分文本：
        # 优先按段落(\n\n)切分 → 再按行(\n) → 句号 → 分号 → 逗号 → 字符
        # 这种策略能最大程度保持语义完整性
        if source_type == 'symptom':
            chunk_size = chroma_conf['symptom_chunk_size']
            chunk_overlap = chroma_conf['symptom_chunk_overlap']

        elif source_type == 'herb':
            chunk_size = chroma_conf['herb_chunk_size']
            chunk_overlap = chroma_conf['herb_chunk_overlap']
        else:
            chunk_size = chroma_conf['prescription_chunk_size']
            chunk_overlap = chroma_conf['prescription_chunk_overlap']

        spliter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # 每块最大字符数
            chunk_overlap=chunk_overlap,  # 块间重叠字符数（保持上下文连贯）
            separators=chroma_conf['separators'],  # 分隔符优先级列表
            length_function=len
        )
        return spliter

    def get_retriever(self, source_type: str = None):
        """
        获取向量检索器

        返回配置了 top-k 参数的 Chroma 检索器。
        k 值由 chroma.yml 文件中的配置控制（默认 3）。

        :param source_type: 来源类型过滤，None 表示检索全部
        :return: 配置好的检索器对象
        """
        search_kwargs = {'k': chroma_conf['k']}
        if source_type:
            search_kwargs['filter'] = {'source_type': source_type}
        return self.vector_store.as_retriever(
            search_kwargs=search_kwargs
        )

    def load_document(self, filepath: str ):
        """
        增量加载知识库文档到 Chroma 数据库

        完整执行流程：
        1. 扫描 data/ 目录下的 txt/pdf/docx 文件
        2. 对每个文件计算 MD5 值，与 md5.txt 中已加载的记录比对
        3. 新文件 → 加载 → 分片 → 存入 Chroma → 记录 MD5
        4. 已加载文件 → 跳过

        支持重复运行，不会重复添加已入库的数据。

        :param filepath: 目录路径或文件路径，默认为 data/
        :return: 成功加载的文件数量
        """
        load_count = 0
        msgs = []

        # ======== 主加载流程 ========

        if os.path.isdir(filepath):
            # 1. 获取所有允许类型的文件路径
            allowed_file_path: list[str] = listdir_with_allowed_type(
                filepath,
                tuple(chroma_conf['allow_knowledge_file_type']),
            )

            # 2. 逐个文件处理
            for path in allowed_file_path:
                md5_hex = get_file_md5_hex(path)

                # MD5 去重检查
                if not check_md5_hex(md5_hex):
                    logger.info(f'[加载知识库]{path}内容已存在知识库内，跳过')
                    msgs.append(f'[加载知识库]{path}内容已存在知识库内，跳过')
                    continue

                try:
                    # 加载原始文档
                    documents: list[Document] = get_file_document(path)
                    if not documents:
                        logger.warning(f'[加载知识库]{path}内没有有效文本内容，跳过')
                        msgs.append(f'[加载知识库]{path}内没有有效文本内容，跳过')
                        continue

                    source_type = infer_source_type(path)

                    # 文本分片
                    split_document: list[Document] = self.spliters(source_type).split_documents(documents)
                    if not split_document:
                        logger.warning(f'[加载知识库]{path}分片后没有有效文本内容，跳过')
                        msgs.append(f'[加载知识库]{path}分片后没有有效文本内容，跳过')
                        continue


                    file_basename = os.path.basename(path)
                    for doc in split_document:
                        doc.metadata['source_type'] = source_type
                        doc.metadata['source'] = file_basename

                    # 存入 Chroma 向量数据库
                    self.vector_store.add_documents(split_document)

                    # 复制文件到 data/ 目录（仅当源目录不是 data/ 时）
                    if filepath != get_abs_path(chroma_conf['data_path']):
                        copy_file(path)

                    # 所有操作成功后，记录 MD5 供下次去重
                    save_md5_hex(md5_hex)
                    load_count += 1

                    # 更新知识库索引
                    KnowledgeManager().update_index(
                        filename=file_basename,
                        source_type=source_type,
                        chunk_count=len(split_document),
                        md5=md5_hex,
                    )

                    logger.info(f'[加载知识库]{path}内容加载成功')
                    msgs.append(f'[加载知识库]{path}内容加载成功')

                except Exception as e:
                    logger.error(f'[加载知识库]{path}加载失败，{str(e)}')
                    msgs.append(f'[加载知识库]{path}加载失败，{str(e)}')
                    continue

            logger.info(f'[加载知识库]目录扫描完成，成功加载 {load_count} 个文件')
            msgs.append(f'[加载知识库]成功加载 {load_count} 个文件')
            return "\n".join(msgs)

        elif os.path.isfile(filepath):
            return self._load_single_file(filepath)

        else:
            logger.warning(f'[加载知识库]{filepath}加载失败,不是有效文件或目录')
            msgs.append(f'[加载知识库]{filepath}加载失败,不是有效文件或目录')
            return ",".join(msgs)

    def _load_single_file(self, filepath: str) -> str:
        """加载单个文件，成功返回 1，跳过或失败返回 0"""
        md5_hex = get_file_md5_hex(filepath)
        msgs = []
        if not check_md5_hex(md5_hex):
            logger.info(f'[加载知识库]{filepath}内容已存在知识库内，跳过')
            msgs.append(f'[加载知识库]{filepath}内容已存在知识库内，跳过')
            return ','.join(msgs)

        try:
            documents: list[Document] = get_file_document(filepath)
            if not documents:
                logger.warning(f'[加载知识库]{filepath}内没有有效文本内容，跳过')
                msgs.append(f'[加载知识库]{filepath}内没有有效文本内容，跳过')
                return ','.join(msgs)

            source_type = infer_source_type(filepath)

            split_document: list[Document] = self.spliters(source_type).split_documents(documents)
            if not split_document:
                logger.warning(f'[加载知识库]{filepath}分片后没有有效文本内容，跳过')
                msgs.append(f'[加载知识库]{filepath}分片后没有有效文本内容，跳过')
                return ','.join(msgs)

            file_basename = os.path.basename(filepath)
            for doc in split_document:
                doc.metadata['source_type'] = source_type
                doc.metadata['source'] = file_basename

            self.vector_store.add_documents(split_document)

            # 复制文件到 data/ 目录（仅当源不在 data/ 内时）
            data_dir = get_abs_path(chroma_conf['data_path'])
            if not str(filepath).startswith(str(data_dir)):
                copy_file(filepath)

            # 所有操作成功后，记录 MD5 供下次去重
            save_md5_hex(md5_hex)

            # 更新知识库索引
            KnowledgeManager().update_index(
                filename=file_basename,
                source_type=source_type,
                chunk_count=len(split_document),
                md5=md5_hex,
            )

            logger.info(f'[加载知识库]{filepath}内容加载成功')
            msgs.append(f'[加载知识库]{filepath}内容加载成功')
            return ','.join(msgs)

        except Exception as e:
            logger.error(f'[加载知识库]{filepath}加载失败，{str(e)}')
            msgs.append(f'[加载知识库]{filepath}加载失败，{str(e)}')
            return ','.join(msgs)


if __name__ == '__main__':
    # 本地测试：加载知识库文档到 Chroma
    vs = VectorStoreService()
    print(vs.load_document(r'E:\666666\AI\中医辩证助手Agent\data'))
