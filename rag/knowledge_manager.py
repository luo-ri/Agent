"""
知识库管理模块

维护一个文件级索引（knowledge_index.json），记录已入库的文件信息，
并提供查看、删除文档的功能，无需重新扫描 Chroma 数据库。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from utils.config_handler import chroma_conf,rag_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
from utils.file_handler import remove_md5_hex

INDEX_FILE = get_abs_path(rag_conf['INDEX_FILE'])


class KnowledgeManager:
    """
    知识库管理器

    职责：
    1. 维护 knowledge_index.json，记录每个已入库文件的元信息
    2. 提供列表查看、删除文档、清空知识库等操作
    3. 通过 VectorStoreService 操作 Chroma 中的数据
    """

    def __init__(self):
        self.index = self._load_index()
        self._vector_store = None

    def _get_vector_store(self):
        """延迟获取 VectorStoreService 实例（避免模块级循环导入）"""
        if self._vector_store is None:
            from rag.vector_store import VectorStoreService
            self._vector_store = VectorStoreService()
        return self._vector_store

    # ──────────────────────────────────────────────
    # 索引文件读写
    # ──────────────────────────────────────────────

    def _load_index(self) -> Dict:
        """从 JSON 文件加载索引"""
        try:
            with open(INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_index(self):
        """保存索引到 JSON 文件"""
        Path(INDEX_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_FILE, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────
    # 索引更新（由 vector_store.py 在入库成功后调用）
    # ──────────────────────────────────────────────

    def update_index(self, filename: str, source_type: str, chunk_count: int, md5: str):
        """添加或更新一个文件的索引记录"""
        self.index[filename] = {
            "chunk_count": chunk_count,
            "source_type": source_type,
            "md5": md5,
            "added_at": datetime.now().isoformat(),
        }
        self._save_index()

    # ──────────────────────────────────────────────
    # 查询接口
    # ──────────────────────────────────────────────

    def list_documents(self) -> List[Dict]:
        """返回所有已入库文件的摘要列表"""
        result = []
        for filename, info in self.index.items():
            result.append({
                "filename": filename,
                "chunk_count": info.get("chunk_count", 0),
                "source_type": info.get("source_type", "未知"),
                "added_at": info.get("added_at", ""),
                "md5": info.get("md5", ""),
            })
        # 按入库时间降序排列
        result.sort(key=lambda x: x.get("added_at", ""), reverse=True)
        return result

    def get_chroma_collection_count(self) -> int:
        """查询 Chroma 中的总文档数（用于验证索引准确性）"""
        try:
            return self._get_vector_store().vector_store._collection.count()
        except Exception:
            return 0

    # ──────────────────────────────────────────────
    # 删除操作
    # ──────────────────────────────────────────────

    def delete_document(self, filename: str) -> str:
        """从 Chroma 和索引中删除指定文件的文档

        :param filename: 要删除的文件名
        :return: 操作结果消息
        """
        if filename not in self.index:
            return f"❌ 文件「{filename}」不在索引中"

        try:
            # 从 Chroma 中删除该文件关联的所有 chunk
            # 使用 source 元数据字段匹配（需要在入库时设置 metadata.source = 文件名）
            vs = self._get_vector_store()
            chroma_collection = vs.vector_store._collection
            chunk_count_before = chroma_collection.count()

            chroma_collection.delete(where={"source": filename})

            chunk_count_after = chroma_collection.count()
            deleted = chunk_count_before - chunk_count_after

            # 从索引中移除
            md5 = self.index[filename].get("md5", "")
            del self.index[filename]
            self._save_index()

            # 从 MD5 记录中移除，允许同文件再次入库
            if md5:
                remove_md5_hex(md5)

            logger.info(f"[知识库管理] 删除文件 {filename}，移除 {deleted} 个 chunk")
            return f"✅ 文件「{filename}」已删除（移除 {deleted} 个文档块）"

        except Exception as e:
            logger.error(f"[知识库管理] 删除文件 {filename} 失败：{e}")
            return f"❌ 删除失败：{e}"

    def clear_all(self) -> str:
        """清空整个知识库

        :return: 操作结果消息
        """
        try:
            # 清空 Chroma
            vs = self._get_vector_store()
            chroma_collection = vs.vector_store._collection
            count = chroma_collection.count()
            # 获取所有 ID 后批量删除
            all_ids = chroma_collection.get()["ids"]
            if all_ids:
                chroma_collection.delete(ids=all_ids)

            # 清空索引
            self.index.clear()
            self._save_index()

            # 清空 MD5 记录，允许所有文件重新入库
            open(get_abs_path(chroma_conf['md5_hex_store']), 'w', encoding='utf-8').close()

            logger.info(f"[知识库管理] 清空知识库，删除 {count} 个文档块")
            return f"✅ 知识库已清空（移除 {count} 个文档块）"

        except Exception as e:
            logger.error(f"[知识库管理] 清空知识库失败：{e}")
            return f"❌ 清空失败：{e}"
