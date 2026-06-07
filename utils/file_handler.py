"""
文件处理模块

提供文件相关的实用工具函数：
- 计算文件 MD5 校验值
- 按扩展名过滤目录中的文件
- 加载 PDF 和 TXT 文档（用于 RAG 知识库）
"""
import os
import hashlib
import shutil
from pathlib import Path

from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from utils.path_tool import get_abs_path


def get_file_md5_hex(filepath: str):
    """
    计算文件的 MD5 哈希值（十六进制），用于文件完整性校验或去重

    :param filepath: 文件绝对路径
    :return: 32 位 MD5 十六进制字符串；如果文件不存在或读取失败返回 None
    """
    # 检查文件是否存在
    if not os.path.exists(filepath):
        logger.error(f'[md5计算]文件{filepath}不存在')
        return

    # 检查路径是否为文件
    if not os.path.isfile(filepath):
        logger.error(f'[md5计算]路径{filepath}不是文件')
        return

    md5_obj = hashlib.md5()
    chunk_size = 4096  # 分块读取，避免大文件占用过多内存
    try:
        with open(filepath, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_obj.update(chunk)
                md5_hx = md5_obj.hexdigest()
                return md5_hx

    except Exception as e:
        logger.error(f'计算文件{filepath}md5失败，{str(e)}')
        return None

def check_md5_hex(md5_for_check: str) -> bool:
    """
    检查文件 MD5 是否已存在于记录中

    :param md5_for_check: 待检查的 MD5 值
    :return: True=新文件(需要加载)，False=已存在(跳过)
    """
    # 如果 MD5 记录文件不存在，创建空文件
    md5_path = get_abs_path(chroma_conf['md5_hex_store'])
    if not os.path.exists(md5_path):
        open(md5_path, 'w', encoding='utf-8').close()
        return True

    # 逐行比对 MD5
    with open(md5_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line.strip() == md5_for_check:
                return False
    return True

def remove_md5_hex(md5_to_remove: str):
    """
    从 MD5 记录文件中删除指定的 MD5 值

    删除文件时调用此函数，否则重新上传同一文件会被 MD5 去重拦截。
    :param md5_to_remove: 待删除的 MD5 值
    """
    md5_path = get_abs_path(chroma_conf['md5_hex_store'])
    if not os.path.exists(md5_path):
        return

    with open(md5_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(md5_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip() != md5_to_remove:
                f.write(line)


def save_md5_hex(md5_for_check: str):
    """
    将已加载文件的 MD5 值追加到记录文件

    :param md5_for_check: 文件的 MD5 值
    """
    with open(
            get_abs_path(chroma_conf['md5_hex_store']),
            'a', encoding='utf-8'
    ) as f:
        f.write(md5_for_check + '\n')

def listdir_with_allowed_type(path: str, allowed_type: tuple[str]):
    """
    列出目录中所有符合允许扩展名的文件完整路径

    :param path: 目录路径
    :param allowed_type: 允许的文件扩展名元组，如 ('.pdf', '.txt')
    :return: 符合条件的文件完整路径元组
    """
    files = []
    if not os.path.isdir(path):
        logger.error(f'[listdir_with_allowed_type]{path}不是文件夹')
        return tuple(files)

    # 遍历目录，筛选出指定扩展名的文件
    for f in os.listdir(path):
        if f.endswith(allowed_type):
            files.append(os.path.join(path, f))

    return tuple(files)

def pdf_loader(filepath: str, passwd=None) -> list[Document]:
    """
    加载 PDF 文件，返回 LangChain Document 列表

    :param filepath: PDF 文件路径
    :param passwd: PDF 文件密码（可选）
    :return: Document 对象列表
    """
    return PyPDFLoader(filepath, passwd).load()

def txt_loader(filepath: str) -> list[Document]:
    """
    加载 TXT 文件，返回 LangChain Document 列表

    :param filepath: TXT 文件路径
    :return: Document 对象列表
    """
    return TextLoader(filepath,encoding='utf-8').load()

def docx_loader(filepath: str) -> list[Document]:
    return Docx2txtLoader(filepath).load()

def get_file_document(read_path: str) -> list[Document]:
    """
    根据文件扩展名选择对应的文档加载器

    :param read_path: 文件绝对路径
    :return: 加载后的 Document 列表
    """
    if read_path.endswith('.txt'):
        return txt_loader(read_path)
    if read_path.endswith('.pdf'):
        return pdf_loader(read_path)
    if read_path.endswith('.docx'):
        return docx_loader(read_path)
    return []

def copy_file(file_path):
    dst = Path(get_abs_path(chroma_conf['data_path'])) / Path(file_path).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, str(dst))

def infer_source_type(filepath: str) -> str:
    """根据文件名推测文档类型"""
    name = os.path.basename(filepath)
    if '药材' in name or 'herb' in name:
        return 'herb'
    if '方剂' in name or 'prescription' in name:
        return 'prescription'
    if '证型' in name or 'symptom' in name:
        return 'symptom'
    return 'general'

if __name__ == '__main__':
    print(txt_loader(r'/temp_data/方剂.txt')[0].page_content)