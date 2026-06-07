"""
路径工具模块

为整个工程提供统一的绝对路径解析功能。
所有模块通过此模块获取路径，确保路径一致性，避免相对路径带来的问题。
"""
import os


def get_project_root() -> str:
    """
    获取项目根目录的绝对路径

    计算方法：当前文件所在目录的父目录（即 utils 的父目录）
    :return: 项目根目录的绝对路径字符串
    """
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(current_dir)
    return project_root


def get_abs_path(relative_path: str) -> str:
    """
    将相对于项目根目录的路径转换为绝对路径

    :param relative_path: 相对于项目根目录的路径，如 'config/rag.yml'
    :return: 拼接后的绝对路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, relative_path)


if __name__ == '__main__':
    print(get_abs_path('config/config.txt'))
