"""
配置加载模块

提供统一的 YAML 配置文件加载功能，用于加载项目中各类配置：
- RAG 配置 (rag.yml)
- Chroma 向量数据库配置 (chroma.yml)
- 提示词路径配置 (prompts.yml)
- Agent 配置 (agent.yml)
"""
import yaml
from utils.path_tool import get_abs_path


def load_rag_config(config_path: str = get_abs_path('config/rag.yml'), encoding: str = 'utf-8'):
    """
    加载 RAG（检索增强生成）相关配置

    :param config_path: 配置文件路径，默认为 config/rag.yml
    :param encoding: 文件编码，默认为 utf-8
    :return: 解析后的 YAML 配置字典
    """
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_chroma_config(config_path: str = get_abs_path('config/chroma.yml'), encoding: str = 'utf-8'):
    """
    加载 Chroma 向量数据库相关配置

    :param config_path: 配置文件路径，默认为 config/chroma.yml
    :param encoding: 文件编码，默认为 utf-8
    :return: 解析后的 YAML 配置字典
    """
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str = get_abs_path('config/prompts.yml'), encoding: str = 'utf-8'):
    """
    加载提示词路径相关配置

    :param config_path: 配置文件路径，默认为 config/prompts.yml
    :param encoding: 文件编码，默认为 utf-8
    :return: 解析后的 YAML 配置字典
    """
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str = get_abs_path('config/agent.yml'), encoding: str = 'utf-8'):
    """
    加载 Agent 配置

    :param config_path: 配置文件路径，默认为 config/agent.yml
    :param encoding: 文件编码，默认为 utf-8
    :return: 解析后的 YAML 配置字典
    """
    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def load_db_config(config_path: str = get_abs_path('config/db.yml'), encoding: str = 'utf-8'):

    with open(config_path, 'r', encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


# 模块加载时即初始化全局配置对象，供其他模块直接导入使用
rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
db_conf = load_db_config()

if __name__ == '__main__':
    print(rag_conf['chat_model_name'])