"""
提示词加载模块

根据 prompts.yml 配置中指定的提示词文件路径，加载各类提示词内容：
- 系统提示词（system prompt）
- RAG 摘要提示词（rag summarize prompt）
- 报告生成提示词（report prompt）
"""
from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


def load_system_prompts() -> str:
    """
    加载系统提示词（System Prompt）

    从配置文件中读取 main_prompt_path 指定的路径，加载提示词内容。
    :return: 系统提示词文本
    :raises KeyError: 配置项中缺少 main_prompt_path
    :raises Exception: 文件读取失败
    """
    try:
        system_prompt_path = get_abs_path(prompts_conf['main_prompt_path'])
    except KeyError as e:
        logger.error(f'[load_system_prompts]在yaml配置项中没有main_prompt_path配置项')
        raise e

    try:
        return open(system_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f'[load_system_prompt]解析系统提示词出错,{str(e)}')
        raise e


def load_rag_prompts() -> str:
    """
    加载 RAG 摘要提示词

    从配置文件中读取 rag_summarize_prompt_path 指定的路径，加载提示词内容。
    :return: RAG 摘要提示词文本
    :raises KeyError: 配置项中缺少 rag_summarize_prompt_path
    :raises Exception: 文件读取失败
    """
    try:
        rag_prompt_path = get_abs_path(prompts_conf['rag_summarize_prompt_path'])
    except KeyError as e:
        logger.error(f'[load_rag_prompts]在yaml配置项中没有rag_summarize_prompt_path配置项')
        raise e

    try:
        return open(rag_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f'[load_rag_prompts]解析rag提示词出错,{str(e)}')
        raise e


def load_analyze_image_prompts() -> str:
    try:
        analyze_image_prompt_path = get_abs_path(prompts_conf['analyze_image_prompt_path'])
    except KeyError as e:
        logger.error(f'[load_analyze_image_prompts]在yaml配置项中没有analyze_image_prompt_path配置项')
        raise e

    try:
        return open(analyze_image_prompt_path, 'r', encoding='utf-8').read()
    except Exception as e:
        logger.error(f'[load_analyze_image_prompts]解析报告提示词出错,{str(e)}')
        raise e


if __name__ == '__main__':
    print(load_system_prompts())