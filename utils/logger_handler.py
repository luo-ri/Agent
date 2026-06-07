"""
日志处理模块

统一管理项目的日志输出，支持：
- 同时输出到控制台和日志文件
- 控制台和文件可设置不同的日志级别
- 日志文件按天自动轮转（文件名包含日期）
"""
import logging
from datetime import datetime

from utils.path_tool import get_abs_path
import os

# 日志根目录，位于项目根目录下的 logs 文件夹
LOG_ROOT = get_abs_path('logs')

# 确保日志目录存在
os.makedirs(LOG_ROOT, exist_ok=True)

# 默认日志格式：时间 - 日志器名称 - 级别 - 源文件:行号 - 消息
DEFAULT_LOG_FORMAT = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)


def get_logger(
        name: str = 'agent',
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file=None
) -> logging.Logger:
    """
    获取或创建 Logger 实例，同时配置控制台输出和文件输出

    :param name: 日志器名称，默认为 'agent'
    :param console_level: 控制台日志级别，默认为 INFO
    :param file_level: 文件日志级别，默认为 DEBUG
    :param log_file: 日志文件路径，不指定则自动生成（按天命名）
    :return: 配置好的 Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 创建控制台输出处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(console_handler)

    # 未指定日志文件时，按日期自动生成文件名
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')

    # 创建文件输出处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


# 全局默认日志实例，供其他模块直接导入使用
logger = get_logger()

if __name__ == '__main__':
    logger.info('信息日志')
    logger.error('错误日志')
    logger.warning('警告日志')
    logger.debug('调试日志')


