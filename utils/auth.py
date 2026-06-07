"""
认证与密码管理模块
==================
提供密码哈希生成、验证和用户登录功能。
使用 bcrypt 算法加密密码，不存储明文。

依赖:
    - bcrypt: 密码加密
    - utils.db_handler.DBHandler: 数据库查询用户
"""

import bcrypt

from utils.db_handler import DBHandler


def hash_password(plain: str) -> str:
    """将明文密码转换为 bcrypt 哈希

    bcrypt 内置随机盐（salt），每次生成的哈希值不同，
    即使相同密码也会得到不同的哈希结果。

    参数:
        plain: 明文密码，如 "123456"

    返回:
        str: bcrypt 哈希字符串，形如 $2b$12$...，可直接存入数据库
    """
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码是否与哈希值匹配

    参数:
        plain:  用户输入的明文密码
        hashed: 数据库中存储的 bcrypt 哈希

    返回:
        bool: True 表示密码正确
    """
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def login(username: str, password: str) -> dict | None:
    """用户登录验证

    流程:
        1. 从数据库查询用户信息（含密码哈希）
        2. 用 bcrypt 对比明文密码与哈希
        3. 匹配成功返回用户信息，失败返回 None
        4. 通过 try...finally 确保数据库连接在任意情况下都被释放

    参数:
        username: 用户名
        password: 明文密码

    返回:
        dict: 用户信息，包含 id, username, password(hash), role, created_at
        None: 用户名不存在或密码错误
    """
    db = DBHandler()
    try:
        user = db.get_user(username)
        if user and verify_password(password, user['password']):
            return user
        return None
    finally:
        db.close()  # 确保连接释放，避免连接泄漏
