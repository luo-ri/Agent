"""
数据库操作封装模块
===================
本模块基于 PyMySQL 封装了与 tcm_agent 数据库中 users 表的交互，
提供用户增删查功能，所有查询使用参数化查询防止 SQL 注入。
连接信息从 config/db.yml 读取。
"""

import pymysql
from utils.config_handler import db_conf


class DBHandler:
    """MySQL 数据库操作处理器

    每次实例化会创建一个新的数据库连接，用完需调用 close() 释放。

    使用示例:
        db = DBHandler()
        user = db.get_user("admin")
        db.close()
    """

    def __init__(self):
        """初始化数据库连接

        连接参数从 config/db.yml 的 db_conf 读取，包括:
        - host: 数据库地址
        - port: 端口号
        - user / password: 登录凭据
        - database: 数据库名
        - charset: utf8mb4，支持中文和 emoji
        - connect_timeout: 连接超时 5 秒
        - cursorclass: DictCursor，查询结果以字典形式返回
        """
        self.conn = pymysql.connect(
            host=db_conf['host'],
            port=db_conf['port'],
            user=db_conf['user'],
            password=db_conf['password'],
            database=db_conf['database'],
            charset='utf8mb4',
            connect_timeout=5,
            cursorclass=pymysql.cursors.DictCursor
        )

    def close(self):
        """关闭数据库连接，释放资源"""
        if self.conn and self.conn.open:
            self.conn.close()

    # ============================================================
    # 用户相关操作
    # ============================================================

    def get_user(self, username: str) -> dict | None:
        """根据用户名查询用户信息

        参数:
            username: 用户名（如 "admin"）

        返回:
            dict: 用户信息字典，包含 id, username, password(hash), role, created_at
            None: 用户不存在
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username=%s", (username,))
            return cur.fetchone()

    def add_user(self, username, password_hash, role='user'):
        """添加新用户（由管理员操作）

        参数:
            username:      用户名
            password_hash: bcrypt 加密后的密码哈希
            role:          角色，'admin' 或 'user'，默认 'user'

        返回:
            True: 添加成功（用户名冲突会抛出 IntegrityError）
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (username, password, role) VALUES (%s, %s, %s)",
                (username, password_hash, role)
            )
        self.conn.commit()  # INSERT 需要手动提交
        return True

    def delete_user(self, username):
        """根据用户名删除用户

        参数:
            username: 要删除的用户名

        返回:
            bool: True 表示删除成功，False 表示用户不存在
        """
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE username=%s", (username,))
        self.conn.commit()
        return cur.rowcount > 0

    def list_users(self):
        """列出所有用户（不含密码哈希）

        返回:
            list[dict]: 用户列表，每条包含 id, username, role, created_at
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT id, username, role, created_at FROM users ORDER BY id")
            return cur.fetchall()


# ============================================================
# 模块自检：直接运行此文件可验证数据库连接和用户数据
# ============================================================
if __name__ == '__main__':
    db = DBHandler()
    try:
        with db.conn.cursor() as cur:
            # 查询 MySQL 版本，确认连接正常
            cur.execute("SELECT VERSION()")
            version = cur.fetchone()
            print(f"✅ 数据库连接成功！MySQL 版本: {version}")

            # 列出所有用户
            cur.execute("SELECT * FROM users")
            users = cur.fetchall()
            print(f"用户列表: {users}")
    finally:
        db.close()  # 确保连接释放
