-- ============================================================
-- 中医辩证助手 — 数据库初始化脚本
-- ============================================================
-- 用法：在 MySQL 客户端中执行
--   mysql -u root -p
--   SET NAMES utf8mb4;
--   source E:\666666\AI\中医辩证助手Agent\user.sql
-- ============================================================

-- 设置字符集为 utf8mb4，确保中文正常存储
SET NAMES utf8mb4;

-- 创建数据库（如果已存在则跳过）
CREATE DATABASE IF NOT EXISTS tcm_agent
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- 切换到目标数据库
USE tcm_agent;

-- 用户表：存储登录账号信息
-- password 字段存储 bcrypt 哈希，不存明文
-- role 字段控制权限：admin 可管理知识库和聊天，user 只能聊天
CREATE TABLE IF NOT EXISTS users (
    id          INT AUTO_INCREMENT PRIMARY KEY,          -- 自增主键
    username    VARCHAR(50)  NOT NULL UNIQUE,            -- 用户名，唯一
    password    VARCHAR(255) NOT NULL,                   -- bcrypt 密码哈希
    role        ENUM('admin','user') NOT NULL DEFAULT 'user',  -- 角色
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP      -- 创建时间
);

-- ============================================================
-- 插入默认用户
-- 管理员密码为 "123456" 的 bcrypt 哈希
-- 生产环境请修改密码并重新生成哈希
-- ============================================================

-- 管理员（拥有全部权限）
INSERT INTO users (username, password, role)
VALUES ('admin', '$2b$12$obyqwX6837z.xOS784vV3Os4IFSZu4EDmkyG/LcdwAcsY8VLc47dC', 'admin');

-- 普通用户（仅聊天权限）
INSERT INTO users (username, password, role)
VALUES ('张三', '$2b$12$obyqwX6837z.xOS784vV3Os4IFSZu4EDmkyG/LcdwAcsY8VLc47dC', 'user');
