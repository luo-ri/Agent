# 🏥 中医辨证助手 Agent

基于 **LangChain ReAct 架构**的领域智能体，集成 RAG 向量检索与多模态视觉分析，支持舌象/面色望诊融合辨证。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **ReAct Agent** | 自主「思考→工具调用→观察→再思考」推理循环，遵循辨证论治理念 |
| **RAG 检索增强** | ChromaDB 向量库 + BM25 + CrossEncoder 重排序，覆盖药材/方剂/证型三维知识 |
| **多模态望诊** | 上传舌象/面色图片 → qwen-vl-plus 视觉分析 → 融合文字症状 → Agent 综合辨证 |
| **短期记忆系统** | 结构化提取关键事实替代原始对话上下文，Token 消耗降低 ~90% |
| **用户权限体系** | MySQL + bcrypt + RBAC，管理员（聊天/知识库/用户管理）与普通用户（仅聊天） |
| **流式输出** | 逐 Token 实时返回，图片缩略预览，BrowserState 会话持久化 |

---

## 🏗️ 项目结构

```
中医辩证助手Agent/
├── app.py                     # 主入口 — Gradio Web 界面
├── user.sql                   # MySQL 数据库初始化脚本
├── config/                    # YAML 配置文件
│   ├── agent.yml              # Agent 模型配置
│   ├── rag.yml                # RAG 模型配置
│   ├── chroma.yml             # 向量库参数
│   └── db.yml                 # MySQL 连接配置
├── model/                     # 模型工厂
│   ├── factory.py             # ChatModel / Embedding / Multimodal
│   └── models--BAAI--bge-reranker-v2-m3/  # 本地重排序模型
├── rag/                       # RAG 检索增强
│   ├── vector_store.py        # ChromaDB 管理（入库/检索）
│   ├── hybrid_retriever.py    # 混合检索（BM25 + 向量 + 重排序）
│   ├── rag_service.py         # 检索→生成流程编排
│   └── knowledge_manager.py   # 知识库文件索引
├── tools/                     # Agent 工具
│   ├── react_agent.py         # ReAct Agent 核心逻辑
│   ├── agent_tools.py         # 3 个 RAG 工具（get_herbs / get_Prescription / get_symptoms）
│   ├── middleware.py           # 中间件（工具监控 + 日志）
│   └── shorttermMemory.py     # 会话级结构化短期记忆
├── utils/                     # 工具模块
│   ├── auth.py                # bcrypt 认证
│   ├── db_handler.py          # PyMySQL 数据库操作
│   ├── config_handler.py      # YAML 配置加载
│   ├── file_handler.py        # 文档加载（txt/pdf/docx）
│   ├── logger_handler.py      # 日志配置
│   ├── path_tool.py           # 项目路径工具
│   └── prompt_loader.py       # Prompt 模板加载
├── prompts/                   # Prompt 模板
│   ├── main_prompt.txt        # 系统提示词（Agent 角色定义）
│   ├── rag_prompt.txt         # RAG 总结提示词
│   └── analyze_image_prompt.txt  # 图片望诊分析提示词
├── data/                      # 知识库源数据（药材/方剂/证型）
├── history/                   # 按用户隔离的聊天记录
├── chroma_db/                 # ChromaDB 向量数据库
└── logs/                      # 运行日志
```

---

## 🚀 快速开始

### 1. 环境要求

- Python **3.10+**
- MySQL **8.0+**
- 阿里云 DashScope API Key

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

#### MySQL

```bash
# 创建数据库并导入初始化脚本
mysql -u root -p
SET NAMES utf8mb4;
source user.sql;
```

#### API Key

设置环境变量：

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "your-api-key"

# Linux / macOS
export DASHSCOPE_API_KEY="your-api-key"
```

#### 配置文件

编辑 `config/db.yml` 填入 MySQL 连接信息，编辑 `config/agent.yml` 和 `config/rag.yml` 调整模型参数。

### 4. 启动

```bash
python app.py
```

访问 `http://127.0.0.1:7860`

### 5. 默认账号

| 用户名 | 密码 | 角色 | 权限 |
|--------|------|------|------|
| `admin` | `123456` | 管理员 | 聊天 + 知识库管理 + 用户管理 |
| `user1` | `123456` | 普通用户 | 仅聊天 |

---

## 📊 技术架构

```
用户浏览器 ←→ Gradio 6.x
    ├── 登录/注册 (BrowserState 持久化)
    ├── RBAC 权限 (admin / user)
    └── Agent 核心
         ├── ReAct 循环 (思考 → 工具调用 → 观察)
         ├── 3 个工具: get_herbs / get_symptoms / get_Prescription
         ├── 混合检索 (BM25 + 向量 + CrossEncoder)
         ├── 短期记忆 (结构化事实替代原始历史)
         └── 2 个中间件 (监控 + 日志)

MySQL ←→ bcrypt 认证 ←→ 用户管理
ChromaDB ←→ 向量检索 ←→ 知识库
DashScope API ←→ 对话模型(qwen3-max) + 多模态(qwen-vl-plus) + 嵌入(text-embedding-v4)
```

---

## 🔧 可调用工具

| 工具 | 参数 | 说明 |
|------|------|------|
| `get_herbs` | 药材名（如"黄芪"） | 检索性味归经、功效主治、禁忌 |
| `get_Prescription` | 方剂名（如"四君子汤"） | 检索组成、功效、加减应用 |
| `get_symptoms` | 症状组合（如"口干,盗汗"） | 检索证型分析、病因病机 |

---

## ⚠️ 免责声明

本系统仅供学习研究用途，所有辨证分析和用药建议仅供参考，不构成医疗诊断意见。具体诊疗请线下咨询执业中医师。
