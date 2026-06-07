"""
中医辩证助手 Agent - Gradio 前端界面（含登录鉴权）

页面：
1. 登录页面：用户名+密码，区分管理员和普通用户
2. 管理员：聊天 + 知识库管理 + 用户管理
3. 普通用户：仅聊天

"""

import os
# 必须在其他所有 HuggingFace 导入之前设置离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import gradio as gr
import json
import shutil
import base64
from pathlib import Path
from rag.vector_store import VectorStoreService
from tools.react_agent import ReactAgent
from utils.path_tool import get_abs_path
from utils.auth import login as do_login, hash_password
from utils.db_handler import DBHandler
from utils.config_handler import agent_conf

DATA_DIR = Path(get_abs_path("temp_data"))
DATA_DIR.mkdir(exist_ok=True)

def get_chat_file(username: str) -> Path:
    """根据用户名返回专属聊天记录文件路径"""
    return Path(get_abs_path(agent_conf["HISTORY_FILE"])) / f"{username}.json"

agent = ReactAgent()


# ============================================================
# 登录 / 注册处理函数
# ============================================================

def handle_login(username, password, session):
    """验证登录，返回界面切换和会话信息，同时加载聊天历史"""
    empty_history = []  # 默认空历史
    if not username.strip() or not password.strip():
        return (
            gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=False),
            session, "⚠️ 请输入用户名和密码",
            gr.Column(visible=True), gr.Column(visible=False),
            empty_history, empty_history,
        )
    user = do_login(username.strip(), password)
    if user is None:
        return (
            gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=False),
            session, "❌ 用户名或密码错误",
            gr.Column(visible=True), gr.Column(visible=False),
            empty_history, empty_history,
        )
    session["role"] = user["role"]
    session["username"] = user["username"]
    # 加载该用户的历史记录
    history = load_chat(user["username"])
    if user["role"] == "admin":
        return (
            gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=False),
            session, "",
            gr.Column(visible=True), gr.Column(visible=False),
            history, history,  # 同时设置 admin 和 user 两个 chatbot（隐藏的那个无影响）
        )
    else:
        return (
            gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=True),
            session, "",
            gr.Column(visible=True), gr.Column(visible=False),
            history, history,
        )

def handle_register(username, password, confirm_password):
    """处理用户注册（仅限普通用户角色）"""
    # 注册失败 → 保持注册面板可见，方便用户看到错误并修改
    if not username.strip():
        return "⚠️ 请输入用户名", gr.Column(visible=False), gr.Column(visible=True)
    if not password:
        return "⚠️ 请输入密码", gr.Column(visible=False), gr.Column(visible=True)
    if len(password) < 6:
        return "⚠️ 密码至少需要 6 位", gr.Column(visible=False), gr.Column(visible=True)
    if password != confirm_password:
        return "⚠️ 两次输入的密码不一致", gr.Column(visible=False), gr.Column(visible=True)

    db = DBHandler()
    try:
        existing = db.get_user(username.strip())
        if existing:
            return f"❌ 用户名 {username.strip()} 已被注册", gr.Column(visible=False), gr.Column(visible=True)

        db.add_user(username.strip(), hash_password(password), role="user")
    except Exception as e:
        return f"❌ 注册失败：{str(e)}", gr.Column(visible=False), gr.Column(visible=True)
    finally:
        db.close()

    # 注册成功 → 切回登录页
    return f"✅ 注册成功！请使用 {username.strip()} 登录", gr.Column(visible=False), gr.Column(visible=True)

def switch_to_register():
    """切换到注册面板"""
    return gr.Column(visible=True), gr.Column(visible=False), ""

def switch_to_login():
    """切换到登录面板"""
    return gr.Column(visible=True), gr.Column(visible=False), ""

def handle_logout(session):
    """退出登录，返回登录页，清空聊天框"""
    session["role"] = None
    session["username"] = None
    return (
        gr.Column(visible=True),     # login_area
        gr.Column(visible=False),    # admin_view
        gr.Column(visible=False),    # user_view
        session,
        "",                          # login_msg
        gr.Column(visible=True),     # login_col
        gr.Column(visible=False),    # register_col
        [],                          # admin_chatbot 清空
        [],                          # user_chatbot 清空
    )


# ============================================================
# 页面 1：文件上传 / 知识库管理页面
# ============================================================

def delete_temp_file():
    """清空 temp_data 目录下所有文件和子文件夹"""
    for item in Path(DATA_DIR).iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

def handle_upload(input_data):
    if input_data is None:
        return "请先选择文件或文件夹。", None

    if isinstance(input_data, list):
        docs = [f for f in input_data if os.path.basename(f).lower().endswith((".pdf", ".txt", ".docx"))]
        msgs = [os.path.basename(f) for f in docs]
        return (
            f"✅ 文件夹上传成功！<br>"
            f'文件：{",".join(msgs)}<br>'
            f"共 {len(docs)} 个文件",
            docs,
        )

    file_name = os.path.basename(input_data)
    file_size = os.path.getsize(str(input_data))
    return ((f"✅ 上传成功！<br>"
             f"**文件名**：{file_name}<br>"
             f"**大小**：{file_size / 1024:.1f} KB"),
            str(input_data))

def use_file(file_path):
    """处理上传的文件 → 入库"""
    if not file_path:
        return "请先上传文件。"
    load = VectorStoreService()
    if isinstance(file_path, list):
        src_folder = os.path.commonpath(file_path)
        folder_name = os.path.basename(src_folder)
        dst_folder = DATA_DIR / folder_name
        dst_folder.mkdir(parents=True, exist_ok=True)
        for src in file_path:
            file_name = os.path.basename(src)
            if file_name.lower().endswith((".pdf", ".txt", ".docx")):
                shutil.copy2(src, str(dst_folder / file_name))
        res = load.load_document(str(dst_folder))
        delete_temp_file()
        return res

    src_path = str(file_path)
    file_name = os.path.basename(src_path)
    dst_path = DATA_DIR / file_name
    shutil.copy2(src_path, str(dst_path))
    res = load.load_document(str(dst_path))
    delete_temp_file()
    return res

def build_upload_page():
    """构建文件上传 / 知识库管理页面"""
    with gr.Blocks() as upload_page:
        tab_choice = gr.Radio(
            choices=["📁 文件上传", "📚 知识库管理"],
            value="📁 文件上传",
            label="选择功能",
            interactive=True,
        )

        with gr.Column(visible=True) as upload_col:
            state = gr.State(None)
            with gr.Row():
                file_input = gr.File(label="选择文件", file_types=[".pdf", ".txt", ".docx"])
            with gr.Row():
                folder_btn = gr.UploadButton("📂 上传文件夹", file_count="directory", variant="secondary")
            output = gr.Markdown()
            file_input.change(fn=handle_upload, inputs=file_input, outputs=[output, state])
            folder_btn.upload(fn=handle_upload, inputs=folder_btn, outputs=[output, state])
            with gr.Row():
                process_btn = gr.Button("⚙️ 入库", variant="primary")
            result_output = gr.Textbox(label="入库结果", lines=3)
            process_btn.click(fn=use_file, inputs=state, outputs=result_output)

        with gr.Column(visible=False) as kb_col:
            gr.Markdown("查看和管理已入库的文件。")

            def kb_list():
                from rag.knowledge_manager import KnowledgeManager
                km = KnowledgeManager()
                docs = km.list_documents()
                if not docs:
                    return "<p style='color:#888'>知识库为空</p>"

                type_labels = {"herb": "🌿 药材", "prescription": "📜 方剂", "symptom": "🔍 证型"}
                html = """<table style='width:100%;border-collapse:collapse'>
                        <tr style='background:#2a5a2a;color:#fff'>
                        <th>文件名</th><th>类型</th><th>chunks</th><th>入库时间</th>
                        </tr>"""
                for d in docs:
                    stype = d.get("source_type", "unknown")
                    label = type_labels.get(stype, stype)
                    html += (
                        "<tr style='border-bottom:1px solid #ddd'>"
                        f"<td><b>{d['filename']}</b></td>"
                        f"<td>{label}</td>"
                        f"<td>{d['chunk_count']}</td>"
                        f"<td>{d['added_at'][:16]}</td>"
                        "</tr>"
                    )
                html += "</table>"
                return html

            kb_output = gr.HTML(value=kb_list())
            kb_refresh = gr.Button("🔄 刷新列表")
            kb_refresh.click(fn=kb_list, outputs=kb_output)

            gr.Markdown("---")
            with gr.Row():
                del_input = gr.Textbox(label="输入要删除的文件名", placeholder="例如：药材.txt", scale=3)
                del_btn = gr.Button("🗑️ 删除", variant="secondary", scale=1)

            def delete_file(filename):
                if not filename.strip():
                    return "请输入文件名", kb_list()
                from rag.knowledge_manager import KnowledgeManager
                km = KnowledgeManager()
                msg = km.delete_document(filename.strip())
                return msg, kb_list()

            del_btn.click(fn=delete_file, inputs=del_input, outputs=[del_input, kb_output])

            gr.Markdown("---")
            clear_all_btn = gr.Button("⚠️ 清空整个知识库", variant="stop")

            def clear_all():
                from rag.knowledge_manager import KnowledgeManager
                km = KnowledgeManager()
                msg = km.clear_all()
                return msg, kb_list()

            clear_all_btn.click(fn=clear_all, outputs=[del_input, kb_output])

        def switch_view(choice):
            if choice == "📁 文件上传":
                return gr.Column(visible=True), gr.Column(visible=False)
            return gr.Column(visible=False), gr.Column(visible=True)

        tab_choice.change(fn=switch_view, inputs=tab_choice, outputs=[upload_col, kb_col])

    return upload_page


# ============================================================
# 页面 2：用户管理页面（仅管理员可见）
# ============================================================

def build_user_mgmt_page():
    """构建用户管理页面"""
    with gr.Blocks() as user_page:
        gr.Markdown("### 👥 用户管理")
        gr.Markdown("添加、删除或查看系统用户。")

        # 用户列表
        def refresh_user_list():
            db = DBHandler()
            try:
                users = db.list_users()
                if not users:
                    return "<p style='color:#888'>暂无用户</p>"
                html = """<table style='width:100%;border-collapse:collapse'>
                        <tr style='background:#2a5a2a;color:#fff'>
                        <th>ID</th><th>用户名</th><th>角色</th><th>创建时间</th>
                        </tr>"""
                for u in users:
                    role_badge = "🔧 管理员" if u["role"] == "admin" else "👤 普通用户"
                    html += (
                        "<tr style='border-bottom:1px solid #ddd'>"
                        f"<td>{u['id']}</td>"
                        f"<td><b>{u['username']}</b></td>"
                        f"<td>{role_badge}</td>"
                        f"<td>{str(u['created_at'])[:19]}</td>"
                        "</tr>"
                    )
                html += "</table>"
                return html
            finally:
                db.close()

        user_list = gr.HTML(value=refresh_user_list())
        refresh_btn = gr.Button("🔄 刷新列表")
        refresh_btn.click(fn=refresh_user_list, outputs=user_list)

        gr.Markdown("---")

        # 添加用户
        gr.Markdown("#### ➕ 添加用户")
        with gr.Row():
            new_username = gr.Textbox(label="用户名", placeholder="请输入用户名", scale=2)
            new_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码", scale=2)
            new_role = gr.Radio(choices=["user", "admin"], value="user", label="角色", scale=1)
        add_btn = gr.Button("添加用户", variant="primary")
        add_msg = gr.Markdown("")

        def add_user_handler(username, password, role):
            if not username.strip() or not password.strip():
                return "⚠️ 用户名和密码不能为空", refresh_user_list()
            db = DBHandler()
            try:
                db.add_user(username.strip(), hash_password(password), role)
                return f"✅ 用户 {username} 添加成功", refresh_user_list()
            except Exception as e:
                return f"❌ 添加失败：{str(e)}", refresh_user_list()
            finally:
                db.close()

        add_btn.click(
            fn=add_user_handler,
            inputs=[new_username, new_password, new_role],
            outputs=[add_msg, user_list]
        )

        gr.Markdown("---")

        # 删除用户
        gr.Markdown("#### 🗑️ 删除用户")
        with gr.Row():
            del_username = gr.Textbox(label="用户名", placeholder="输入要删除的用户名", scale=3)
            del_btn = gr.Button("删除", variant="stop", scale=1)
        del_msg = gr.Markdown("")

        def del_user_handler(username):
            if not username.strip():
                return "⚠️ 请输入用户名", refresh_user_list()
            if username.strip() == "admin":
                return "⚠️ 不能删除内置管理员账号", refresh_user_list()
            db = DBHandler()
            try:
                ok = db.delete_user(username.strip())
                if ok:
                    return f"✅ 用户 {username} 已删除", refresh_user_list()
                return f"⚠️ 用户 {username} 不存在", refresh_user_list()
            except Exception as e:
                return f"❌ 删除失败：{str(e)}", refresh_user_list()
            finally:
                db.close()

        del_btn.click(
            fn=del_user_handler,
            inputs=[del_username],
            outputs=[del_msg, user_list]
        )

    return user_page


# ============================================================
# 聊天相关函数
# ============================================================

def save_chat(history, username: str):
    """保存聊天记录到用户专属 JSON 文件"""
    file_path = get_chat_file(username)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def load_chat(username: str):
    """从用户专属 JSON 文件恢复聊天记录"""
    try:
        with open(get_chat_file(username), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def respond(message, history, session):
    """聊天响应函数（流式输出），完成后自动保存"""
    user_text = message.get("text", "") if isinstance(message, dict) else message
    user_images = message.get("files", []) if isinstance(message, dict) else []

    history = history or []

    # 空输入兜底
    if not user_text.strip() and not user_images:
        history.append({"role": "user", "content": " "})
        history.append({"role": "assistant", "content": "请输入您的中医相关问题。"})
        yield "", history
        return

    # 构建用户展示文本（文字 + 图片缩略图预览）
    if user_images:
        imgs_html = "".join(
            f'<img src="data:image/{os.path.splitext(p)[1][1:]};base64,'
            f'{base64.b64encode(open(p,"rb").read()).decode()}'
            f'" style="max-width:180px;max-height:180px;border-radius:8px;margin:4px;display:inline-block;" />'
            for p in user_images
        )
        user_display = f"{imgs_html}\n\n{user_text}" if user_text.strip() else imgs_html
    else:
        user_display = user_text

    prev_history = history.copy()
    history.append({"role": "user", "content": user_display})
    history.append({"role": "assistant", "content": ''})

    # 立即展示用户消息
    yield "", history

    # 图片分析（在用户消息已显示后进行）
    img_result = ""
    if user_images:
        img_result = agent.analyze_image(user_images)

    # 拼接给智能体的完整内容（含图片分析，用户不可见）
    if img_result and user_text.strip():
        content = f"{user_text}\n\n【图片分析结果】\n{img_result}"
    elif img_result:
        content = f"请根据以下图片分析结果进行中医辨证分析：\n\n{img_result}"
    else:
        content = user_text

    for chunk in agent.execute_stream(content, prev_history):
        history[-1]["content"] += chunk
        yield "", history

    username = (session or {}).get("username", "")
    if username:
        save_chat(history, username)


def build_chat_page(session=None):
    """构建聊天页面，session 来自外部传入的 gr.State"""
    with gr.Blocks() as chat_page:
        gr.Markdown("## 💬 中医辨证助手")
        gr.Markdown("描述您的症状或询问中医药相关问题，助手将为您提供辨证分析和建议。")

        chatbot = gr.Chatbot(
            value=[],
            label="对话记录",
            height=500,
        )

        chat_page.load(fn=lambda: [], outputs=chatbot)

        with gr.Row():
            msg_input = gr.MultimodalTextbox(
                label="输入您的问题",
                placeholder="例如：我最近神疲乏力，少气懒言，语声低微，是什么问题？",
                scale=9,
                file_types=["image"],
            )
            send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)

        clear_btn = gr.Button("🗑️ 清空对话")

        send_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, session],
            outputs=[msg_input, chatbot]
        )

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, session],
            outputs=[msg_input, chatbot]
        )

        def clear_chat(session):
            username = session.get("username") if session else None
            if username:
                get_chat_file(username).unlink(missing_ok=True)
            return []

        clear_btn.click(fn=clear_chat, inputs=[session], outputs=chatbot)

        gr.Markdown("---")
        gr.Markdown("⚠️ **免责声明**：以上仅供参考，具体诊疗请线下就医咨询执业中医师。")

    build_chat_page.last_chatbot = chatbot  # 供外部获取组件引用
    return chat_page


# ============================================================
# 主程序：登录 + 角色区分
# ============================================================

def build_admin_view(session=None):
    """管理员视图：聊天 + 知识库管理 + 用户管理"""
    with gr.Tabs():
        with gr.TabItem("💬 聊天"):
            build_chat_page(session)
        with gr.TabItem("📤 知识库管理"):
            build_upload_page()
        with gr.TabItem("👥 用户管理"):
            build_user_mgmt_page()
    return build_chat_page.last_chatbot


def build_user_view(session=None):
    """普通用户视图：仅聊天"""
    with gr.Tabs():
        with gr.TabItem("💬 聊天"):
            build_chat_page(session)
    return build_chat_page.last_chatbot


def main():
    with gr.Blocks(title="中医辩证助手 Agent") as demo:

        # 会话状态：BrowserState 存在浏览器 localStorage，刷新不丢失
        session = gr.BrowserState({"role": None, "username": None})

        # ============================================================
        # 登录/注册页面（默认显示，已登录用户自动跳过）
        # ============================================================
        with gr.Column(visible=True) as login_area:
            gr.Markdown("# 🏥 中医辩证助手")

            # 登录面板
            with gr.Column(visible=True, scale=1, min_width=400) as login_col:
                gr.Markdown("### 登录")
                login_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                login_password = gr.Textbox(label="密码", type="password", placeholder="请输入密码")
                login_btn = gr.Button("🔐 登录", variant="primary", size="lg")
                login_msg = gr.Markdown("")
                gr.Markdown("")
                to_register_btn = gr.Button("还没有账号？点击注册", variant="secondary", size="sm")

            # 注册面板（默认隐藏）
            with gr.Column(visible=False, scale=1, min_width=400) as register_col:
                gr.Markdown("### 注册新账号")
                reg_username = gr.Textbox(label="用户名", placeholder="请输入用户名")
                reg_password = gr.Textbox(label="密码", type="password", placeholder="至少6位")
                reg_confirm = gr.Textbox(label="确认密码", type="password", placeholder="再次输入密码")
                reg_btn = gr.Button("📝 注册", variant="primary", size="lg")
                reg_msg = gr.Markdown("")
                gr.Markdown("")
                to_login_btn = gr.Button("已有账号？返回登录", variant="secondary", size="sm")

        # ============================================================
        # 管理员视图（登录后显示）
        # ============================================================
        with gr.Column(visible=False) as admin_view:
            with gr.Row():
                gr.Markdown("# 🏥 中医辩证助手（管理员）")
                admin_user_info = gr.Markdown("")
                admin_logout = gr.Button("退出登录", variant="secondary", scale=0)
            admin_chatbot = build_admin_view(session)

        # ============================================================
        # 普通用户视图（登录后显示）
        # ============================================================
        with gr.Column(visible=False) as user_view:
            with gr.Row():
                gr.Markdown("# 🏥 中医辩证助手")
                user_user_info = gr.Markdown("")
                user_logout = gr.Button("退出登录", variant="secondary", scale=0)
            user_chatbot = build_user_view(session)

        # ============================================================
        # 事件绑定
        # ============================================================

        # 登录（同时设置两个 chatbot 的历史，隐藏的那个不受影响）
        login_btn.click(
            fn=handle_login,
            inputs=[login_username, login_password, session],
            outputs=[login_area, admin_view, user_view, session, login_msg, login_col, register_col,
                     admin_chatbot, user_chatbot]
        )

        # 注册
        reg_btn.click(
            fn=handle_register,
            inputs=[reg_username, reg_password, reg_confirm],
            outputs=[reg_msg, login_col, register_col]
        )

        # 切换面板
        to_register_btn.click(
            fn=switch_to_register,
            inputs=[],
            outputs=[login_col, register_col, login_msg]
        )

        to_login_btn.click(
            fn=switch_to_login,
            inputs=[],
            outputs=[login_col, register_col, reg_msg]
        )

        # 管理员退出
        admin_logout.click(
            fn=handle_logout,
            inputs=[session],
            outputs=[login_area, admin_view, user_view, session, login_msg, login_col, register_col,
                     admin_chatbot, user_chatbot]
        )

        # 普通用户退出
        user_logout.click(
            fn=handle_logout,
            inputs=[session],
            outputs=[login_area, admin_view, user_view, session, login_msg, login_col, register_col,
                     admin_chatbot, user_chatbot]
        )

        # 页面加载时：检查是否有已登录会话，有则跳过登录
        def on_load(session):
            username = (session or {}).get("username", "")
            if username:
                history = load_chat(username)
                role = session.get("role", "user")
                if role == "admin":
                    return (
                        "", "",
                        session,
                        gr.Column(visible=False), gr.Column(visible=True), gr.Column(visible=False),
                        history, history,
                    )
                else:
                    return (
                        "", "",
                        session,
                        gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=True),
                        history, history,
                    )
            return "", "", session, gr.Column(visible=True), gr.Column(visible=False), gr.Column(visible=False), [], []

        demo.load(
            fn=on_load,
            inputs=[session],
            outputs=[login_username, login_password, session,
                     login_area, admin_view, user_view,
                     admin_chatbot, user_chatbot]
        )

    return demo


if __name__ == "__main__":
    demo = main()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="emerald"),
        css="footer {visibility: hidden}"
    )
