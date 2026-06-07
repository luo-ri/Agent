import base64
import mimetypes
import os
from langchain.agents import create_agent
import tiktoken
from utils.config_handler import agent_conf, rag_conf
from model.factory import chat_model, mult_model
from utils.prompt_loader import load_system_prompts, load_analyze_image_prompts
from tools.middleware import *
from tools.agent_tools import *
from langchain_core.messages import HumanMessage
from tools.shorttermMemory import ShortTermMemory

memory = ShortTermMemory()

class ReactAgent(object):
    """ReAct 智能体封装类

    将 LangChain Agent 的创建与执行封装为统一的接口，
    外部只需调用 execute_stream() 即可获取流式回复。

    历史管理已移交 app.py，按用户隔离存储，本类不再负责持久化。
    """

    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[get_herbs, get_symptoms, get_Prescription],
            middleware=[
                monitor_tool,
                log_before_model,
            ]
        )
        self.mutl_model = mult_model
        self.encoder = tiktoken.get_encoding("cl100k_base")
        total_window = self._get_model_window(rag_conf['chat_model_name'])
        self.max_tokens = int(total_window * 0.8)
        

    @staticmethod
    def _get_model_window(model_name: str) -> int:
        """根据模型名称返回上下文窗口大小"""
        windows = agent_conf['windows']
        for key, size in windows.items():
            if key in model_name.lower():
                return size
        return 128000

    def _img_to_data_uri(self, path):
        """将本地图片转为 Base64 数据 URI，供多模态模型消费"""
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1][1:].replace("jpg", "jpeg")
        return f"data:image/{ext};base64,{b64}"

    def _is_image(self,file_path: str) -> bool:
        """判断文件是否为图片"""
        IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg", ".ico"}

        # 先快速检查扩展名
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in IMAGE_EXTENSIONS:
            return False
        # 再检查 MIME 类型
        mime, _ = mimetypes.guess_type(file_path)
        return mime is not None and mime.startswith("image/")

    def analyze_image(self,imgs):
        # C:\\Users\\CWK\\AppData\\Local\\Temp\\gradio\\8d6a6b99907726df7debbb0964694d3234b1211f9e3b0de6efe90e9263706555\\01.jpg
        content = [{'type':'text','text':load_analyze_image_prompts()}]
        for img in imgs:
            if not self._is_image(img):
                logger.warning(f'{img}不是图片')
                continue
            content.append({"type": "image", "image": self._img_to_data_uri(img)})

        try:
            res = self.mutl_model.invoke([HumanMessage(content=content)])
            # 多模态返回的 content 可能是 list[dict] 或 str
            if isinstance(res.content, str):
                return res.content
            # 列表中每个元素直接是 {"text": "..."}，没有 "type" 字段
            texts = [item["text"] for item in res.content if "text" in item]
            return "\n".join(texts)

        except Exception as e:
            logger.error(f"图片分析异常：{str(e)}")
            raise e

    def execute_stream(self, query: str, history: list | None = None):
        """
        执行用户查询，以流式方式返回 Agent 回复

        不使用原始对话历史，而是用结构化短期记忆替代，
        大幅降低 token 消耗，同时保留关键辨证信息。

        :param query: 用户输入文本
        :param history: 保留参数兼容性（忽略）
        :yield: Agent 回复文本的逐块输出
        """
        # 短期记忆作为上下文，拼接当前问题
        user_query = query
        context = memory.to_context()
        full_query = f"{context}\n\n{query}" if context else query

        messages = [{"role": "user", "content": full_query}]


        input_dict = {'messages': messages}

        full_reply = ""
        for chunk in self.agent.stream(
            input_dict,
            stream_mode='values',
        ):
            latest_message = chunk['messages'][-1]
            if latest_message.content:
                content = latest_message.content.strip() + '\n'
                full_reply += content
                yield content

        # 流式结束后，用原始问题 + 完整回复更新短期记忆
        if memory and full_reply.strip():
            memory.update(user_query, full_reply.strip(), chat_model)




if __name__ == '__main__':
    # 本地测试：创建 Agent 并执行一个报告生成请求
    # agent = ReactAgent()
    # for chunk in agent.execute_stream('你好'):
    #     print(chunk, end='', flush=True)
    ag = ReactAgent()
    res = ag.analyze_image([r'E:\666666\AI\中医辩证助手Agent\111.jpg'])
    print(res)