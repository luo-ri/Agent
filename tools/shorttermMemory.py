"""
短期记忆模块
===========
会话级结构化事实存储，每轮对话后从用户和助手消息中抽取中医关键信息，
注入到下一轮提问的上下文前缀中，使 Agent 无需重复推理已确定的事实。
"""

import json
import re
from utils.logger_handler import logger


def _parse_json(text: str) -> dict:
    """从 LLM 返回文本中稳健提取 JSON（处理 markdown 代码块包裹等情况）

    参数:
        text: LLM 原始返回文本

    返回:
        dict: 解析后的字典，失败返回空字典
    """
    # 尝试去除 markdown json 代码块标记
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试用正则提取最外层大括号内容
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning(f"短期记忆 JSON 解析失败，原始文本: {text[:200]}")
        return {}


class ShortTermMemory:
    """会话级短期记忆

    使用示例:
        mem = ShortTermMemory()
        mem.update("我口干", "这可能是阴虚", chat_model)
        context = mem.to_context()  # "已知症状：口干\n已辨证据：阴虚"
    """

    def __init__(self):
        self.facts = {
            "symptoms": [],      # 症状列表，如 ["口干", "盗汗"]
            "patterns": [],      # 证型列表，如 ["阴虚火旺"]
            "herbs_used": [],    # 已讨论药材
            "allergies": [],     # 过敏药物
            "lifestyle": [],     # 生活习惯
        }

    # ------------------------------------------------------------------
    # 核心方法
    # ------------------------------------------------------------------

    def update(self, user_msg: str, assistant_msg: str, llm):
        """每轮对话后从消息中提取新增事实

        参数:
            user_msg:      用户本轮消息
            assistant_msg: 助手本轮回复
            llm:           语言模型实例（需支持 invoke(str) 返回 str）
        """
        prompt = (
            "从以下中医对话中提取关键信息，仅返回 JSON，不要其他文字。\n\n"
            f"用户: {user_msg}\n"
            f"助手: {assistant_msg[:500]}\n\n"  # 截断防过长
            "提取字段（值为字符串数组，未提取到则为空数组）：\n"
            "  symptoms:  症状（如口干、盗汗、乏力）\n"
            "  patterns:  证型（如阴虚火旺、肝郁脾虚）\n"
            "  herbs:     涉及的药材（如知母、黄芪）\n"
            "  allergies: 用户提到的过敏药物\n"
            "  lifestyle: 生活相关（如常熬夜、喜冷饮）\n\n"
            f"当前已知: {json.dumps(self.facts, ensure_ascii=False)}\n\n"
            "要求：只输出本次新增的，已存在的不重复。"
        )
        try:
            raw = llm.invoke(prompt)
            # 兼容 AIMessage 和 str 两种返回类型
            resp_text = raw.content if hasattr(raw, "content") else str(raw)
            new_facts = _parse_json(resp_text)
        except Exception as e:
            logger.warning(f"短期记忆更新失败: {e}")
            return

        # 合并：只添加不重复的新项
        for key in self.facts:
            items = new_facts.get(key, [])
            if not isinstance(items, list):
                items = [items] if items else []
            for item in items:
                item_str = str(item).strip()
                if item_str and item_str not in self.facts[key]:
                    self.facts[key].append(item_str)


    def to_context(self) -> str:
        """将已知事实转为上下文前缀文本

        返回:
            str: 注入下一轮消息的上下文片段，无事实时返回空字符串
        """
        parts = []
        if self.facts["symptoms"]:
            parts.append(f"已知症状：{'、'.join(self.facts['symptoms'])}")
        if self.facts["patterns"]:
            parts.append(f"已辨证据：{'、'.join(self.facts['patterns'])}")
        if self.facts["herbs_used"]:
            parts.append(f"涉及药材：{'、'.join(self.facts['herbs_used'])}")
        if self.facts["allergies"]:
            parts.append(f"⚠️ 过敏：{'、'.join(self.facts['allergies'])}")
        if self.facts["lifestyle"]:
            parts.append(f"生活习惯：{'、'.join(self.facts['lifestyle'])}")

        return "\n".join(parts)

    def clear(self):
        """清空所有记忆"""
        for key in self.facts:
            self.facts[key].clear()

    def is_empty(self) -> bool:
        """检查是否有任何已记录的事实"""
        return all(len(v) == 0 for v in self.facts.values())
