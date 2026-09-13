"""通用工具函数：Token 估算、JSON 提取、稳定哈希、中文分词等。"""

from __future__ import annotations

import json
import re
import zlib
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def estimate_tokens(text: str) -> int:
    """粗略 Token 估算（演示用）：CJK 字符按 1 token，其余按每 4 字符 1 token。

    生产环境应使用 tokenizer（tiktoken / 各厂商 SDK）精确计数，接口保持不变。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3000" <= ch <= "\u303f")
    return max(1, cjk + (len(text) - cjk + 3) // 4)


def stable_hash(text: str) -> int:
    """跨进程稳定的哈希（内建 hash 有随机化，不能用于持久化索引）。"""
    return zlib.crc32(text.encode("utf-8"))


def truncate(text: str, limit: int = 600) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(截断，共{len(text)}字符)"


def extract_json(text: str):
    """从 LLM 输出中尽力提取 JSON 对象/数组（容忍 markdown 代码块与前后缀文本）。"""
    if text is None:
        return None
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # 从首个 { 或 [ 开始做括号配对
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


_CN_STOP = set("，。！？、；：（）《》“”‘’ \t\n\r,.;:!?()<>\"'`~@#$%^&*-_=+[]{}|\\/0123456789")


def tokenize_cn(text: str) -> list[str]:
    """轻量中文分词：CJK 字符二元组(bigram) + 拉丁单词。

    演示与检索场景足够用；生产可替换 jieba / 领域词典，接口不变。
    """
    tokens: list[str] = []
    latin = re.findall(r"[A-Za-z0-9_]+", text)
    tokens.extend(w.lower() for w in latin)
    chars = [ch for ch in text if ch not in _CN_STOP]
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    if len(chars) == 1:
        tokens.append(chars[0])
    return tokens
