"""工具抽象层：Tool 基类 + 轻量 JSON Schema 校验。

统一 Tool Registry 的核心契约：
- 所有工具（进程内 / MCP 远程 / 外部 API 适配）都实现同一接口；
- 参数声明使用 JSON Schema 子集，调用前统一校验（与具体工具实现解耦）；
- 工具声明 timeout / retries / fallback，由弹性执行层统一施策。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    ok: bool
    output: Any = None          # 成功时的结构化输出（渲染为 Observation）
    error: str | None = None    # 失败时的错误信息
    meta: dict[str, Any] = field(default_factory=dict)  # latency / attempts / degraded 等执行元数据

    def observe(self) -> str:
        """渲染为 ReAct 的 Observation 文本。"""
        if not self.ok:
            return f"ERROR: {self.error}"
        if isinstance(self.output, str):
            return self.output
        import json
        try:
            return json.dumps(self.output, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(self.output)


class RetryableError(Exception):
    """工具内部标记的可重试错误（如网络抖动、上游 5xx）。"""


class ToolContext:
    """工具执行上下文：会话隔离 + 依赖注入（内存、事件总线、工作区）。"""

    def __init__(self, session_id: str, workspace_dir=None, memory=None,
                 event_bus=None, trace_id: str = "", config=None) -> None:
        self.session_id = session_id
        self.workspace_dir = workspace_dir
        self.memory = memory
        self.event_bus = event_bus
        self.trace_id = trace_id
        self.config = config


class BaseTool:
    """工具基类：子类实现 run()，并声明 name/description/params。"""

    name: str = "base_tool"
    description: str = ""
    params: dict = {"type": "object", "properties": {}, "required": []}
    timeout_s: float = 10.0
    retries: int = 2                    # 可重试错误的最大尝试次数
    source: str = "in-process"          # in-process | mcp://xxx
    fallback: "BaseTool | None" = None  # 降级备用工具

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:  # pragma: no cover
        raise NotImplementedError

    def validate_args(self, args: dict) -> list[str]:
        return validate_schema(self.params, args or {})

    def schema_line(self) -> str:
        props = self.params.get("properties") or {}
        req = self.params.get("required") or []
        parts = []
        for key, spec in props.items():
            star = "*" if key in req else ""
            desc = spec.get("description", "")
            parts.append(f"{key}{star}({spec.get('type', 'any')}){': ' + desc if desc else ''}")
        return f"{self.name}({', '.join(parts)}) - {self.description} [来源: {self.source}]"


# ------------------------------------------------------------ Schema 校验

_TYPE_MAP = {"string": str, "number": (int, float), "integer": int,
             "boolean": bool, "object": dict, "array": list, "null": type(None)}


def validate_schema(schema: dict, args: dict) -> list[str]:
    """校验入参是否符合 JSON Schema 子集（type/required/enum/properties/items）。

    返回错误列表；空列表表示通过。生产可替换为 jsonschema 库，接口一致。
    """
    errors: list[str] = []
    if schema.get("type") != "object":
        return errors
    for key in schema.get("required", []):
        if key not in args:
            errors.append(f"缺少必填参数: {key}")
    for key, value in (args or {}).items():
        spec = (schema.get("properties") or {}).get(key)
        if spec is None:
            if schema.get("additionalProperties") is False:
                errors.append(f"不允许的参数: {key}")
            continue
        expected = spec.get("type")
        if expected and expected in _TYPE_MAP:
            py_type = _TYPE_MAP[expected]
            if expected == "integer" and isinstance(value, bool):
                errors.append(f"参数 {key} 应为 integer")
                continue
            if not isinstance(value, py_type):
                errors.append(f"参数 {key} 类型应为 {expected}，实际为 {type(value).__name__}")
                continue
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"参数 {key} 取值 {value!r} 不在允许范围 {spec['enum']}")
        if spec.get("type") == "array" and isinstance(value, list) and "items" in spec:
            item_type = _TYPE_MAP.get(spec["items"].get("type"))
            if item_type and any(not isinstance(v, item_type) for v in value):
                errors.append(f"参数 {key} 数组元素类型不符")
    return errors


def run_sync_tool(fn: Callable, *args, **kwargs):
    """兼容同步工具函数：统一包成协程调用。"""
    result = fn(*args, **kwargs)
    if inspect.isawaitable(result):
        return result
    return result
