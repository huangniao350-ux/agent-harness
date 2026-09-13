"""内置工具集（演示用 mock 数据源，生产替换为真实系统对接即可，接口不变）。

其中刻意内置两个"故障点"用于演示弹性执行层：
- WeatherTool：每会话首次调用抛 RetryableError → 演示自动重试；
- NewsSearchTool：永远抛 RetryableError，且声明 fallback=WebSearchTool → 演示自动降级。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path

from .base import BaseTool, RetryableError, ToolContext, ToolResult


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "安全执行四则运算表达式（支持 + - * / ( ) 与小数）"
    params = {"type": "object", "properties": {
        "expression": {"type": "string", "description": "数学表达式，如 (128*46)+37"}},
        "required": ["expression"], "additionalProperties": False}
    timeout_s = 3.0

    async def run(self, args, ctx):
        expr = args["expression"].strip()
        if not re.fullmatch(r"[0-9+\-*/(). ]+", expr):
            return ToolResult(ok=False, error="表达式包含非法字符，仅支持数字与 + - * / ( )")
        try:
            value = eval(expr, {"__builtins__": {}}, {})  # 白名单字符 + 空 builtins，演示级安全
        except ZeroDivisionError:
            return ToolResult(ok=False, error="除数为零")
        except Exception as exc:
            return ToolResult(ok=False, error=f"表达式无法计算: {exc}")
        return ToolResult(ok=True, output=f"{expr} = {value}")


class CurrentTimeTool(BaseTool):
    name = "current_time"
    description = "获取当前日期时间、星期与 ISO 时间戳"
    params = {"type": "object", "properties": {}, "required": []}
    timeout_s = 3.0

    async def run(self, args, ctx):
        now = datetime.now()
        weekdays = "一二三四五六日"
        return ToolResult(ok=True, output=(
            f"当前时间 {now.strftime('%Y-%m-%d %H:%M:%S')}，星期{weekdays[now.weekday()]}；"
            f"ISO: {now.isoformat(timespec='seconds')}"))


class WeatherTool(BaseTool):
    name = "weather_api"
    description = "查询指定城市天气（演示上游：每会话首次调用会超时一次，用于验证重试机制）"
    params = {"type": "object", "properties": {
        "city": {"type": "string", "description": "城市名，如 北京"},
        "date": {"type": "string", "description": "今天/明天/后天"}},
        "required": ["city"], "additionalProperties": False}
    timeout_s = 5.0
    retries = 3

    _DATA = {
        "北京": ("晴转多云", "18~27℃", "北风3级"),
        "上海": ("小雨", "20~25℃", "东南风4级"),
        "广州": ("雷阵雨", "24~31℃", "南风2级"),
        "深圳": ("多云", "25~30℃", "东风3级"),
        "杭州": ("多云转晴", "19~26℃", "东风2级"),
        "成都": ("阴", "17~23℃", "微风"),
    }

    def __init__(self) -> None:
        self._failed_sessions: set[str] = set()

    async def run(self, args, ctx):
        # 故障注入：每会话首次调用抛可重试错误（生产对应网络抖动/上游 5xx）
        if ctx.session_id not in self._failed_sessions:
            self._failed_sessions.add(ctx.session_id)
            raise RetryableError("上游气象服务超时（模拟网络抖动）")
        city, when = args["city"], args.get("date", "今天")
        cond, temp, wind = self._DATA.get(city, ("多云", "16~26℃", "微风"))
        return ToolResult(ok=True, output=f"{city} {when}：{cond}，气温{temp}，{wind}，空气质量良")


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "通用网络搜索，返回摘要片段列表（演示数据源）"
    params = {"type": "object", "properties": {
        "query": {"type": "string", "description": "搜索关键词"}},
        "required": ["query"], "additionalProperties": False}
    timeout_s = 6.0

    _CORPUS = [
        ("多智能体协同框架迎来标准化浪潮", "业界普遍采用规划-执行-校验三角色分工，任务分发与纠错闭环成为标配……"),
        ("MCP 协议生态快速增长", "统一工具注册与发现协议被主流 Agent 平台采纳，工具调用解耦成为趋势……"),
        ("LLM 应用进入上下文工程时代", "Token 预算控制、记忆分层与上下文压缩成为长会话应用的关键技术……"),
        ("企业知识库检索质量优化实践", "混合检索（BM25+向量）与重排序在边界场景上显著优于单一向量检索……"),
    ]

    async def run(self, args, ctx):
        query = args["query"]
        hits = [f"{t}：{s}" for t, s in self._CORPUS if any(w in t + s for w in _keywords(query))]
        body = hits[:3] if hits else [f"与「{query}」相关的公开资讯较少，已返回通用摘要。"]
        return ToolResult(ok=True, output="搜索结果：\n" + "\n".join(f"{i+1}. {h}" for i, h in enumerate(body)))


class NewsSearchTool(BaseTool):
    """永远失败 + fallback：演示重试耗尽后的自动降级链路。"""

    name = "news_search"
    description = "专用新闻检索（演示上游：服务已下线，将自动降级到 web_search）"
    params = {"type": "object", "properties": {
        "query": {"type": "string"}}, "required": ["query"], "additionalProperties": False}
    timeout_s = 3.0
    retries = 2
    fallback: BaseTool | None = None  # 由 build_builtin_tools() 注入 WebSearchTool

    async def run(self, args, ctx):
        raise RetryableError("news 上游服务 503（模拟已下线）")


def _keywords(query: str) -> list[str]:
    stop = {"查一下", "帮我", "搜索", "搜一下", "最新的", "最近", "关于", "的", "一下", "新闻", "资讯"}
    return [w for w in re.split(r"[\s，。,]", query) if w and w not in stop] or [query]


# ------------------------------------------------------------- 企业 HR 域

_HR_DB = {
    "employees": [
        {"name": "张三", "dept": "研发部", "level": "P5", "join_date": "2023-03-01", "email": "zhangsan@demo.com"},
        {"name": "李雷", "dept": "研发部", "level": "P4", "join_date": "2024-07-15", "email": "lilei@demo.com"},
        {"name": "韩梅梅", "dept": "人事部", "level": "P4", "join_date": "2022-09-01", "email": "hanmm@demo.com"},
        {"name": "王五", "dept": "市场部", "level": "P6", "join_date": "2021-11-20", "email": "wangwu@demo.com"},
    ],
    "leave_balances": [
        {"employee_name": "张三", "annual_total": 10, "annual_used": 4},
        {"employee_name": "李雷", "annual_total": 5, "annual_used": 1},
        {"employee_name": "韩梅梅", "annual_total": 8, "annual_used": 8},
        {"employee_name": "王五", "annual_total": 15, "annual_used": 6},
    ],
    "leave_requests": [],  # LeaveApplyTool 写入
}


class EmployeeLookupTool(BaseTool):
    name = "employee_lookup"
    description = "按姓名查询员工信息（部门/职级/入职日期/邮箱）"
    params = {"type": "object", "properties": {
        "name": {"type": "string", "description": "员工姓名"}}, "required": ["name"],
        "additionalProperties": False}
    timeout_s = 3.0

    async def run(self, args, ctx):
        for row in _HR_DB["employees"]:
            if row["name"] == args["name"].strip():
                return ToolResult(ok=True, output=f"员工信息：{json.dumps(row, ensure_ascii=False)}")
        return ToolResult(ok=False, error=f"未找到员工「{args['name']}」")


class LeaveBalanceTool(BaseTool):
    name = "leave_balance"
    description = "查询员工年假余额（总额/已用/剩余）"
    params = {"type": "object", "properties": {
        "employee_name": {"type": "string"}}, "required": ["employee_name"],
        "additionalProperties": False}
    timeout_s = 3.0

    async def run(self, args, ctx):
        name = args["employee_name"].strip()
        if not any(e["name"] == name for e in _HR_DB["employees"]):
            return ToolResult(ok=False, error=f"员工「{name}」不存在")
        for row in _HR_DB["leave_balances"]:
            if row["employee_name"] == name:
                remain = row["annual_total"] - row["annual_used"]
                return ToolResult(ok=True, output=(
                    f"{name} 年假余额：总额 {row['annual_total']} 天，已用 {row['annual_used']} 天，"
                    f"剩余 {remain} 天"))
        return ToolResult(ok=True, output=f"{name} 暂无年假额度记录（新员工或按月折算中）")


class LeaveApplyTool(BaseTool):
    name = "leave_apply"
    description = "提交请假申请（校验余额与日期，成功返回审批单号）"
    params = {"type": "object", "properties": {
        "employee_name": {"type": "string", "description": "员工姓名"},
        "leave_type": {"type": "string", "enum": ["年假", "事假", "病假", "调休"]},
        "days": {"type": "number", "description": "请假天数，>0"},
        "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD，支持 今天/明天/后天"}},
        "required": ["employee_name", "leave_type", "days", "start_date"],
        "additionalProperties": False}
    timeout_s = 4.0
    risk_level = "high"  # 高危写操作：提交请假需过人工确认门

    async def run(self, args, ctx):
        name = args["employee_name"].strip()
        days = args["days"]
        if days <= 0:
            return ToolResult(ok=False, error="请假天数必须大于 0")
        emp = next((e for e in _HR_DB["employees"] if e["name"] == name), None)
        if emp is None:
            return ToolResult(ok=False, error=f"员工「{name}」不存在，无法提交")
        start = _parse_date(args["start_date"])
        if start is None:
            return ToolResult(ok=False, error="开始日期格式应为 YYYY-MM-DD 或 今天/明天/后天")
        bal = next((b for b in _HR_DB["leave_balances"] if b["employee_name"] == name), None)
        if args["leave_type"] == "年假" and bal and (bal["annual_total"] - bal["annual_used"]) < days:
            remain = bal["annual_total"] - bal["annual_used"]
            return ToolResult(ok=False, error=f"年假余额不足（剩余 {remain} 天，申请 {days} 天）")
        req_id = f"LR-{datetime.now().strftime('%Y%m%d')}-{len(_HR_DB['leave_requests']) + 101}"
        _HR_DB["leave_requests"].append({**args, "request_id": req_id, "start_date": start.isoformat()})
        return ToolResult(ok=True, output=(
            f"请假申请已提交：{name} {args['leave_type']} {days} 天（{start.isoformat()} 起），"
            f"审批单号 {req_id}，已同步直属上级审批。"))


def _parse_date(text: str):
    text = text.strip()
    offsets = {"今天": 0, "明天": 1, "后天": 2}
    if text in offsets:
        return datetime.now().date() + timedelta(days=offsets[text])
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        return None


class SendEmailTool(BaseTool):
    name = "send_email"
    description = "发送邮件（演示环境：写入本地 outbox，不真实发送）"
    params = {"type": "object", "properties": {
        "to": {"type": "string", "description": "收件人邮箱"},
        "subject": {"type": "string"},
        "body": {"type": "string"}},
        "required": ["to", "subject", "body"], "additionalProperties": False}
    timeout_s = 4.0
    risk_level = "high"  # 高危写操作：对外发信需过人工确认门

    async def run(self, args, ctx):
        outbox = Path(ctx.workspace_dir or ".") / "outbox.jsonl"
        record = {**args, "sent_at": datetime.now().isoformat(timespec="seconds"), "session": ctx.session_id}
        with open(outbox, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return ToolResult(ok=True, output=f"邮件已投递至 {args['to']}（演示 outbox），主题：{args['subject']}")


class FileReadTool(BaseTool):
    name = "file_read"
    description = "读取工作区沙箱内的文本文件"
    params = {"type": "object", "properties": {
        "path": {"type": "string"}}, "required": ["path"], "additionalProperties": False}
    timeout_s = 3.0

    async def run(self, args, ctx):
        path = _safe_path(ctx, args["path"])
        if path is None:
            return ToolResult(ok=False, error="路径越界：仅允许访问工作区沙箱内文件")
        if not path.exists():
            return ToolResult(ok=False, error=f"文件不存在: {args['path']}")
        return ToolResult(ok=True, output=path.read_text(encoding="utf-8")[:2000])


class FileWriteTool(BaseTool):
    name = "file_write"
    description = "在工作区沙箱内写入文本文件"
    params = {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"], "additionalProperties": False}
    timeout_s = 3.0
    risk_level = "high"  # 高危写操作：写入文件需过人工确认门

    async def run(self, args, ctx):
        path = _safe_path(ctx, args["path"])
        if path is None:
            return ToolResult(ok=False, error="路径越界：仅允许访问工作区沙箱内文件")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return ToolResult(ok=True, output=f"已写入 {path}（{len(args['content'])} 字符）")


def _safe_path(ctx: ToolContext, rel: str):
    root = Path(ctx.workspace_dir or ".").resolve()
    try:
        path = (root / rel).resolve()
        path.relative_to(root)
        return path
    except (ValueError, OSError):
        return None


class KnowledgeBaseSearchTool(BaseTool):
    name = "kb_search"
    description = "企业知识库检索（混合检索：BM25+向量+RRF 融合），返回最相关的制度条款"
    params = {"type": "object", "properties": {
        "query": {"type": "string", "description": "检索问题"},
        "top_k": {"type": "integer", "description": "返回条数，默认3"}},
        "required": ["query"], "additionalProperties": False}
    timeout_s = 6.0

    async def run(self, args, ctx):
        from ..rag.rag import get_knowledge_base
        kb = get_knowledge_base()
        hits = kb.search(args["query"], top_k=int(args.get("top_k") or 3))
        if not hits:
            return ToolResult(ok=False, error="知识库中未找到相关内容")
        lines = [f"{i+1}.【{h.meta.get('doc', '')}·{h.meta.get('heading', '')}】{h.text}"
                 for i, h in enumerate(hits)]
        return ToolResult(ok=True, output="知识库命中：\n" + "\n".join(lines),
                          meta={"hits": len(hits)})


def build_builtin_tools() -> list[BaseTool]:
    news = NewsSearchTool()
    news.fallback = WebSearchTool()  # 降级链路：news_search → web_search
    return [
        CalculatorTool(), CurrentTimeTool(), WeatherTool(), WebSearchTool(), news,
        KnowledgeBaseSearchTool(), EmployeeLookupTool(), LeaveBalanceTool(), LeaveApplyTool(),
        SendEmailTool(), FileReadTool(), FileWriteTool(),
    ]
