"""MockLLM：脚本化的离线模型实现（演示与评测的"零依赖"基石）。

设计思路：
- 上层 Harness 通过 [MODE: XXX] 标记区分调用意图（ReAct / 规划 / 校验 / 反思 / 摘要…），
  MockLLM 据此分发到对应的脚本处理器；
- ReAct 脚本通过统计 Prompt 中 Observation 的数量推断当前步数，从而按剧本推进；
- 场景（scenario）按目标文本关键词匹配，未命中时走通用兜底脚本。

这样整套平台在无网络、无 API Key 的环境也能完整、确定性地演示。
切换真实模型只需 AGENT_LLM_PROVIDER=openai，Harness 侧零改动。
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from .base import LLMClient, LLMMessage, LLMResponse, normalize_messages, usage_from_text

TOMORROW = (date.today() + timedelta(days=1)).isoformat()


# ---------------------------------------------------------------- ReAct 脚本

def _obs_count(user_text: str) -> int:
    return user_text.count("Observation:")


def _last_observation(user_text: str) -> str:
    parts = user_text.rsplit("Observation:", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def _goal_of(user_text: str) -> str:
    m = re.search(r"目标：(.*?)$", user_text, re.MULTILINE)
    if m:
        return m.group(1).split("（总目标")[0].strip()
    return user_text.strip().splitlines()[0] if user_text.strip() else ""


def _memory_value(user_text: str, key: str) -> str | None:
    m = re.search(rf"key={key}\s+value=([^\s\n]+)", user_text)
    return m.group(1) if m else None


_NAME_BLACKLIST = "我的你他她它一下公司这个查帮为几天提交申请有没询看交"
_NAME_STOPWORDS = {"年假", "请假", "余额", "员工", "信息", "事假", "病假", "调休", "资料"}


def _extract_name(goal: str, user_text: str) -> str:
    """从目标中逐位置匹配姓名（零宽前瞻避免黑名单命中吃掉整段匹配）。

    目标形如「查询张三的年假余额」时，需要跳过 查询/一下 等动词前缀，
    以及 年假/请假 等业务词本身，才能准确定位人名。
    """
    pattern = r"(?=([\u4e00-\u9fa5]{2,3}?)(?:的)?(?:年假|请假|余额|员工|信息))"
    for m in re.finditer(pattern, goal):
        name = m.group(1).rstrip("的")
        if len(name) < 2 or name in _NAME_STOPWORDS:
            continue
        if not any(ch in name for ch in _NAME_BLACKLIST):
            return name
    mem = _memory_value(user_text, "name")
    return mem or "张三"


def _extract_city(goal: str) -> str:
    m = re.search(r"(北京|上海|广州|深圳|杭州|成都|西安|武汉|南京)", goal)
    return m.group(1) if m else "北京"


def _extract_days(goal: str) -> int:
    m = re.search(r"(\d+)\s*天", goal)
    return int(m.group(1)) if m else 3


def _extract_expr(goal: str) -> str:
    m = re.search(r"([\d\s()+\-*/.]{3,})", goal)
    return m.group(1).strip() if m else goal


def _has_error(obs: str) -> bool:
    return bool(re.search(r"ERROR|失败|Traceback|异常|错误", obs))


def react_hr_leave(user_text: str) -> str:
    goal, obs_n, name = _goal_of(user_text), _obs_count(user_text), _extract_name(_goal_of(user_text), user_text)
    days = _extract_days(goal)
    if obs_n == 0:
        return (f"Thought: 提交请假前需要先确认员工「{name}」的身份信息是否在系统中。\n"
                f"Action: employee_lookup\nAction Input: {{\"name\": \"{name}\"}}")
    if obs_n == 1:
        return (f"Thought: 员工信息已确认，接下来查询该员工当前的年假余额是否足够。\n"
                f"Action: leave_balance\nAction Input: {{\"employee_name\": \"{name}\"}}")
    if obs_n == 2:
        return (f"Thought: 余额充足，按目标提交从明天开始 {days} 天的年假申请。\n"
                f"Action: leave_apply\nAction Input: {{\"employee_name\": \"{name}\", \"leave_type\": \"年假\", "
                f"\"days\": {days}, \"start_date\": \"{TOMORROW}\"}}")
    return (f"Thought: 请假申请已提交成功，任务完成。\n"
            f"Final Answer: 已成功为「{name}」提交 {days} 天年假申请（{TOMORROW} 开始）。"
            f"{_last_observation(user_text)[:180]}")


def react_employee_info(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    name = _extract_name(goal, user_text)
    if obs_n == 0:
        return (f"Thought: 需要查询员工「{name}」的基本信息。\n"
                f"Action: employee_lookup\nAction Input: {{\"name\": \"{name}\"}}")
    return f"Thought: 已获取员工信息。\nFinal Answer: 查询结果：{_last_observation(user_text)[:200]}"


def react_leave_balance(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    name = _extract_name(goal, user_text)
    if obs_n == 0:
        return ("Thought: 查询年假余额前先确认员工身份。\n"
                f"Action: employee_lookup\nAction Input: {{\"name\": \"{name}\"}}")
    if obs_n == 1:
        return (f"Thought: 身份已确认，查询 {name} 的年假余额。\n"
                f"Action: leave_balance\nAction Input: {{\"employee_name\": \"{name}\"}}")
    return f"Thought: 已拿到余额信息。\nFinal Answer: 该员工年假余额情况：{_last_observation(user_text)[:200]}"


def react_kb(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    if obs_n == 0:
        return ("Thought: 这是制度政策类问题，应检索企业知识库后再回答。\n"
                f"Action: kb_search\nAction Input: {{\"query\": \"{goal[:40]}\"}}")
    last = _last_observation(user_text)
    if _has_error(last):
        # 知识库未命中：明确拒答而不是编造（幻觉治理）
        return ("Thought: 知识库未命中任何相关条款，不应编造答案。\n"
                f"Final Answer: 抱歉，企业知识库中未检索到与「{goal[:30]}」相关的内容，建议咨询人事或行政部门。")
    return ("Thought: 已检索到相关制度条款，可以基于检索结果回答。\n"
            f"Final Answer: 根据企业知识库检索结果：{last[:260]}")


def react_weather_single(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    city = _extract_city(goal)
    if obs_n == 0:
        return (f"Thought: 查询 {city} 的天气需要调用天气接口（该接口可能不稳定，若失败会自动重试）。\n"
                f"Action: weather_api\nAction Input: {{\"city\": \"{city}\", \"date\": \"明天\"}}")
    return f"Thought: 天气已获取。\nFinal Answer: {_city_answer(city, _last_observation(user_text))}"


def _city_answer(city: str, obs: str) -> str:
    return f"{city}明天天气：{obs[:150]}"


def react_weather_compare(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    obs_all = [o.strip() for o in re.findall(r"Observation:\s*(.*)", user_text)]
    if obs_n == 0:
        return ("Thought: 对比两地天气，先查北京。\n"
                "Action: weather_api\nAction Input: {\"city\": \"北京\", \"date\": \"明天\"}")
    if obs_n == 1 and not any("上海" in o for o in obs_all):
        return ("Thought: 再查上海。\n"
                "Action: weather_api\nAction Input: {\"city\": \"上海\", \"date\": \"明天\"}")
    bj = next((o for o in obs_all if "北京" in o), "")
    sh = next((o for o in obs_all if "上海" in o), "")
    return ("Thought: 两地天气都已拿到，给出对比与出行建议。\n"
            f"Final Answer: 对比结果——北京：{bj[:90]}；上海：{sh[:90]}。"
            f"综合建议：上海有小雨请随身携带雨具，北京天气更适宜户外安排，出行前请再次确认动态。")


def react_news_degrade(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    query = re.sub(r"[，。！？]|帮我|查一下|搜一下|搜索|看看", "", goal).strip() or "AI 行业新闻"
    if obs_n == 0:
        return (f"Thought: 优先使用专用新闻检索工具查询「{query}」。\n"
                f"Action: news_search\nAction Input: {{\"query\": \"{query}\"}}")
    last = _last_observation(user_text)
    if obs_n == 1 and _has_error(last):
        return ("Thought: news_search 上游持续失败，已触发降级。改用通用搜索工具 web_search 完成查询。\n"
                f"Action: web_search\nAction Input: {{\"query\": \"{query}\"}}")
    return ("Thought: 已通过检索工具获取到结果。\n"
            f"Final Answer: 已为你检索到相关资讯（news_search 不可用后经 web_search 降级完成）：{last[:220]}")


def react_convert(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    m1 = re.search(r"([\d.]+)\s*(千米|公里|米|厘米|英寸|英尺|英里|千克|公斤|克|磅|摄氏度|华氏度)", goal)
    m2 = re.search(r"(?:换算成|转换成|等于多少|是多少)\s*(英里|千米|公里|米|英寸|英尺|千克|磅|华氏度)", goal)
    value = float(m1.group(1)) if m1 else 5.0
    src = m1.group(2) if m1 else "千米"
    dst = m2.group(1) if m2 else ("英里" if src in ("千米", "公里") else "千米")
    if "时区" in goal or obs_n >= 1 and "时差" in goal:
        return react_timezone(user_text)
    if obs_n == 0:
        return ("Thought: 单位换算可以通过 MCP 远程工具 unit_convert 完成。\n"
                f"Action: unit_convert\nAction Input: {{\"value\": {value}, \"from_unit\": \"{src}\", \"to_unit\": \"{dst}\"}}")
    return f"Thought: 换算完成。\nFinal Answer: 换算结果：{_last_observation(user_text)[:120]}"


def react_timezone(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    m = re.search(r"(北京|上海|伦敦|纽约|东京|UTC[+-]?\d*).{0,6}?(\d{1,2}[点:：]\d{0,2}?)", goal)
    time_s = m.group(2) if m else "15:00"
    src = "UTC+8"
    dst = "UTC+0" if "伦敦" in goal else ("UTC-5" if "纽约" in goal else ("UTC+9" if "东京" in goal else "UTC+0"))
    if obs_n == 0:
        return ("Thought: 时区换算使用 MCP 远程工具 timezone_convert。\n"
                f"Action: timezone_convert\nAction Input: {{\"time\": \"{time_s}\", \"from_tz\": \"{src}\", \"to_tz\": \"{dst}\"}}")
    return f"Thought: 时区换算完成。\nFinal Answer: 换算结果：{_last_observation(user_text)[:120]}"


def react_calc(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    if obs_n == 0:
        return ("Thought: 数值计算应使用 calculator 工具保证准确性。\n"
                f"Action: calculator\nAction Input: {{\"expression\": \"{_extract_expr(goal)}\"}}")
    return f"Thought: 计算完成。\nFinal Answer: 计算结果：{_last_observation(user_text)[:80]}"


def react_email(user_text: str) -> str:
    goal, obs_n = _goal_of(user_text), _obs_count(user_text)
    if obs_n == 0:
        m_to = re.search(r"([\w*._%+-]+@[\w*.-]+\.\w+)", goal)
        m_sub = re.search(r"主题为?\s*([^，。,\s]+?)的邮件", goal) or \
            re.search(r"主题[:：]?\s*([^，。,\s]+)", goal)
        to = m_to.group(1) if m_to else "hr@demo.com"
        subject = m_sub.group(1) if m_sub else "通知"
        return ("Thought: 发送邮件需要调用邮件工具。\n"
                f"Action: send_email\nAction Input: {{\"to\": \"{to}\", "
                f"\"subject\": \"{subject}\", \"body\": \"您好，这是一封来自 Agent 平台的自动邮件。\"}}")
    return f"Thought: 邮件已发送。\nFinal Answer: {_last_observation(user_text)[:150]}"


def react_memory_recall(user_text: str) -> str:
    name = _memory_value(user_text, "name") or "（尚未记住你的名字）"
    dept = _memory_value(user_text, "dept")
    return ("Thought: 这是长期记忆召回问题，无需调用工具。\n"
            f"Final Answer: 当然记得！你是{name}"
            + (f"，来自{dept}" if dept else "")
            + "。这条信息来自我的长期记忆（跨会话持久化，混合检索召回）。")


def react_time(user_text: str) -> str:
    if _obs_count(user_text) == 0:
        return ("Thought: 获取当前时间需要调用时钟工具。\n"
                "Action: current_time\nAction Input: {}")
    return f"Thought: 已获取时间。\nFinal Answer: 当前时间：{_last_observation(user_text)[:80]}"


def react_memory_intro(user_text: str) -> str:
    goal = _goal_of(user_text)
    m = re.search(r"我叫([\u4e00-\u9fa5]{2,4})", goal)
    name = m.group(1) if m else "朋友"
    m2 = re.search(r"([\u4e00-\u9fa5]{2,8})部门", goal)
    dept = m2.group(1) if m2 else ""
    return (f"Thought: 用户在自我介绍，无需调用工具，直接回应并记住关键信息。\n"
            f"Final Answer: 你好，{name}！已记住你的信息{('（' + dept + '）') if dept else ''}，"
            f"后续为你查询假期、报销等事务时会直接使用。")


_REACT_SCENARIOS: list[tuple[str, re.Pattern, object]] = [
    ("hr_leave", re.compile(r"(提交|申请).{0,8}(请假|年假)|(请假|年假).{0,10}(提交|申请)"), react_hr_leave),
    ("leave_balance", re.compile(r"(还剩|余额|几天).{0,6}年假|年假.{0,6}(还剩|余额|几天)"), react_leave_balance),
    ("weather_compare", re.compile(r"天气.*(对比|比较|两地|分别|建议)|对比.*(天气)"), react_weather_compare),
    ("weather", re.compile(r"天气|气温|下雨"), react_weather_single),
    ("news", re.compile(r"新闻|资讯|行业动态|搜一下|搜索"), react_news_degrade),
    ("convert", re.compile(r"换算|英寸|英里|公斤|磅|华氏|时区|时差"), react_convert),
    ("calc", re.compile(r"计算|等于多少|[0-9]\s*[+\-*/]\s*[0-9]"), react_calc),
    ("time", re.compile(r"几点|现在的时间|今天.*(几号|日期)|当前时间"), react_time),
    ("memory_recall", re.compile(r"我是谁|记得我|我的名字"), react_memory_recall),
    ("email", re.compile(r"发(一封|个)?邮件|发.{0,6}邮件|主题为"), react_email),
    ("employee_info", re.compile(r"员工(信息|资料)|人员信息"), react_employee_info),
    ("kb", re.compile(r"制度|政策|规定|顺延|报销|流程|手册|补贴|考勤|试用期|工资|福利"), react_kb),
    ("memory_intro", re.compile(r"我叫|我是|记一下"), react_memory_intro),
]


def react_dispatch(user_text: str) -> str:
    goal = _goal_of(user_text)
    for _, pattern, handler in _REACT_SCENARIOS:
        if pattern.search(goal):
            return handler(user_text)
    # 通用兜底：先查知识库，再基于结果作答
    return react_kb(user_text)


# ---------------------------------------------------------------- Plan 脚本

def plan_dispatch(user_text: str) -> str:
    """[MODE: PLAN] —— 按目标关键词输出任务拆解 JSON。

    兼容两种编排器：Plan-and-Execute 请求 steps；Multi-Agent 请求 tasks+executor
    （通过 Prompt 中的 executor 字样区分）。
    """
    import json
    m = re.search(r"目标：(.*?)$", user_text, re.MULTILINE)
    goal = m.group(1) if m else user_text
    name = _extract_name(goal, user_text)
    is_multi_agent = "executor" in user_text

    if re.search(r"(提交|申请).{0,8}(请假|年假)|(请假|年假).{0,10}(提交|申请)", goal):
        if is_multi_agent:
            tasks = [
                {"id": "t1", "description": f"查询{name}的员工信息", "executor": "research", "depends_on": []},
                {"id": "t2", "description": f"查询{name}的年假余额", "executor": "research", "depends_on": ["t1"]},
                {"id": "t3", "description": f"为{name}提交年假请假申请", "executor": "ops", "depends_on": ["t2"]},
            ]
            return json.dumps({"tasks": tasks}, ensure_ascii=False)
        steps = [
            {"id": "s1", "description": f"查询{name}的员工信息，确认在系统中存在", "depends_on": []},
            {"id": "s2", "description": f"查询{name}的年假余额，判断是否足够", "depends_on": ["s1"]},
            {"id": "s3", "description": f"为{name}提交年假请假申请", "depends_on": ["s2"]},
        ]
    elif re.search(r"天气.*(对比|比较|两地|分别|建议)|对比.*天气", goal):
        if is_multi_agent:
            tasks = [
                {"id": "t1", "description": "查询北京明天的天气", "executor": "research", "depends_on": []},
                {"id": "t2", "description": "查询上海明天的天气", "executor": "ops", "depends_on": []},
                {"id": "t3", "description": "综合两地天气给出对比与出行建议", "executor": "research", "depends_on": ["t1", "t2"]},
            ]
            return json.dumps({"tasks": tasks}, ensure_ascii=False)
        steps = [
            {"id": "s1", "description": "查询北京明天的天气", "depends_on": []},
            {"id": "s2", "description": "查询上海明天的天气", "depends_on": []},
            {"id": "s3", "description": "综合两地天气给出对比与出行建议", "depends_on": ["s1", "s2"]},
        ]
    elif re.search(r"新闻|资讯|行业动态", goal):
        steps = [
            {"id": "s1", "description": "检索最新相关资讯", "depends_on": []},
            {"id": "s2", "description": "汇总检索结果并输出要点", "depends_on": ["s1"]},
        ]
    elif re.search(r"制度|政策|顺延|报销|流程|考勤|补贴", goal):
        steps = [
            {"id": "s1", "description": "在企业知识库中检索相关制度条款", "depends_on": []},
            {"id": "s2", "description": "基于检索到的条款回答用户问题", "depends_on": ["s1"]},
        ]
    else:
        steps = [{"id": "s1", "description": f"直接理解并回答：{goal[:60]}", "depends_on": []}]
    return json.dumps({"steps": steps}, ensure_ascii=False)


# ---------------------------------------------------------------- 其他模式

def verify_dispatch(user_text: str) -> str:
    import json
    if _has_error(user_text):
        return json.dumps({"pass": False, "reason": "执行结果中包含错误信息（ERROR/失败），未达成目标"},
                          ensure_ascii=False)
    if re.search(r"无法|抱歉，当前|暂不支持", user_text):
        return json.dumps({"pass": False, "reason": "回答未能解决用户目标"}, ensure_ascii=False)
    return json.dumps({"pass": True, "reason": "结果完整、与目标一致，未发现错误标记"}, ensure_ascii=False)


def reflect_dispatch(user_text: str) -> str:
    import json
    if re.search(r"超时|Timeout|重试|不稳定|5\d\d|Connection", user_text):
        analysis = {"stage": "tool_call", "cause": "工具调用持续超时/上游不稳定",
                    "suggestion": "改用备用工具（降级）或调整参数后重试"}
    elif re.search(r"未知工具|Unknown tool|not a valid|没有这个工具", user_text):
        analysis = {"stage": "planning", "cause": "规划引用了不存在的工具",
                    "suggestion": "重新规划，仅使用工具列表中存在的工具"}
    elif re.search(r"未找到|没有找到|无结果|为空", user_text):
        analysis = {"stage": "context", "cause": "检索/上下文未覆盖所需信息",
                    "suggestion": "更换检索关键词或扩大检索范围后重试"}
    else:
        analysis = {"stage": "planning", "cause": "规划步骤与实际执行能力不匹配",
                    "suggestion": "拆解为更小的步骤，并显式绑定可用工具"}
    return json.dumps(analysis, ensure_ascii=False)


def extract_dispatch(user_text: str) -> str:
    import json
    facts = []
    m = re.search(r"我叫([\u4e00-\u9fa5]{2,4})", user_text)
    if m:
        facts.append({"subject": "user", "key": "name", "value": m.group(1)})
    m = re.search(r"(研发部|人事部|市场部|财务部|产品部|运营部|销售部)", user_text)
    if m:
        facts.append({"subject": "user", "key": "dept", "value": m.group(1)})
    m = re.search(r"工号\s*([A-Za-z0-9]+)", user_text)
    if m:
        facts.append({"subject": "user", "key": "employee_id", "value": m.group(1)})
    return json.dumps({"facts": facts}, ensure_ascii=False)


def synth_dispatch(user_text: str) -> str:
    results = re.findall(r"\[(s\d+)\]\s*(.*?)(?=\n\[s\d+\]|\Z)", user_text, re.DOTALL)
    joined = "；".join(f"{sid}：{txt.strip()[:200]}" for sid, txt in results)
    if "天气" in user_text:
        return (f"两地天气对比与建议：{joined[:400]}。综合来看出行请关注温差与降水，"
                f"建议随身携带雨具并预留路途时间。")
    return f"任务已完成，各步骤结果：{joined[:600]}"


def compress_dispatch(user_text: str) -> str:
    body = user_text[-400:]
    return f"【历史摘要】此前对话要点：{body[:200]}……（已压缩，详情见长期记忆）"


def chat_dispatch(user_text: str) -> str:
    if re.search(r"你好|您好|hi|hello", user_text, re.IGNORECASE):
        return "你好！我是企业级 Agent 平台演示助手，可以帮你查询假期、办理请假、检索公司制度、查天气、算数据等。有什么可以帮你？"
    if "能力" in user_text or "能做什么" in user_text:
        return ("我支持：① 多轮工具调用（员工/假期/邮件/文件等 12+ 工具）② 企业知识库检索问答 "
                "③ 任务规划与并行执行 ④ 失败自动重试/降级/反思纠错。试试：『查一下张三的年假余额并帮他提交3天请假』")
    return f"（演示模式）收到输入：「{user_text[:60]}」。当前为 MockLLM 离线脚本，配置真实模型后可自由对话。"


# ---------------------------------------------------------------- 客户端

class MockLLM:
    """脚本化 LLM：按 [MODE] 分发到对应处理器；provider 名称为 mock-strong / mock-mini。"""

    name = "mock"

    async def chat(self, messages, *, json_mode: bool = False, role: str = "default", **_) -> LLMResponse:
        msgs = normalize_messages(messages)
        sys_text = "\n".join(m.content for m in msgs if m.role == "system")
        user_text = "\n".join(m.content for m in msgs if m.role == "user")
        if "[MODE: PLAN]" in sys_text:
            out = plan_dispatch(user_text)
        elif "[MODE: REACT]" in sys_text:
            out = react_dispatch(user_text)
        elif "[MODE: VERIFY]" in sys_text:
            out = verify_dispatch(user_text)
        elif "[MODE: REFLECT]" in sys_text:
            out = reflect_dispatch(user_text)
        elif "[MODE: EXTRACT]" in sys_text:
            out = extract_dispatch(user_text)
        elif "[MODE: SYNTH]" in sys_text:
            out = synth_dispatch(user_text)
        elif "[MODE: COMPRESS]" in sys_text:
            out = compress_dispatch(user_text)
        else:
            out = chat_dispatch(user_text)
        model = "mock-strong" if role in ("planner", "verifier", "synthesizer") else "mock-mini"
        return LLMResponse(content=out, model=model, usage=usage_from_text(model, msgs, out))
