"""评测执行器：加载 evals/tasks.jsonl，逐任务运行 Agent 并产出评估报告。

三类场景 + 特殊场景：
- tool_calling：工具调用链路（期望工具集 + 关键词校验）
- rag_qa：知识库检索问答
- plan_execute / multi_agent：规划并行与多智能体
- multi_turn_memory：多轮会话记忆（同会话多轮，验证跨轮记忆召回）
- guardrails：护栏拦截（expect_blocked）

判分规则（评测运行时实时计算）：
- success：任务执行成功 + 关键词全部出现在最终回答（或按预期被拦截）
- tool accuracy：期望工具全部被调用（子集校验）
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_harness.harness.state import MODE_MULTI_AGENT, MODE_PLAN_EXECUTE, MODE_REACT
from agent_harness.observability.metrics import EvalReport, TaskOutcome
from agent_harness.runtime import AgentRuntime

TASKS_FILE = Path(__file__).parent / "tasks.jsonl"


def load_tasks(limit: int | None = None) -> list[dict]:
    tasks = [json.loads(line) for line in TASKS_FILE.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.startswith("#")]
    return tasks[:limit] if limit else tasks


async def _run_task(runtime: AgentRuntime, task: dict, idx: int) -> TaskOutcome:
    goals = task.get("turns") or [task["goal"]]
    mode = task.get("mode", MODE_REACT)
    expected_tools = task.get("expected_tools") or []
    expected_keywords = task.get("expected_keywords") or []
    session_id = f"eval-{task['id']}-{idx}"
    used_tools: list[str] = []
    last_result = None
    for goal in goals:
        last_result = await runtime.run(goal, mode=mode, session_id=session_id)
        for call in last_result.tool_calls:
            if call["tool"] not in used_tools:
                used_tools.append(call["tool"])

    answer = last_result.answer if last_result else ""
    expect_blocked = task.get("expect_blocked", False)
    if expect_blocked:
        success = (last_result is not None and not last_result.success
                   and all(k in answer for k in expected_keywords))
        error = "" if success else "预期应被护栏拦截"
    else:
        kw_ok = all(k in answer for k in expected_keywords)
        success = bool(last_result and last_result.success and kw_ok)
        error = "" if success else _failure_reason(last_result, expected_keywords, answer)

    return TaskOutcome(task_id=task["id"], category=task["category"], goal=goals[-1],
                       success=success, expected_tools=expected_tools, used_tools=used_tools,
                       steps=last_result.steps if last_result else 0,
                       latency_ms=last_result.latency_ms if last_result else 0,
                       tokens=last_result.tokens_used if last_result else 0,
                       answer=answer, error=error,
                       ttft_ms=last_result.ttft_ms if last_result else 0.0,
                       cache_hits=last_result.cache_hits if last_result else 0)


def _failure_reason(result, keywords: list[str], answer: str) -> str:
    if result is None:
        return "任务未执行"
    if not result.success:
        return result.error or "任务失败"
    missing = [k for k in keywords if k not in answer]
    if missing:
        return f"回答缺少关键词: {missing}；回答片段: {answer[:80]}"
    return "未知原因"


async def run_eval(runtime: AgentRuntime, limit: int | None = None) -> EvalReport:
    report = EvalReport()
    await runtime.startup()
    try:
        tasks = load_tasks(limit)
        for idx, task in enumerate(tasks):
            outcome = await _run_task(runtime, task, idx)
            report.outcomes.append(outcome)
        # 报告落盘
        out_dir = runtime.config.data_dir / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        import time
        path = out_dir / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.md"
        path.write_text("# Agent 评估报告\n\n```\n" + report.render() + "\n```\n", encoding="utf-8")
        print(f"报告已写入: {path}")
    finally:
        await runtime.close()
    return report
