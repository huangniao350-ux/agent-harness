"""AgentHarness —— 企业级 Agent 平台核心能力演示实现。

模块分层总览：
- llm:          模型抽象层（MockLLM / OpenAI 兼容客户端 + 模型路由）
- tools:        统一 Tool Registry（进程内工具 + MCP 远程工具）+ 弹性执行层
- memory / rag: 三层记忆、上下文压缩、混合检索（BM25 + 向量 + RRF）
- guardrails:   输入/输出安全护栏与审计
- harness:      单 Agent 核心执行链路（ReAct / Plan-and-Execute / 反思 / Checkpoint）
- multiagent:   规划-执行-校验多智能体协同编排
- observability: Trace / Replay / Metrics 可观测与评估
- runtime:      组合根，装配所有组件
"""

__version__ = "0.1.0"
