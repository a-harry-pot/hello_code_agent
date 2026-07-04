"""Plan and Solve Agent实现 - 分解规划与逐步执行的智能体"""

import ast
import time
from typing import Optional, List, Dict
from core.agent import Agent
from core.llm import HelloAgentsLLM
from core.config import Config
from core.message import Message
from core.session_logger import SessionLogger, _NoopLogger

# 默认规划器提示词模板
DEFAULT_PLANNER_PROMPT = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 默认执行器提示词模板
DEFAULT_EXECUTOR_PROMPT = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
"""


class Planner:
    """规划器 - 负责将复杂问题分解为简单步骤"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None,
                 session_logger: Optional[SessionLogger] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_PLANNER_PROMPT
        self._log = session_logger or _NoopLogger()

    def plan(self, question: str, **kwargs) -> List[str]:
        """
        生成执行计划

        Args:
            question:要解决的问题
            **kwargs:LLM调用的参数

        Returns：
            步骤列表
        """
        prompt = self.prompt_template.format(question=question)
        message = [{"role": "user", "content": prompt}]

        self._log.console("--- 正在生成计划 ---")
        response_text = self.llm_client.invoke(message, **kwargs) or ""
        self._log.console(f"✅ 计划已生成:\n{response_text}")

        try:
            # 提取Python代码块中的列表
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            result = plan if isinstance(plan, list) else []
            self._log.event("plan_generated", {
                "plan": result,
                "plan_length": len(result),
            })
            return result
        except (ValueError, SyntaxError, IndexError) as e:
            self._log.console(f"❌ 解析计划时出错: {e}")
            self._log.console(f"原始响应: {response_text}")
            self._log.event("error", {
                "context": "plan_parse",
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            self._log.inc_error()
            return []
        except Exception as e:
            self._log.console(f"❌ 解析计划时发生未知错误: {e}")
            self._log.event("error", {
                "context": "plan_parse",
                "error_type": type(e).__name__,
                "error_message": str(e),
            })
            self._log.inc_error()
            return []


class Executor:
    """执行器 负责按照计划逐步执行"""

    def __init__(self, llm_client: HelloAgentsLLM, prompt_template: Optional[str] = None,
                 session_logger: Optional[SessionLogger] = None):
        self.llm_client = llm_client
        self.prompt_template = prompt_template if prompt_template else DEFAULT_EXECUTOR_PROMPT
        self._log = session_logger or _NoopLogger()

    def execute(self, question: str, plan: List[str], **kwargs) -> str:
        """
        按照计划执行任务

        :param question: 原始问题
        :param plan: 执行计划
        :param kwargs: LLM调用参数
        :return: 最终答案
        """

        history = ""
        final_answer = ""

        self._log.console("\n--- 正在执行计划 ---")
        for i, step in enumerate(plan, 1):
            self._log.console(f"\n-> 正在执行步骤 {i}/{len(plan)}: {step}")
            self._log.event("step_executed", {
                "step_index": i,
                "step_total": len(plan),
                "step_description": step,
                "status": "started",
            })
            prompt = self.prompt_template.format(
                question=question,
                plan=plan,
                history=history if history else "无",
                current_step=step
            )
            messages = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.invoke(messages, **kwargs) or ""

            history += f"步骤{i}:{step}\n结果: {response_text}\n\n"
            final_answer += response_text
            self._log.console(f"✅ 步骤 {i} 已完成，结果: {final_answer}")

            self._log.event("step_executed", {
                "step_index": i,
                "step_total": len(plan),
                "step_description": step,
                "step_result": response_text[:4000],
                "status": "completed",
            })

        return final_answer


class PlanAndSolveAgent(Agent):
    """
    Plan and Solve Agent 分解规划与逐步执行的智能体

    这个Agent能够：
    1. 将复杂问题分解为简单步骤
    2. 按照计划逐步执行
    3. 维护执行历史和上下文
    4. 得出最终答案

    适合多步骤推理，数学问题，复杂分析等任务
    """

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            custom_prompts: Optional[Dict[str, str]] = None,
            session_logger: Optional[SessionLogger] = None,
    ):
        """
        初始化PlanAndSolveAgent

        Args:
            name: Agent名称
            llm: LLM实例
            system_prompt: 系统提示词
            config: 配置对象
            custom_prompts: 自定义提示词模板 {"planner": "", "executor": ""}
            session_logger: 会话日志记录器（可选）
        """
        super().__init__(name, llm, system_prompt, config, session_logger=session_logger)

        # 设置提示词模板，用户自定义优先，否则使用默认模板
        if custom_prompts:
            planner_prompt = custom_prompts.get("planner")
            executor_prompt = custom_prompts.get("executor")
        else:
            planner_prompt = None
            executor_prompt = None

        # Planner/Executor will share the agent's logger after it's set
        self._planner_prompt = planner_prompt
        self._executor_prompt = executor_prompt
        self.planner: Optional[Planner] = None
        self.executor: Optional[Executor] = None

    def _ensure_planner_executor(self):
        """延迟创建 Planner/Executor，确保它们能拿到 agent 的 logger"""
        if self.planner is None:
            self.planner = Planner(self.llm, self._planner_prompt, session_logger=self._log)
            self.executor = Executor(self.llm, self._executor_prompt, session_logger=self._log)

    def run(self, input_text: str, **kwargs) -> str:
        """
        运行Plan and Solve Agent
        :param input_text: 要解决的问题
        :param kwargs: 其他参数
        :return: 最终答案
        """
        import time
        t_start = time.time()

        self._ensure_planner_executor()

        self._log.console(f"\n🤖 {self.name} 开始处理问题: {input_text}")
        self._log.event("user_input", {
            "content": input_text,
            "content_length": len(input_text),
        })

        # 1.生成计划
        plan = self.planner.plan(input_text, **kwargs)
        if not plan:
            final_answer = "无法生成有效的行动计划，任务终止"
            self._log.console(f"\n--- 任务终止 ---\n{final_answer}")

            self.add_message(Message(input_text, "user"))
            self.add_message(Message(final_answer, "assistant"))

            self._log.event("agent_answer", {
                "answer": final_answer,
                "total_duration_ms": int((time.time() - t_start) * 1000),
            })
            self._log.close(final_answer=final_answer)
            return final_answer

        # 2. 执行计划
        final_answer = self.executor.execute(input_text, plan, **kwargs)
        self._log.console(f"\n--- 任务完成 ---\n最终答案: {final_answer}")

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        self._log.event("agent_answer", {
            "answer": final_answer,
            "plan_length": len(plan),
            "total_duration_ms": int((time.time() - t_start) * 1000),
        })
        self._log.close(final_answer=final_answer)

        return final_answer
