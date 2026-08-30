"""Tool call orchestration boundary for the agent runtime.

工具调用编排层：负责将模型输出的 tool_calls 解析、分批、执行，
并对结果做预算控制和归一化处理。
"""

from __future__ import annotations

import json
import os
import traceback as tb
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

# --- 数据类定义 ---

@dataclass(frozen=True)
class ToolObservation:
    """工具执行后的观察结果，作为 tool role 消息返回给模型。"""
    tool_name: str
    tool_call_id: str
    observation: str               # 发给模型的观察文本（可能被截断）
    raw_observation: str | None = None   # 原始完整输出，保留用于后续预算处理
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolResultBudget:
    """工具结果预算限制：单个工具最大字节数 + 整轮消息总字节数。"""
    max_tool_bytes: int
    max_message_bytes: int


@dataclass(frozen=True)
class ToolCallPlan:
    """单个工具调用的解析计划，包含是否可并发执行的标记。"""
    call: dict[str, Any]              # 原始 tool_call 字典
    tool_name: str
    tool_call_id: str
    parsed_input: dict[str, Any]      # 解析后的工具参数
    parse_error: Exception | None     # 参数解析是否失败
    concurrency_safe: bool            # 是否可与其他工具并发执行


@dataclass(frozen=True)
class ToolBatch:
    """一个执行批次：同一批次内的调用要么全部并发，要么全部串行。"""
    concurrency_safe: bool
    calls: list[ToolCallPlan]


class ToolOrchestrator:
    """执行模型工具调用，保证模型输出顺序不被并发打乱。

    核心流程：
    1. plan_tool_calls() — 解析并分类每个 tool_call
    2. partition_tool_calls() — 按并发安全性分组为批次
    3. run() — 逐批次执行（并发批次用线程池，串行批次逐个执行）
    4. _apply_result_budget() — 对超长结果做截断，控制 token 消耗
    """

    # 只读工具，可安全并发执行
    SAFE_TOOL_NAMES = {"Read", "Grep", "Glob", "ListFiles"}
    # 写/副作用工具，必须串行执行以保持顺序一致性
    UNSAFE_TOOL_NAMES = {"Write", "Edit", "MultiEdit", "Bash", "Task", "Skill", "TodoWrite"}

    def __init__(self, host: Any):
        self.host = host

    # --- Transcript 记录辅助方法 ---

    def _get_transcript_recorder(self):
        return getattr(self.host, "transcript_recorder", None)

    def _get_transcript_run_id(self) -> str:
        run_id = getattr(self.host, "_active_transcript_run_id", None)
        if run_id is not None:
            return str(run_id)
        return f"run-{getattr(self.host, '_run_id', 0)}"

    def _record_tool_lifecycle(
        self,
        *,
        step: int,
        tool_name: str,
        tool_call_id: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        recorder = self._get_transcript_recorder()
        if recorder is None:
            return
        recorder.record_tool_lifecycle(
            run_id=self._get_transcript_run_id(),
            step=step,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status=status,
            payload=payload or {},
        )

    def run(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """主入口：解析 → 分批 → 按批次执行 → 应用结果预算。

        批次间保持顺序（先并发批次完成才进入下一个批次），
        同一并发批次内的工具在线程池中并行执行。
        """
        plans = self.plan_tool_calls(tool_calls)
        batches = self.partition_tool_calls(plans)
        self._log_plan(trace_logger, step, batches)

        observations: list[ToolObservation] = []
        for batch_index, batch in enumerate(batches):
            self._log_batch_start(trace_logger, step, batch_index, batch)
            batch_observations = (
                self._run_batch_concurrently(batch, step=step, trace_logger=trace_logger)
                if batch.concurrency_safe
                else self._run_batch_serially(batch, step=step, trace_logger=trace_logger)
            )
            self._log_batch_end(trace_logger, step, batch_index, batch, batch_observations)
            observations.extend(batch_observations)

        # 空结果归一化：将空输出包装为标准 JSON 结构
        observations = [self._normalize_empty_result(obs) for obs in observations]
        return self._apply_result_budget(observations, step=step, trace_logger=trace_logger)

    def run_serial(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """全串行执行模式：忽略并发标记，所有工具逐个执行。"""
        plans = self.plan_tool_calls(tool_calls)
        observations = self._run_batch_serially(
            ToolBatch(concurrency_safe=False, calls=plans),
            step=step,
            trace_logger=trace_logger,
        )
        return observations

    def plan_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[ToolCallPlan]:
        """将原始 tool_call 字典解析为 ToolCallPlan 列表。

        对每个调用：提取工具名、生成 call_id、解析参数 JSON、
        判断并发安全性。
        """
        plans: list[ToolCallPlan] = []
        for call in tool_calls:
            tool_name = call.get("name") or "unknown_tool"
            tool_call_id = call.get("id") or f"call_{uuid.uuid4().hex}"
            raw_args = call.get("arguments") or {}
            parsed_input, parse_error = self.host._ensure_json_input(raw_args)
            plans.append(
                ToolCallPlan(
                    call=call,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    parsed_input=parsed_input if isinstance(parsed_input, dict) else {},
                    parse_error=parse_error,
                    concurrency_safe=self.is_concurrency_safe(tool_name, parse_error),
                )
            )
        return plans

    def is_concurrency_safe(self, tool_name: str, parse_error: Exception | None) -> bool:
        """判断工具是否可以并发执行。

        规则：解析失败的视为不安全；在 SAFE_TOOL_NAMES 中的可并发；
        在 UNSAFE_TOOL_NAMES 中的必须串行；未知工具默认串行。
        """
        if parse_error is not None:
            return False
        if tool_name in self.SAFE_TOOL_NAMES:
            return True
        if tool_name in self.UNSAFE_TOOL_NAMES:
            return False
        return False

    def partition_tool_calls(self, plans: list[ToolCallPlan]) -> list[ToolBatch]:
        """将 plans 按并发安全性分组为批次。

        连续的并发安全调用会合并到同一个批次（可并行执行），
        任何不安全调用会单独成为一个批次（串行执行），并中断连续合并。
        """
        batches: list[ToolBatch] = []
        for plan in plans:
            if (
                batches
                and plan.concurrency_safe
                and batches[-1].concurrency_safe
            ):
                # 与前一个批次同为并发安全 → 合并
                batches[-1].calls.append(plan)
                continue
            batches.append(ToolBatch(concurrency_safe=plan.concurrency_safe, calls=[plan]))
        return batches

    def _run_batch_serially(
        self,
        batch: ToolBatch,
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """串行执行批次：逐个调用 _execute_plan。"""
        observations: list[ToolObservation] = []
        for plan in batch.calls:
            observations.append(self._execute_plan(plan, step=step, trace_logger=trace_logger))
        return observations

    def _run_batch_concurrently(
        self,
        batch: ToolBatch,
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """并发执行批次：使用线程池并行调用，但按原始索引顺序返回结果。"""
        observations: dict[int, ToolObservation] = {}
        max_workers = min(len(batch.calls), self._get_max_concurrency())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                offset: executor.submit(self._execute_plan, plan, step=step, trace_logger=trace_logger)
                for offset, plan in enumerate(batch.calls)
            }
            for offset, future in futures.items():
                observations[offset] = future.result()
        # 按原始偏移量排序，保持模型输出顺序
        return [observations[idx] for idx in range(len(batch.calls))]

    def _execute_plan(self, plan: ToolCallPlan, *, step: int, trace_logger) -> ToolObservation:
        """执行单个工具调用计划。

        流程：记录生命周期 → 若参数解析失败返回错误 → 否则执行工具 → 包装为 ToolObservation。
        """
        self._record_tool_lifecycle(
            step=step,
            tool_name=plan.tool_name,
            tool_call_id=plan.tool_call_id,
            status="requested",
            payload={"args": plan.parsed_input},
        )
        if plan.parse_error is not None:
            # 参数解析失败：直接返回错误观察，不实际调用工具
            observation = self._parse_error_observation(plan.parse_error)
            self._log_parse_error(trace_logger, step, plan.tool_name, plan.tool_call_id, plan.parse_error)
            self._record_tool_lifecycle(
                step=step,
                tool_name=plan.tool_name,
                tool_call_id=plan.tool_call_id,
                status="failed",
                payload={"error": str(plan.parse_error), "args": plan.parsed_input},
            )
        else:
            trace_logger.log_event(
                "tool_call",
                {"tool": plan.tool_name, "args": plan.parsed_input, "tool_call_id": plan.tool_call_id},
                step=step,
            )
            observation = self._execute_one(
                plan.tool_name,
                plan.parsed_input,
                plan.tool_call_id,
                trace_logger,
                step,
            )

        return ToolObservation(
            tool_name=plan.tool_name,
            tool_call_id=plan.tool_call_id,
            observation=observation,
        )

    def _execute_one(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool_call_id: str,
        trace_logger,
        step: int,
    ) -> str:
        """实际调用工具执行器，记录生命周期（started → completed/failed）。"""
        host = self.host
        self._record_tool_lifecycle(
            step=step,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            status="started",
            payload={"args": tool_input},
        )
        try:
            # 优先使用新版 tool_executor，回退到旧版 _execute_tool
            if hasattr(host, "tool_executor") and host.tool_executor is not None:
                observation = host.tool_executor.execute(
                    tool_name,
                    tool_input,
                    trace_logger=trace_logger,
                    step=step,
                )
            else:
                observation = host._execute_tool(tool_name, tool_input)
            self._log_tool_result(trace_logger, step, tool_name, observation)
            lifecycle_status, lifecycle_payload = self._tool_lifecycle_result_payload(observation)
            self._record_tool_lifecycle(
                step=step,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status=lifecycle_status,
                payload=lifecycle_payload,
            )
            return observation
        except Exception as exc:
            # 工具执行异常：包装为标准 JSON 错误格式返回给模型
            error_result = {
                "status": "error",
                "error": {"code": "EXECUTION_ERROR", "message": str(exc)},
                "data": {},
            }
            trace_logger.log_event(
                "error",
                {
                    "stage": "tool_execution",
                    "error_code": "EXECUTION_ERROR",
                    "message": str(exc),
                    "tool": tool_name,
                    "traceback": tb.format_exc(),
                },
                step=step,
            )
            self._record_tool_lifecycle(
                step=step,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                status="failed",
                payload={"error": str(exc), "args": tool_input},
            )
            return json.dumps(error_result, ensure_ascii=False)

    def _tool_lifecycle_result_payload(self, observation: str) -> tuple[str, dict[str, Any]]:
        """根据工具返回的 JSON 判断执行状态：status=="error" → failed，其他 → completed。"""
        try:
            result_obj = json.loads(observation)
        except json.JSONDecodeError:
            return "completed", {"result_text": observation}
        status = str(result_obj.get("status") or "")
        lifecycle_status = "failed" if status == "error" else "completed"
        return lifecycle_status, {"result": result_obj}

    def _log_tool_result(self, trace_logger, step: int, tool_name: str, observation: str) -> None:
        try:
            result_obj = json.loads(observation)
            trace_logger.log_event("tool_result", {"tool": tool_name, "result": result_obj}, step=step)
        except json.JSONDecodeError:
            trace_logger.log_event(
                "tool_result",
                {"tool": tool_name, "result": {"text": observation}},
                step=step,
            )

    def _parse_error_observation(self, parse_err: Exception) -> str:
        error_result = {
            "status": "error",
            "error": {
                "code": "INVALID_PARAM",
                "message": f"Tool arguments parse error: {parse_err}",
            },
            "data": {},
        }
        return json.dumps(error_result, ensure_ascii=False)

    def _log_parse_error(
        self,
        trace_logger,
        step: int,
        tool_name: str,
        tool_call_id: str,
        parse_err: Exception,
    ) -> None:
        trace_logger.log_event(
            "error",
            {
                "stage": "tool_call_parse",
                "error_code": "INVALID_PARAM",
                "message": str(parse_err),
                "tool": tool_name,
                "tool_call_id": tool_call_id,
            },
            step=step,
        )

    def _get_max_concurrency(self) -> int:
        """最大并发数，由环境变量 MYCODEAGENT_MAX_TOOL_CONCURRENCY 控制，默认 4。"""
        raw_value = os.getenv("MYCODEAGENT_MAX_TOOL_CONCURRENCY", "4")
        try:
            return max(1, int(raw_value))
        except ValueError:
            return 4

    def _get_result_budget(self) -> ToolResultBudget:
        """从环境变量读取结果预算配置。"""
        def _read_env(name: str, default: int) -> int:
            try:
                return max(1, int(os.getenv(name, str(default))))
            except ValueError:
                return default

        return ToolResultBudget(
            max_tool_bytes=_read_env("MYCODEAGENT_MAX_TOOL_RESULT_BYTES", 50000),
            max_message_bytes=_read_env("MYCODEAGENT_MAX_TOOL_MESSAGE_BYTES", 200000),
        )

    def _byte_len(self, text: str) -> int:
        """UTF-8 编码后的字节长度。"""
        return len((text or "").encode("utf-8"))

    def _apply_result_budget(
        self,
        observations: list[ToolObservation],
        *,
        step: int,
        trace_logger,
    ) -> list[ToolObservation]:
        """对工具结果应用预算控制，防止超长输出撑爆上下文窗口。

        两级截断策略：
        1. 单工具级：单个工具输出超过 max_tool_bytes → 截断并保留 raw_observation
        2. 聚合级：所有结果总字节超过 max_message_bytes → 从最大的开始截断（跳过已截断的）
        """
        budget = self._get_result_budget()
        trace_logger.log_event(
            "tool_result_budget_start",
            {
                "tool_count": len(observations),
                "max_tool_bytes": budget.max_tool_bytes,
                "max_message_bytes": budget.max_message_bytes,
            },
            step=step,
        )
        budgeted: list[ToolObservation] = []
        replaced_count = 0
        raw_total_bytes = 0
        visible_total_bytes = 0

        for obs in observations:
            raw_text = obs.raw_observation if obs.raw_observation is not None else obs.observation
            raw_bytes = self._byte_len(raw_text)
            raw_total_bytes += raw_bytes
            next_obs = obs
            # 第一级：单个工具结果超限 → 截断
            if raw_bytes > budget.max_tool_bytes:
                from runtime.observation_store import force_truncate_observation

                compressed = force_truncate_observation(
                    obs.tool_name,
                    raw_text,
                    self.host.project_root,
                )
                visible_bytes = self._byte_len(compressed)
                metadata = {
                    **(obs.metadata or {}),
                    "budgeted": True,
                    "replaced": True,
                    "reason": "single_tool_budget",
                    "raw_bytes": raw_bytes,
                    "visible_bytes": visible_bytes,
                }
                try:
                    parsed = json.loads(compressed)
                    full_output_path = (
                        parsed.get("data", {})
                        .get("truncation", {})
                        .get("full_output_path")
                    )
                    if full_output_path:
                        metadata["full_output_path"] = full_output_path
                except json.JSONDecodeError:
                    pass
                next_obs = ToolObservation(
                    tool_name=obs.tool_name,
                    tool_call_id=obs.tool_call_id,
                    observation=compressed,
                    raw_observation=raw_text,
                    metadata=metadata,
                )
                replaced_count += 1
                trace_logger.log_event(
                    "tool_result_budget_item",
                    {
                        "tool_call_id": obs.tool_call_id,
                        "reason": "single_tool_budget",
                        "replaced": True,
                        "raw_bytes": raw_bytes,
                        "visible_bytes": visible_bytes,
                    },
                    step=step,
                )
            budgeted.append(next_obs)
            visible_total_bytes += self._byte_len(next_obs.observation)

        # 第二级：聚合总量超限 → 从最大的结果开始截断
        if visible_total_bytes > budget.max_message_bytes:
            # 按观察结果大小降序排列，从最大的开始截断
            indexed = list(enumerate(budgeted))
            indexed.sort(key=lambda item: self._byte_len(item[1].observation), reverse=True)
            for idx, obs in indexed:
                if visible_total_bytes <= budget.max_message_bytes:
                    break
                if (obs.metadata or {}).get("replaced") is True:
                    continue  # 已在第一级被截断的跳过
                from runtime.observation_store import force_truncate_observation

                source_text = obs.raw_observation if obs.raw_observation is not None else obs.observation
                previous_visible = self._byte_len(obs.observation)
                compressed = force_truncate_observation(
                    obs.tool_name,
                    source_text,
                    self.host.project_root,
                )
                visible_bytes = self._byte_len(compressed)
                metadata = {
                    **(obs.metadata or {}),
                    "budgeted": True,
                    "replaced": True,
                    "reason": "aggregate_message_budget",
                    "raw_bytes": self._byte_len(source_text),
                    "visible_bytes": visible_bytes,
                }
                try:
                    parsed = json.loads(compressed)
                    full_output_path = (
                        parsed.get("data", {})
                        .get("truncation", {})
                        .get("full_output_path")
                    )
                    if full_output_path:
                        metadata["full_output_path"] = full_output_path
                except json.JSONDecodeError:
                    pass
                budgeted[idx] = ToolObservation(
                    tool_name=obs.tool_name,
                    tool_call_id=obs.tool_call_id,
                    observation=compressed,
                    raw_observation=source_text,
                    metadata=metadata,
                )
                visible_total_bytes = visible_total_bytes - previous_visible + visible_bytes
                replaced_count += 1
                trace_logger.log_event(
                    "tool_result_budget_item",
                    {
                        "tool_call_id": obs.tool_call_id,
                        "reason": "aggregate_message_budget",
                        "replaced": True,
                        "raw_bytes": self._byte_len(source_text),
                        "visible_bytes": visible_bytes,
                    },
                    step=step,
                )

        trace_logger.log_event(
            "tool_result_budget_end",
            {
                "tool_count": len(observations),
                "max_tool_bytes": budget.max_tool_bytes,
                "max_message_bytes": budget.max_message_bytes,
                "raw_total_bytes": raw_total_bytes,
                "visible_total_bytes": visible_total_bytes,
                "replaced_count": replaced_count,
            },
            step=step,
        )
        return budgeted

    def _normalize_empty_result(self, obs: ToolObservation) -> ToolObservation:
        """空结果归一化：将无输出的工具结果包装为标准的 success JSON 结构。"""
        if not self._is_empty_observation(obs.observation):
            return obs

        payload = {
            "status": "success",
            "data": {},
            "text": f"{obs.tool_name} completed with no output.",
        }
        metadata = {**(obs.metadata or {}), "budgeted": True, "reason": "empty_result", "replaced": False}
        return ToolObservation(
            tool_name=obs.tool_name,
            tool_call_id=obs.tool_call_id,
            observation=json.dumps(payload, ensure_ascii=False),
            raw_observation=obs.observation,
            metadata=metadata,
        )

    def _is_empty_observation(self, text: str) -> bool:
        """判断工具结果是否为"空"：空字符串、空白文本、或 JSON 中无实质内容。"""
        if not text or not str(text).strip():
            return True
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
        if parsed.get("error"):
            return False  # 错误信息不是空结果
        payload_text = parsed.get("text")
        payload_data = parsed.get("data")
        if isinstance(payload_text, str) and payload_text.strip():
            return False
        if payload_data:
            return False
        return True

    def _log_plan(self, trace_logger, step: int, batches: list[ToolBatch]) -> None:
        trace_logger.log_event(
            "tool_orchestration_plan",
            {
                "batch_count": len(batches),
                "batches": [
                    {
                        "concurrency_safe": batch.concurrency_safe,
                        "tool_names": [plan.tool_name for plan in batch.calls],
                    }
                    for batch in batches
                ],
            },
            step=step,
        )

    def _log_batch_start(self, trace_logger, step: int, batch_index: int, batch: ToolBatch) -> None:
        trace_logger.log_event(
            "tool_batch_start",
            {
                "batch_index": batch_index,
                "concurrency_safe": batch.concurrency_safe,
                "tool_count": len(batch.calls),
                "tool_names": [plan.tool_name for plan in batch.calls],
            },
            step=step,
        )

    def _log_batch_end(
        self,
        trace_logger,
        step: int,
        batch_index: int,
        batch: ToolBatch,
        observations: list[ToolObservation],
    ) -> None:
        trace_logger.log_event(
            "tool_batch_end",
            {
                "batch_index": batch_index,
                "concurrency_safe": batch.concurrency_safe,
                "tool_count": len(batch.calls),
                "completed_count": len(observations),
            },
            step=step,
        )
