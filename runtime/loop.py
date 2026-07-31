"""Runtime runner for the canonical single-agent turn loop."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from runtime.completion import (
    CompletionGateVerdict,
    DeterministicCompletionVerifier,
    build_completion_candidate,
    collect_verification_evidence,
    infer_completion_requirements,
)
from runtime.input_preprocess import preprocess_input
from runtime.model_errors import ModelErrorKind, classify_model_error
from runtime.state import LoopState, TerminalReason, TransitionReason


class RuntimeRunner:
    """Canonical single-agent turn loop.

    核心职责：驱动单个 Agent 的 ReAct 循环（Think → Act → Observe），
    直到模型返回最终文本或达到最大步数限制。

    主要流程：
    1. run() — 入口，预处理用户输入后进入 react_loop
    2. _react_loop() — 主循环，每步：构建上下文 → 调用 LLM → 处理输出
    3. 输出分三类：tool_calls（执行工具后继续）/ 最终文本（经 completion gate 校验）/ 错误（重试或终止）
    """

    def __init__(self, host: Any):
        self.host = host

    def _transition(
        self,
        state: LoopState,
        reason: TransitionReason,
        trace_logger,
        *,
        step: int | None = None,
        details: dict[str, Any] | None = None,
        **changes: Any,
    ) -> LoopState:
        # 将 changes 分为两类：属于 LoopState 字段的直接更新，其余的合并到 details
        next_step = step if step is not None else state.step
        state_field_names = set(LoopState.__dataclass_fields__)
        state_changes = {key: value for key, value in changes.items() if key in state_field_names}
        detail_changes = {key: value for key, value in changes.items() if key not in state_field_names}
        payload_details = details if details is not None else detail_changes
        next_state = state.next(reason, step=next_step, details=payload_details, **state_changes)
        if trace_logger:
            trace_logger.log_event(
                "state_transition",
                {
                    "step": next_state.step,
                    "turn_count": next_state.turn_count,
                    "reason": reason.value,
                    "message_count": len(next_state.messages),
                    "details": payload_details,
                },
                step=step if step is not None else next_state.step,
            )
        self._record_transcript_state_transition(
            from_state=state.transition.reason.value if state.transition else None,
            to_state=reason.value,
            reason=reason.value,
            step=step if step is not None else next_state.step,
            details=payload_details,
        )
        return next_state

    def _terminal(
        self,
        reason: TerminalReason,
        trace_logger,
        *,
        step: int = 0,
        **details: Any,
    ) -> None:
        """记录循环终止事件（日志 + transcript），不抛异常，只做记录。"""
        if trace_logger:
            trace_logger.log_event(
                "terminal",
                {"reason": reason.value, "details": details},
                step=step,
            )
        self._record_transcript_terminal(reason=reason.value, step=step, details=details)

    def _get_transcript_recorder(self):
        return getattr(self.host, "transcript_recorder", None)

    def _get_transcript_run_id(self) -> str:
        run_id = getattr(self.host, "_active_transcript_run_id", None)
        if run_id is not None:
            return str(run_id)
        fallback = getattr(self.host, "_run_id", 0)
        return f"run-{fallback}"

    def _record_transcript_message(
        self,
        *,
        role: str,
        content: str,
        step: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        recorder = self._get_transcript_recorder()
        if recorder is None:
            return
        recorder.record_message(
            run_id=self._get_transcript_run_id(),
            step=step,
            role=role,
            content=content,
            metadata=metadata or {},
        )

    def _record_transcript_state_transition(
        self,
        *,
        from_state: str | None,
        to_state: str,
        reason: str,
        step: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        recorder = self._get_transcript_recorder()
        if recorder is None:
            return
        recorder.record_state_transition(
            run_id=self._get_transcript_run_id(),
            step=step,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            details=details or {},
        )

    def _record_transcript_checkpoint(self, *, step: int, checkpoint_id: str, payload: dict[str, Any]) -> None:
        recorder = self._get_transcript_recorder()
        if recorder is None:
            return
        recorder.record_checkpoint(
            run_id=self._get_transcript_run_id(),
            step=step,
            checkpoint_id=checkpoint_id,
            payload=payload,
        )

    def _record_active_transcript_checkpoint(self, *, step: int) -> None:
        compact_store = getattr(getattr(self.host, "context_engine", None), "compact_store", None)
        checkpoint = getattr(compact_store, "active_checkpoint", None)
        if checkpoint is None:
            return
        self._record_transcript_checkpoint(
            step=step,
            checkpoint_id=checkpoint.id,
            payload={
                "summary": checkpoint.summary,
                "source_message_count": checkpoint.source_message_count,
                "retain_start_idx": checkpoint.retain_start_idx,
                "messages_compacted": checkpoint.messages_compacted,
                "created_at": checkpoint.created_at,
                "metadata": dict(checkpoint.metadata),
            },
        )

    def _record_transcript_terminal(self, *, reason: str, step: int, details: dict[str, Any]) -> None:
        recorder = self._get_transcript_recorder()
        if recorder is None:
            return
        recorder.record_terminal(
            run_id=self._get_transcript_run_id(),
            step=step,
            reason=reason,
            details=details,
        )

    def _trace_model_request_state(
        self,
        trace_logger,
        *,
        tools_schema: list[dict[str, Any]],
        step: int,
    ) -> None:
        """在每次 LLM 调用前记录 prompt 分层的指纹和 tool schema 指纹，用于追踪变更。"""
        host = self.host
        # 追踪各 prompt 层的指纹变化（constitution, tool contracts, project rules, runtime signals）
        if hasattr(host.context_builder, "get_prompt_assembly"):
            prompt_assembly = host.context_builder.get_prompt_assembly()
            previous_prompt_fingerprints = getattr(host, "_last_prompt_fingerprints", {})
            current_prompt_fingerprints = {
                "constitution": prompt_assembly.constitution_fingerprint,
                "tool_contracts": prompt_assembly.tool_contracts_fingerprint,
                "project_rules": prompt_assembly.project_rules_fingerprint,
                "runtime_signals": prompt_assembly.runtime_signals_fingerprint,
            }
            changed_layers = [
                layer
                for layer, value in current_prompt_fingerprints.items()
                if previous_prompt_fingerprints.get(layer) not in (None, value)
            ]
            trace_logger.log_event(
                "prompt_assembly",
                {
                    "constitution_fingerprint": prompt_assembly.constitution_fingerprint,
                    "tool_contracts_fingerprint": prompt_assembly.tool_contracts_fingerprint,
                    "project_rules_fingerprint": prompt_assembly.project_rules_fingerprint,
                    "runtime_signals_fingerprint": prompt_assembly.runtime_signals_fingerprint,
                    "system_fingerprint": prompt_assembly.system_fingerprint,
                    "stable_message_count": len(prompt_assembly.stable_messages),
                    "runtime_signal_count": len(prompt_assembly.runtime_signal_messages),
                    "changed_layers": changed_layers,
                },
                step=step,
            )
            host._last_prompt_fingerprints = current_prompt_fingerprints

        # 对 tool schema 做 SHA256 指纹，检测是否与上一步不同
        tool_schema_payload = json.dumps(
            tools_schema,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        tool_schema_fingerprint = hashlib.sha256(
            tool_schema_payload.encode("utf-8")
        ).hexdigest()
        previous_tool_schema_fingerprint = getattr(host, "_last_tool_schema_fingerprint", None)
        trace_logger.log_event(
            "tool_schema",
            {
                "fingerprint": tool_schema_fingerprint,
                "tool_count": len(tools_schema),
                "changed": previous_tool_schema_fingerprint not in (
                    None,
                    tool_schema_fingerprint,
                ),
            },
            step=step,
        )
        host._last_tool_schema_fingerprint = tool_schema_fingerprint

    def _append_user_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """向历史记录追加用户消息，兼容 metadata 参数可能不被旧版 history_manager 支持的情况。"""
        append_user = self.host.history_manager.append_user
        if metadata is None:
            append_user(content)
            return
        try:
            append_user(content, metadata=metadata)
        except TypeError:
            append_user(content)

    def _get_completion_verifier(self):
        """懒初始化 completion verifier，默认使用确定性验证器。"""
        verifier = getattr(self.host, "completion_verifier", None)
        if verifier is not None:
            return verifier
        verifier = DeterministicCompletionVerifier()
        self.host.completion_verifier = verifier
        return verifier

    def _get_model_recovery_limit(self, kind: ModelErrorKind) -> int:
        """根据错误类型返回对应的重试上限。
        - EMPTY_RESPONSE: 默认 1 次重试
        - PROMPT_TOO_LONG: 固定 1 次（触发压缩后重试）
        - MAX_OUTPUT: 默认 0 次（不重试，直接失败）
        """
        host = self.host
        if kind is ModelErrorKind.EMPTY_RESPONSE:
            return int(getattr(host, "empty_response_retry_limit", 1) or 1)
        if kind is ModelErrorKind.PROMPT_TOO_LONG:
            return 1
        if kind is ModelErrorKind.MAX_OUTPUT:
            return int(getattr(host, "max_output_recovery_limit", 0) or 0)
        return 0

    def _increment_model_recovery_count(self, state: LoopState, kind: ModelErrorKind) -> dict[str, int]:
        """递增指定错误类型的重试计数，返回新的计数字典。"""
        counts = dict(state.model_recovery_counts)
        counts[kind.value] = counts.get(kind.value, 0) + 1
        return counts

    def _trace_model_error_classified(
        self,
        trace_logger,
        *,
        step: int,
        stage: str,
        kind: ModelErrorKind,
        retry_count: int,
        retry_limit: int,
        message: str,
        finish_reason: str | None = None,
    ) -> None:
        trace_logger.log_event(
            "model_error_classified",
            {
                "stage": stage,
                "kind": kind.value,
                "retry_count": retry_count,
                "retry_limit": retry_limit,
                "message": message,
                "finish_reason": finish_reason,
            },
            step=step,
        )

    def _trace_model_recovery_attempted(
        self,
        trace_logger,
        *,
        step: int,
        kind: ModelErrorKind,
        retry_count: int,
        retry_limit: int,
        action: str,
    ) -> None:
        trace_logger.log_event(
            "model_recovery_attempted",
            {
                "kind": kind.value,
                "retry_count": retry_count,
                "retry_limit": retry_limit,
                "action": action,
            },
            step=step,
        )

    def _trace_model_recovery_failed(
        self,
        trace_logger,
        *,
        step: int,
        kind: ModelErrorKind,
        retry_count: int,
        retry_limit: int,
        reason: str,
    ) -> None:
        trace_logger.log_event(
            "model_recovery_failed",
            {
                "kind": kind.value,
                "retry_count": retry_count,
                "retry_limit": retry_limit,
                "reason": reason,
            },
            step=step,
        )

    def run(self, input_text: str, **kwargs) -> str:
        """Agent 主循环入口。

        1. 预处理用户输入（文件引用解析等）
        2. 将处理后输入追加到历史记录
        3. 进入 _react_loop 主循环
        4. finally 中确保记录 run_end 事件
        """
        host = self.host
        show_raw = kwargs.pop("show_raw", False)
        if not show_raw:
            host.last_response_raw = None

        if host.console_progress:
            host._console("⏳ Agent 正在处理，请稍候...")

        # 刷新 skills prompt 并注入到 context_builder
        host._refresh_skills_prompt()
        host.context_builder.set_skills_prompt(host._skills_prompt)
        preprocess_result = preprocess_input(input_text)
        processed_input = preprocess_result.processed_input

        if preprocess_result.mentioned_files:
            mentioned = ", ".join(preprocess_result.mentioned_files)
            if host.console_verbose:
                host._console(f"\n📎 检测到文件引用: {mentioned}")
                if preprocess_result.truncated_count > 0:
                    host._console(f"   (另有 {preprocess_result.truncated_count} 个文件被省略)")
            elif host.logger.isEnabledFor(10):
                host.logger.debug("检测到文件引用: %s", mentioned)
                if preprocess_result.truncated_count > 0:
                    host.logger.debug("另有 %d 个文件被省略", preprocess_result.truncated_count)

        trace_logger = host.trace_logger
        # 递增 run_id 并关联到 transcript
        host._run_id += 1
        run_id = host._run_id
        host._active_transcript_run_id = f"run-{run_id}"

        host._log_system_messages_if_needed(trace_logger)
        trace_logger.log_event(
            "run_start",
            {
                "run_id": run_id,
                "input": input_text,
                "processed": processed_input,
            },
            step=0,
        )

        self._append_user_message(processed_input)
        self._record_transcript_message(role="user", content=processed_input, step=0, metadata={})
        trace_logger.log_event(
            "user_input",
            {"text": input_text, "processed": processed_input},
            step=0,
        )
        host._log_message_write(trace_logger, "user", processed_input, {}, step=0)

        if host.console_verbose:
            host._console(f"\n⚙️ Engine 启动: {input_text}")
        elif host.logger.isEnabledFor(10):
            host.logger.debug("Engine 启动: %s", input_text)

        response_text = ""
        try:
            response_text = self._react_loop(
                pending_input=processed_input,
                show_raw=show_raw,
                trace_logger=trace_logger,
            )
        finally:
            # 确保无论正常/异常退出都记录 run_end
            trace_logger.log_event(
                "run_end",
                {"run_id": run_id, "final": response_text if "response_text" in locals() else ""},
                step=0,
            )
            host._active_transcript_run_id = None
        if host.console_progress:
            host._console("✅ Agent 已完成")

        host.logger.debug("response=%s", response_text)
        host.logger.info(
            "history_size=%d, rounds=%d",
            host.history_manager.get_message_count(),
            host.history_manager.get_rounds_count(),
        )
        return response_text

    def _react_loop(self, pending_input: str, show_raw: bool, trace_logger) -> str:
        """ReAct 主循环：每步构建上下文 → 调用 LLM → 处理输出，直到终止。

        每步的处理逻辑：
        1. 构建模型上下文（含历史压缩检测）
        2. 调用 LLM，处理三类异常（EMPTY_RESPONSE / PROMPT_TOO_LONG / MAX_OUTPUT）
        3. 解析响应：
           - 有 tool_calls → 执行工具，追加观察结果，继续下一轮
           - 无 tool_calls → 通过 completion gate 校验：
             - PASS/UNVERIFIED → 返回最终文本
             - BLOCKED → 注入反馈消息，重试（有次数上限）
        """
        host = self.host
        tool_choice = "auto"
        completion_retry_limit = int(getattr(host, "completion_gate_retry_limit", 2) or 2)
        state = LoopState(
            messages=[],
            step=1,
            turn_count=1,
            tool_choice=tool_choice,
        )
        state = self._transition(
            state,
            TransitionReason.USER_INPUT,
            trace_logger,
            step=0,
            pending_input_len=len(pending_input or ""),
        )

        # 主步进循环，最多 max_steps 步
        for step in range(1, host.max_steps + 1):
            tools_schema = host._get_openai_tools_for_current_mode()
            # Agent 团队模式：导出 team 运行时状态并注入到系统提示
            if (
                host.enable_agent_teams
                and host.team_manager
                and hasattr(host.context_builder, "set_runtime_system_blocks")
            ):
                events = host.team_manager.drain_events()
                runtime_state = host.team_manager.export_state()
                runtime_blocks = host._format_runtime_system_blocks(events, runtime_state=runtime_state)
                host.context_builder.set_runtime_system_blocks(runtime_blocks)

            self._trace_model_request_state(
                trace_logger,
                tools_schema=tools_schema,
                step=step,
            )

            if host.console_verbose:
                host._console(f"\n--- Step {step}/{host.max_steps} ---")
            elif host.console_progress:
                host._console(f"… Step {step}/{host.max_steps}")
            elif host.logger.isEnabledFor(10):
                host.logger.debug("Step %d/%d", step, host.max_steps)

            # 上下文压缩检测：当历史消息过长时自动压缩
            compact_info = host.context_engine.compact_if_needed(
                history_manager=host.history_manager,
                pending_input=pending_input,
                step=step,
                trace_logger=trace_logger,
            )
            if compact_info.get("compacted"):
                self._record_active_transcript_checkpoint(step=step)
                # 压缩后重建上下文视图，确保模型只看到压缩后的精简历史
                state = self._transition(
                    state,
                    TransitionReason.CONTEXT_COMPACTED,
                    trace_logger,
                    step=step,
                    compact_attempted=True,
                    details={
                        "checkpoint_id": compact_info.get("checkpoint_id"),
                        "messages_compacted": compact_info.get("messages_compacted"),
                        "retain_start_idx": compact_info.get("retain_start_idx"),
                    },
                )
                final_context = host.context_engine.build_model_view(
                    history_manager=host.history_manager,
                    pending_input=pending_input,
                    step=step,
                    trace_logger=trace_logger,
                ).messages
                trace_logger.log_event(
                    "history_compression_final_context",
                    {"message_count": len(final_context), "messages": final_context},
                    step=step,
                )

                if host.console_verbose:
                    host._console("\n📦 触发历史压缩...")
                    host._console("✅ 压缩完成，当前轮次数: %d" % host.history_manager.get_rounds_count())
                    host._print_context_preview(final_context)
                elif host.logger.isEnabledFor(10):
                    host.logger.debug("触发历史压缩")
                    host.logger.debug("压缩完成，当前轮次数: %d", host.history_manager.get_rounds_count())
                    host._print_context_preview(final_context)

            # 构建当前步骤的模型上下文视图
            model_view = host.context_engine.build_model_view(
                history_manager=host.history_manager,
                pending_input=pending_input,
                step=step,
                trace_logger=trace_logger,
            )
            messages = model_view.messages
            base_messages = messages
            state = state.update(step=step, messages=messages)

            trace_logger.log_event(
                "context_build",
                {
                    "message_count": len(messages),
                    "history_count": model_view.history_message_count,
                    "source_message_count": model_view.source_message_count,
                    "projection_mode": model_view.projection_mode,
                },
                step=step,
            )

            response_text = ""
            tool_calls: list[dict[str, Any]] = []
            reasoning_content = None
            response_meta: dict[str, Any] = {}

            # 内层重试循环：处理 LLM 调用异常和空响应/超长输出
            while True:
                try:
                    # 核心：调用 LLM
                    raw_response = host.llm.invoke_raw(messages, tools=tools_schema, tool_choice=tool_choice)
                except Exception as exc:
                    # LLM 调用层异常分类与恢复
                    classification = classify_model_error(error=exc)
                    retry_count = state.model_recovery_counts.get(classification.kind.value, 0)
                    retry_limit = self._get_model_recovery_limit(classification.kind)
                    self._trace_model_error_classified(
                        trace_logger,
                        step=step,
                        stage="model_invoke",
                        kind=classification.kind,
                        retry_count=retry_count,
                        retry_limit=retry_limit,
                        message=classification.message,
                        finish_reason=classification.finish_reason,
                    )

                    if (
                        classification.kind is ModelErrorKind.PROMPT_TOO_LONG
                        and retry_count < retry_limit
                        and hasattr(host.context_engine, "reactive_compact")
                    ):
                        # PROMPT_TOO_LONG 恢复策略：触发反应式压缩后重试
                        next_retry_count = retry_count + 1
                        recovery_counts = self._increment_model_recovery_count(state, classification.kind)
                        self._trace_model_recovery_attempted(
                            trace_logger,
                            step=step,
                            kind=classification.kind,
                            retry_count=next_retry_count,
                            retry_limit=retry_limit,
                            action="reactive_compact",
                        )
                        compact_info = host.context_engine.reactive_compact(
                            history_manager=host.history_manager,
                            pending_input=pending_input,
                            step=step,
                            trace_logger=trace_logger,
                        )
                        if compact_info.get("compacted"):
                            self._record_active_transcript_checkpoint(step=step)
                            # 压缩成功后重建上下文，continue 回到 LLM 调用重试
                            state = self._transition(
                                state,
                                TransitionReason.MODEL_RECOVERY_RETRY,
                                trace_logger,
                                step=step,
                                model_recovery_counts=recovery_counts,
                                compact_attempted=True,
                                last_model_error_kind=classification.kind.value,
                                last_model_error_stage="model_invoke",
                                last_error=classification.message,
                                details={
                                    "error_kind": classification.kind.value,
                                    "retry_count": next_retry_count,
                                    "retry_limit": retry_limit,
                                    "action": "reactive_compact",
                                    "checkpoint_id": compact_info.get("checkpoint_id"),
                                },
                            )
                            model_view = host.context_engine.build_model_view(
                                history_manager=host.history_manager,
                                pending_input=pending_input,
                                step=step,
                                trace_logger=trace_logger,
                            )
                            messages = model_view.messages
                            base_messages = messages
                            state = state.update(messages=messages)
                            trace_logger.log_event(
                                "context_build",
                                {
                                    "message_count": len(messages),
                                    "history_count": model_view.history_message_count,
                                    "source_message_count": model_view.source_message_count,
                                    "projection_mode": model_view.projection_mode,
                                },
                                step=step,
                            )
                            continue

                        # 压缩失败或不可压缩：记录恢复失败
                        self._trace_model_recovery_failed(
                            trace_logger,
                            step=step,
                            kind=classification.kind,
                            retry_count=next_retry_count,
                            retry_limit=retry_limit,
                            reason=str(compact_info.get("reason") or "reactive_compact_failed"),
                        )
                        state = self._transition(
                            state,
                            TransitionReason.MODEL_RECOVERY_FAILED,
                            trace_logger,
                            step=step,
                            model_recovery_counts=recovery_counts,
                            compact_attempted=True,
                            last_model_error_kind=classification.kind.value,
                            last_model_error_stage="model_invoke",
                            last_error=classification.message,
                            details={
                                "error_kind": classification.kind.value,
                                "retry_count": next_retry_count,
                                "retry_limit": retry_limit,
                                "action": "reactive_compact",
                                "reason": compact_info.get("reason"),
                            },
                        )
                    else:
                        # 不可恢复的模型错误
                        self._trace_model_recovery_failed(
                            trace_logger,
                            step=step,
                            kind=classification.kind,
                            retry_count=retry_count,
                            retry_limit=retry_limit,
                            reason="non_recoverable" if retry_limit == 0 else "retry_exhausted",
                        )
                        state = self._transition(
                            state,
                            TransitionReason.MODEL_RECOVERY_FAILED,
                            trace_logger,
                            step=step,
                            last_model_error_kind=classification.kind.value,
                            last_model_error_stage="model_invoke",
                            last_error=classification.message,
                            details={
                                "error_kind": classification.kind.value,
                                "retry_count": retry_count,
                                "retry_limit": retry_limit,
                            },
                        )

                    self._terminal(
                        TerminalReason.MODEL_ERROR,
                        trace_logger,
                        step=step,
                        error_kind=classification.kind.value,
                        message=classification.message,
                        retry_count=retry_count,
                        retry_limit=retry_limit,
                    )
                    return "抱歉，我无法在限定步数内完成这个任务。"

                # --- LLM 调用成功，解析响应 ---
                if show_raw:
                    host.last_response_raw = (
                        raw_response.model_dump()
                        if hasattr(raw_response, "model_dump")
                        else raw_response
                    )

                response_text = host._extract_content(raw_response) or ""
                reasoning_content = host._extract_reasoning_content(raw_response)
                usage = host._extract_usage(raw_response)
                if usage and usage.get("total_tokens") is not None:
                    host.context_engine.record_usage(usage["total_tokens"])

                response_meta = host._extract_response_meta(raw_response)
                tool_calls = host._extract_tool_calls(raw_response)
                raw_dump = host._extract_raw_response(raw_response)
                trace_logger.log_event(
                    "model_output",
                    {
                        "raw": response_text,
                        "usage": usage,
                        "meta": response_meta,
                        "raw_response": raw_dump,
                        "tool_calls": tool_calls,
                    },
                    step=step,
                )

                if host.console_verbose and reasoning_content:
                    display_reasoning = reasoning_content
                    if len(display_reasoning) > 1200:
                        display_reasoning = display_reasoning[:1200] + "...(truncated)"
                    host._console(f"\n🧠 Reasoning: {display_reasoning}\n")

                # 对响应内容做错误分类（仅检查 EMPTY_RESPONSE 和 MAX_OUTPUT）
                classification = None
                candidate_error = classify_model_error(
                    response_text=response_text,
                    tool_calls=tool_calls,
                    response_meta=response_meta,
                )
                if candidate_error.kind in {ModelErrorKind.EMPTY_RESPONSE, ModelErrorKind.MAX_OUTPUT}:
                    classification = candidate_error

                if classification is None:
                    break  # 响应正常，跳出重试循环

                # 响应异常处理
                retry_count = state.model_recovery_counts.get(classification.kind.value, 0)
                retry_limit = self._get_model_recovery_limit(classification.kind)
                self._trace_model_error_classified(
                    trace_logger,
                    step=step,
                    stage="model_response",
                    kind=classification.kind,
                    retry_count=retry_count,
                    retry_limit=retry_limit,
                    message=classification.message,
                    finish_reason=classification.finish_reason,
                )

                if classification.kind is ModelErrorKind.EMPTY_RESPONSE and retry_count < retry_limit:
                    # 空响应恢复策略：型重新追加提示消息让模生成
                    next_retry_count = retry_count + 1
                    recovery_counts = self._increment_model_recovery_count(state, classification.kind)
                    hint = "上次 content 为空且未返回 tool_calls，请在 content 中回复最终答案，或使用工具调用。"
                    messages = base_messages + [{"role": "user", "content": hint}]
                    self._trace_model_recovery_attempted(
                        trace_logger,
                        step=step,
                        kind=classification.kind,
                        retry_count=next_retry_count,
                        retry_limit=retry_limit,
                        action="retry_with_hint",
                    )
                    state = self._transition(
                        state,
                        TransitionReason.MODEL_EMPTY_RETRY,
                        trace_logger,
                        step=step,
                        model_recovery_counts=recovery_counts,
                        last_model_error_kind=classification.kind.value,
                        last_model_error_stage="model_response",
                        last_error=classification.message,
                        last_response_meta=response_meta,
                        details={
                            "error_kind": classification.kind.value,
                            "finish_reason": response_meta.get("finish_reason"),
                            "retry_count": next_retry_count,
                            "retry_limit": retry_limit,
                        },
                    )
                    trace_logger.log_event(
                        "empty_response_retry",
                        {
                            "finish_reason": response_meta.get("finish_reason"),
                            "content_len": response_meta.get("content_len"),
                            "reasoning_len": response_meta.get("reasoning_len"),
                            "hint": hint,
                        },
                        step=step,
                    )
                    if host.console_verbose:
                        host._console("⚠️ LLM返回空响应，追加提示后重试一次")
                    else:
                        host.logger.warning("LLM返回空响应，追加提示后重试一次")
                    continue

                # 重试耗尽或不可恢复
                self._trace_model_recovery_failed(
                    trace_logger,
                    step=step,
                    kind=classification.kind,
                    retry_count=retry_count,
                    retry_limit=retry_limit,
                    reason="retry_exhausted" if retry_limit else "non_recoverable",
                )
                transition_reason = (
                    TransitionReason.MODEL_EMPTY_FAILED
                    if classification.kind is ModelErrorKind.EMPTY_RESPONSE
                    else TransitionReason.MODEL_RECOVERY_FAILED
                )
                state_changes: dict[str, Any] = {
                    "last_response_meta": response_meta,
                    "last_model_error_kind": classification.kind.value,
                    "last_model_error_stage": "model_response",
                    "last_error": classification.message,
                }
                if classification.kind is ModelErrorKind.MAX_OUTPUT:
                    state_changes["max_output_recovery_count"] = state.max_output_recovery_count + 1
                state = self._transition(
                    state,
                    transition_reason,
                    trace_logger,
                    step=step,
                    details={
                        "error_kind": classification.kind.value,
                        "finish_reason": response_meta.get("finish_reason"),
                        "retry_count": retry_count,
                        "retry_limit": retry_limit,
                    },
                    **state_changes,
                )
                terminal_reason = (
                    TerminalReason.EMPTY_RESPONSE_FAILED
                    if classification.kind is ModelErrorKind.EMPTY_RESPONSE
                    else TerminalReason.MODEL_ERROR
                )
                self._terminal(
                    terminal_reason,
                    trace_logger,
                    step=step,
                    error_kind=classification.kind.value,
                    finish_reason=response_meta.get("finish_reason"),
                    retry_count=retry_count,
                    retry_limit=retry_limit,
                )
                if classification.kind is ModelErrorKind.EMPTY_RESPONSE:
                    trace_logger.log_event(
                        "error",
                        {
                            "stage": "llm_response",
                            "error_code": "INTERNAL_ERROR",
                            "message": "Empty response",
                            "meta": response_meta,
                        },
                        step=step,
                    )
                return "抱歉，我无法在限定步数内完成这个任务。"

            # --- 响应处理完毕，根据内容分支 ---
            if tool_calls:
                # 分支 A：模型返回了工具调用
                state = self._transition(
                    state,
                    TransitionReason.MODEL_RETURNED_TOOL_CALLS,
                    trace_logger,
                    step=step,
                    last_tool_calls=tool_calls,
                    last_response_meta=response_meta,
                    details={"tool_count": len(tool_calls)},
                )
                for call in tool_calls:
                    if not call.get("id"):
                        call["id"] = f"call_{uuid.uuid4().hex}"
                # 以 assistant 角色写入历史
                assistant_content = str(response_text or "")
                host.history_manager.append_assistant(
                    content=assistant_content,
                    metadata={
                        "step": step,
                        "action_type": "tool_call",
                        "tool_calls": tool_calls,
                    },
                    reasoning_content=reasoning_content,
                )
                self._record_transcript_message(
                    role="assistant",
                    content=assistant_content,
                    step=step,
                    metadata={
                        "action_type": "tool_call",
                        "tool_calls": tool_calls,
                    },
                )
                host._log_message_write(
                    trace_logger,
                    "assistant",
                    assistant_content,
                    {"action_type": "tool_call", "tool_calls": tool_calls},
                    step,
                )
                if getattr(host, "tool_orchestrator", None) is None:
                    raise RuntimeError("Runtime host must provide a ToolOrchestrator instance")
                # 执行工具调用（优先并行执行，回退到串行）
                if hasattr(host.tool_orchestrator, "run"):
                    observations = host.tool_orchestrator.run(
                        tool_calls,
                        step=step,
                        trace_logger=trace_logger,
                    )
                else:
                    observations = host.tool_orchestrator.run_serial(
                        tool_calls,
                        step=step,
                        trace_logger=trace_logger,
                    )
                for obs in observations:
                    obs_metadata = getattr(obs, "metadata", None) or {}
                    # 以 tool 角色写入观察结果
                    host.history_manager.append_tool(
                        tool_name=obs.tool_name,
                        raw_result=obs.observation,
                        metadata={
                            "step": step,
                            "tool_call_id": obs.tool_call_id,
                            **obs_metadata,
                        },
                        project_root=host.project_root,
                    )
                    self._record_transcript_message(
                        role="tool",
                        content=obs.observation,
                        step=step,
                        metadata={
                            "tool_name": obs.tool_name,
                            "tool_call_id": obs.tool_call_id,
                            **obs_metadata,
                        },
                    )
                    host._log_message_write(
                        trace_logger,
                        "tool",
                        obs.observation,
                        {"tool_name": obs.tool_name, "tool_call_id": obs.tool_call_id},
                        step,
                    )

                    if host.console_verbose:
                        display_obs = (
                            obs.observation[:300] + "..." if len(obs.observation) > 300 else obs.observation
                        )
                        host._console(f"\n👀 Observation: {display_obs}\n")
                    elif host.logger.isEnabledFor(10):
                        display_obs = (
                            obs.observation[:300] + "..." if len(obs.observation) > 300 else obs.observation
                        )
                        host.logger.debug("Observation: %s", display_obs)
                state = self._transition(
                    state,
                    TransitionReason.TOOLS_EXECUTED,
                    trace_logger,
                    step=step,
                    last_tool_calls=tool_calls,
                    details={"tool_count": len(tool_calls)},
                )
                continue  # 工具执行完毕，进入下一轮循环

            # 分支 B：无工具调用，模型返回了最终文本
            # 构建 completion candidate 并通过 gate 校验
            final_text = str(response_text).strip()
            # 三步完成 gate 校验：收集证据 → 推断需求 → 评估判决
            candidate = build_completion_candidate(
                final_text=final_text,
                step=step,
                response_meta=response_meta,
                history_messages=host.history_manager.get_messages(),
            )
            trace_logger.log_event(
                "completion_candidate",
                candidate.to_trace_payload(),
                step=step,
            )

            requirements = infer_completion_requirements(
                user_input=pending_input,
                history_messages=host.history_manager.get_messages(),
            )
            trace_logger.log_event(
                "completion_requirements",
                requirements.to_trace_payload(),
                step=step,
            )

            evidence = collect_verification_evidence(host.history_manager.get_messages())
            for item in evidence:
                trace_logger.log_event("verification_evidence", item.to_trace_payload(), step=step)

            verdict = self._get_completion_verifier().evaluate(
                candidate,
                requirements,
                evidence,
                host.history_manager.get_messages(),
            )
            trace_logger.log_event(
                "completion_gate_verdict",
                verdict.to_trace_payload(),
                step=step,
            )

            if verdict.verdict in {CompletionGateVerdict.PASS, CompletionGateVerdict.UNVERIFIED}:
                # 通过或未验证：接受最终文本，写入历史并返回
                action_type = "final" if verdict.verdict is CompletionGateVerdict.PASS else "final_unverified"
                host.history_manager.append_assistant(
                    content=final_text,
                    metadata={"step": step, "action_type": action_type},
                    reasoning_content=reasoning_content,
                )
                self._record_transcript_message(
                    role="assistant",
                    content=final_text,
                    step=step,
                    metadata={"action_type": action_type},
                )
                host._log_message_write(
                    trace_logger,
                    "assistant",
                    final_text,
                    {"action_type": action_type},
                    step,
                )
                state = self._transition(
                    state,
                    TransitionReason.MODEL_RETURNED_FINAL,
                    trace_logger,
                    step=step,
                    last_response_meta={
                        "final_length": len(final_text),
                        "completion_verdict": verdict.verdict.value,
                    },
                    details={
                        "final_length": len(final_text),
                        "completion_verdict": verdict.verdict.value,
                    },
                )
                terminal_reason = (
                    TerminalReason.COMPLETED
                    if verdict.verdict is CompletionGateVerdict.PASS
                    else TerminalReason.COMPLETED_UNVERIFIED
                )
                self._terminal(
                    terminal_reason,
                    trace_logger,
                    step=step,
                    final_length=len(final_text),
                    completion_verdict=verdict.verdict.value,
                )
                trace_logger.log_event(
                    "finish",
                    {"final": final_text, "completion_verdict": verdict.verdict.value},
                    step=step,
                )
                return final_text

            # Gate 阻塞：将 gate 反馈注入用户消息，让模型修正后重试
            host.history_manager.append_assistant(
                content=final_text,
                metadata={"step": step, "action_type": "final_candidate"},
                reasoning_content=reasoning_content,
            )
            self._record_transcript_message(
                role="assistant",
                content=final_text,
                step=step,
                metadata={"action_type": "final_candidate"},
            )
            host._log_message_write(
                trace_logger,
                "assistant",
                final_text,
                {"action_type": "final_candidate"},
                step,
            )
            block_count = state.completion_block_count + 1
            feedback = verdict.blocking_feedback or "Completion blocked by runtime gate."
            # 将 gate 的阻塞反馈作为用户消息注入
            self._append_user_message(
                feedback,
                metadata={"step": step, "source": "completion_gate"},
            )
            self._record_transcript_message(
                role="user",
                content=feedback,
                step=step,
                metadata={"source": "completion_gate"},
            )
            host._log_message_write(
                trace_logger,
                "user",
                feedback,
                {"source": "completion_gate"},
                step,
            )
            state = self._transition(
                state,
                TransitionReason.STOP_HOOK_BLOCKING,
                trace_logger,
                step=step,
                completion_block_count=block_count,
                stop_hook_active=True,
                details={
                    "completion_verdict": verdict.verdict.value,
                    "reasons": list(verdict.reasons),
                    "retry_count": block_count,
                    "retry_limit": completion_retry_limit,
                },
            )
            if block_count >= completion_retry_limit:
                # Gate 重试次数耗尽，强制终止
                self._terminal(
                    TerminalReason.COMPLETION_GATE_BLOCKED,
                    trace_logger,
                    step=step,
                    completion_verdict=verdict.verdict.value,
                    reasons=list(verdict.reasons),
                    retry_count=block_count,
                    retry_limit=completion_retry_limit,
                )
                return "抱歉，我无法在限定步数内完成这个任务。"
            continue  # Gate 阻塞但未达上限，回到主循环让模型重新生成

        # 达到最大步数限制
        state = self._transition(
            state,
            TransitionReason.MAX_STEPS_EXCEEDED,
            trace_logger,
            step=host.max_steps,
            details={"max_steps": host.max_steps},
        )
        self._terminal(
            TerminalReason.MAX_STEPS,
            trace_logger,
            step=host.max_steps,
            max_steps=host.max_steps,
        )
        return "抱歉，我无法在限定步数内完成这个任务。"
