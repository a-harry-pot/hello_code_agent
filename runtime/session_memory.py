"""Session memory derived deterministically from transcript events.

会话记忆：从 transcript 事件中确定性推导出的轻量工作记忆摘要。

设计目的：在长对话中，模型会因为上下文窗口限制而"遗忘"早期发生的事情。
SessionMemory 将对话进展压缩为结构化摘要（当前目标、已完成工作、关键决策等），
在每次构建模型视图时注入为 system 消息，帮助模型保持上下文感知。

与 transcript 的关系：
- transcript 是完整的事件日志（硬盘持久化，可追加重放）
- session_memory 是从 transcript 推导的精简摘要（注入模型上下文）
- 两者一一对应：每条 SessionMemoryItem 都记录了来源事件范围（TranscriptEventRange）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from runtime.transcript import TranscriptEvent


SESSION_MEMORY_SCHEMA_VERSION = 1


# ── 辅助数据结构 ──────────────────────────────────────────────

@dataclass(frozen=True)
class TranscriptEventRange:
    """标记一条记忆项来源于 transcript 中的哪个事件区间。"""
    start_event_id: str | None = None
    end_event_id: str | None = None
    start_step: int = 0
    end_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_event_id": self.start_event_id,
            "end_event_id": self.end_event_id,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TranscriptEventRange":
        payload = data or {}
        return cls(
            start_event_id=payload.get("start_event_id"),
            end_event_id=payload.get("end_event_id"),
            start_step=int(payload.get("start_step") or 0),
            end_step=int(payload.get("end_step") or 0),
        )


@dataclass(frozen=True)
class SessionMemoryItem:
    """单条记忆项：文本描述 + 可追溯的来源事件范围。"""
    text: str
    source: TranscriptEventRange

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source": self.source.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMemoryItem":
        return cls(
            text=str(data.get("text") or ""),
            source=TranscriptEventRange.from_dict(data.get("source")),
        )


@dataclass(frozen=True)
class SessionMemory:
    """会话记忆的完整快照，包含六个维度的记忆分类。

    - current_goal: 当前目标（从最近一次非 gate 反馈的用户消息提取）
    - completed_work: 已完成的工作（模型 final 回复、工具执行成功）
    - key_decisions: 关键决策（checkpoint 创建、压缩触发等）
    - failed_attempts: 失败尝试（模型恢复失败、工具执行失败）
    - todo_items: 待处理项（started 但未完成的工具调用）
    - verification_status: 验证状态（gate 阻塞、terminal 原因等）
    """
    schema_version: int = SESSION_MEMORY_SCHEMA_VERSION
    current_goal: SessionMemoryItem | None = None
    completed_work: tuple[SessionMemoryItem, ...] = ()
    key_decisions: tuple[SessionMemoryItem, ...] = ()
    failed_attempts: tuple[SessionMemoryItem, ...] = ()
    todo_items: tuple[SessionMemoryItem, ...] = ()
    verification_status: tuple[SessionMemoryItem, ...] = ()
    source: TranscriptEventRange = field(default_factory=TranscriptEventRange)
    version: int = 1           # 单调递增版本号，每次 update 自增
    event_count: int = 0       # 已处理的事件总数
    last_event_id: str | None = None
    runtime_state: dict[str, Any] = field(default_factory=dict)  # 内部状态（tool_states 等）

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "current_goal": self.current_goal.to_dict() if self.current_goal else None,
            "completed_work": [item.to_dict() for item in self.completed_work],
            "key_decisions": [item.to_dict() for item in self.key_decisions],
            "failed_attempts": [item.to_dict() for item in self.failed_attempts],
            "todo_items": [item.to_dict() for item in self.todo_items],
            "verification_status": [item.to_dict() for item in self.verification_status],
            "source": self.source.to_dict(),
            "version": self.version,
            "event_count": self.event_count,
            "last_event_id": self.last_event_id,
            "runtime_state": self.runtime_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMemory":
        return cls(
            schema_version=int(data.get("schema_version") or SESSION_MEMORY_SCHEMA_VERSION),
            current_goal=(
                SessionMemoryItem.from_dict(data["current_goal"])
                if data.get("current_goal")
                else None
            ),
            completed_work=tuple(SessionMemoryItem.from_dict(item) for item in data.get("completed_work") or []),
            key_decisions=tuple(SessionMemoryItem.from_dict(item) for item in data.get("key_decisions") or []),
            failed_attempts=tuple(SessionMemoryItem.from_dict(item) for item in data.get("failed_attempts") or []),
            todo_items=tuple(SessionMemoryItem.from_dict(item) for item in data.get("todo_items") or []),
            verification_status=tuple(
                SessionMemoryItem.from_dict(item) for item in data.get("verification_status") or []
            ),
            source=TranscriptEventRange.from_dict(data.get("source")),
            version=int(data.get("version") or 1),
            event_count=int(data.get("event_count") or 0),
            last_event_id=data.get("last_event_id"),
            runtime_state=dict(data.get("runtime_state") or {}),
        )


class SessionMemoryDeriver:
    """从 transcript 事件推导有界工作记忆。

    两种工作模式：
    - rebuild(): 全量重建 —— 遍历所有事件从头构建
    - update(): 增量更新 —— 在已有 memory 基础上追加新事件（性能更好）
    """

    def rebuild(self, events: Iterable["TranscriptEvent"]) -> SessionMemory:
        """全量重建：遍历所有 transcript 事件，从零构建 SessionMemory。"""
        ordered = list(events)
        if not ordered:
            return SessionMemory()

        current_goal: SessionMemoryItem | None = None
        completed_work: list[SessionMemoryItem] = []
        key_decisions: list[SessionMemoryItem] = []
        failed_attempts: list[SessionMemoryItem] = []
        verification_status: list[SessionMemoryItem] = []
        tool_states: dict[str, dict[str, Any]] = {}

        for event in ordered:
            payload = dict(event.payload or {})
            source = TranscriptEventRange(
                start_event_id=event.event_id,
                end_event_id=event.event_id,
                start_step=event.step,
                end_step=event.step,
            )
            event_type = str(getattr(event.event_type, "value", event.event_type))

            if event_type == "message":
                role = str(payload.get("role") or "")
                metadata = dict(payload.get("metadata") or {})
                # 用户消息（排除 completion gate 反馈）→ 当前目标
                if role == "user" and metadata.get("source") != "completion_gate":
                    current_goal = SessionMemoryItem(text=str(payload.get("content") or "").strip(), source=source)
                elif role == "assistant":
                    action_type = str(metadata.get("action_type") or "")
                    content = str(payload.get("content") or "").strip()
                    # 模型的最终回复 → 已完成工作
                    if action_type in {"final", "final_unverified"} and content:
                        completed_work.append(
                            SessionMemoryItem(
                                text=f"Assistant produced final response: {content}",
                                source=source,
                            )
                        )
            elif event_type == "state_transition":
                reason = str(payload.get("reason") or "")
                details = dict(payload.get("details") or {})
                # 上下文压缩 → 关键决策
                if reason == "context_compacted":
                    checkpoint_id = details.get("checkpoint_id") or "unknown"
                    key_decisions.append(
                        SessionMemoryItem(text=f"Created compact checkpoint {checkpoint_id}", source=source)
                    )
                # Gate 阻塞 → 验证状态
                elif reason == "stop_hook_blocking":
                    verification_status.append(
                        SessionMemoryItem(
                            text="Completion gate blocked finalization pending verification.",
                            source=source,
                        )
                    )
                # 模型恢复失败 → 失败尝试
                elif reason == "model_recovery_failed":
                    failed_attempts.append(
                        SessionMemoryItem(
                            text="Model recovery failed and required terminal fallback.",
                            source=source,
                        )
                    )
            elif event_type == "checkpoint":
                checkpoint_id = str(payload.get("checkpoint_id") or "unknown")
                key_decisions.append(
                    SessionMemoryItem(text=f"Recorded checkpoint {checkpoint_id}", source=source)
                )
            elif event_type == "terminal":
                reason = str(payload.get("reason") or "")
                if reason:
                    verification_status.append(
                        SessionMemoryItem(text=f"Run ended with terminal reason {reason}.", source=source)
                    )
            elif event_type == "tool_lifecycle":
                # 跟踪每个工具调用的完整生命周期
                tool_call_id = str(payload.get("tool_call_id") or "")
                if not tool_call_id:
                    continue
                tool_name = str(payload.get("tool_name") or "unknown")
                status = str(payload.get("status") or "")
                current = tool_states.setdefault(
                    tool_call_id,
                    {
                        "tool_name": tool_name,
                        "requested": None,
                        "started": None,
                        "completed": None,
                        "failed": None,
                    },
                )
                current["tool_name"] = tool_name
                current[status] = {
                    "event_id": event.event_id,
                    "step": event.step,
                }
                # 工具完成 → 已完成工作；工具失败 → 失败尝试
                if status == "completed":
                    completed_work.append(
                        SessionMemoryItem(text=f"Completed tool {tool_name} ({tool_call_id}).", source=source)
                    )
                elif status == "failed":
                    failed_attempts.append(
                        SessionMemoryItem(text=f"Failed tool {tool_name} ({tool_call_id}).", source=source)
                    )

        # 最后一轮：找出所有未解决的工具调用（started 但未 completed/failed）
        todo_items, unresolved_verification = self._build_unresolved_items(tool_states)
        first = ordered[0]
        last = ordered[-1]
        return SessionMemory(
            current_goal=current_goal,
            completed_work=tuple(completed_work),
            key_decisions=tuple(key_decisions),
            failed_attempts=tuple(failed_attempts),
            todo_items=tuple(todo_items),
            verification_status=tuple([*verification_status, *unresolved_verification]),
            source=TranscriptEventRange(
                start_event_id=first.event_id,
                end_event_id=last.event_id,
                start_step=first.step,
                end_step=last.step,
            ),
            version=len(ordered),
            event_count=len(ordered),
            last_event_id=last.event_id,
            runtime_state={"tool_states": tool_states},
        )

    def update(
        self,
        previous: SessionMemory | None,
        events: Iterable["TranscriptEvent"],
        *,
        summary_refiner: Callable[[SessionMemory, SessionMemory | None, list["TranscriptEvent"]], SessionMemory] | None = None,
    ) -> SessionMemory:
        """增量更新：在已有 memory 基础上追加新事件。

        - 如果 previous 为 None，回退到全量 rebuild
        - 如果提供了 summary_refiner，在增量合并后调用它做进一步精炼（如 LLM 摘要）
        - version 自动递增
        """
        event_list = list(events)
        if not event_list:
            return previous or SessionMemory()

        if previous is None:
            draft = self.rebuild(event_list)
        else:
            draft = self._apply_incremental(previous, event_list)

        draft = SessionMemory(
            schema_version=draft.schema_version,
            current_goal=draft.current_goal,
            completed_work=draft.completed_work,
            key_decisions=draft.key_decisions,
            failed_attempts=draft.failed_attempts,
            todo_items=draft.todo_items,
            verification_status=draft.verification_status,
            source=draft.source,
            version=(previous.version + 1) if previous is not None else draft.version,
            event_count=draft.event_count,
            last_event_id=draft.last_event_id,
            runtime_state=draft.runtime_state,
        )
        if summary_refiner is None:
            return draft
        # 可选的 LLM 精炼步骤
        try:
            return summary_refiner(draft, previous, event_list)
        except Exception:
            return previous if previous is not None else draft

    def _apply_incremental(self, previous: SessionMemory, new_events: list["TranscriptEvent"]) -> SessionMemory:
        """增量合并：在 previous 的基础上只处理 new_events。

        与 rebuild() 的核心区别：
        - current_goal 从 previous 继承，仅在遇到新用户消息时更新
        - completed_work / key_decisions / failed_attempts 追加到已有列表
        - verification_status 中过滤掉旧的 "uncertain tool action" 条目（新事件可能已解决）
        - tool_states 从 previous.runtime_state 恢复后继续追踪
        """
        current_goal = previous.current_goal
        completed_work = list(previous.completed_work)
        key_decisions = list(previous.key_decisions)
        failed_attempts = list(previous.failed_attempts)
        verification_status = [
            item for item in previous.verification_status if "uncertain tool action" not in item.text.lower()
        ]
        tool_states = dict(previous.runtime_state.get("tool_states") or {})

        for event in new_events:
            payload = dict(event.payload or {})
            source = TranscriptEventRange(
                start_event_id=event.event_id,
                end_event_id=event.event_id,
                start_step=event.step,
                end_step=event.step,
            )
            event_type = str(getattr(event.event_type, "value", event.event_type))

            if event_type == "message":
                role = str(payload.get("role") or "")
                metadata = dict(payload.get("metadata") or {})
                if role == "user" and metadata.get("source") != "completion_gate":
                    current_goal = SessionMemoryItem(text=str(payload.get("content") or "").strip(), source=source)
                elif role == "assistant":
                    action_type = str(metadata.get("action_type") or "")
                    content = str(payload.get("content") or "").strip()
                    if action_type in {"final", "final_unverified"} and content:
                        completed_work.append(
                            SessionMemoryItem(
                                text=f"Assistant produced final response: {content}",
                                source=source,
                            )
                        )
            elif event_type == "state_transition":
                reason = str(payload.get("reason") or "")
                details = dict(payload.get("details") or {})
                if reason == "context_compacted":
                    checkpoint_id = details.get("checkpoint_id") or "unknown"
                    key_decisions.append(
                        SessionMemoryItem(text=f"Created compact checkpoint {checkpoint_id}", source=source)
                    )
                elif reason == "stop_hook_blocking":
                    verification_status.append(
                        SessionMemoryItem(
                            text="Completion gate blocked finalization pending verification.",
                            source=source,
                        )
                    )
                elif reason == "model_recovery_failed":
                    failed_attempts.append(
                        SessionMemoryItem(
                            text="Model recovery failed and required terminal fallback.",
                            source=source,
                        )
                    )
            elif event_type == "checkpoint":
                checkpoint_id = str(payload.get("checkpoint_id") or "unknown")
                key_decisions.append(
                    SessionMemoryItem(text=f"Recorded checkpoint {checkpoint_id}", source=source)
                )
            elif event_type == "terminal":
                reason = str(payload.get("reason") or "")
                if reason:
                    verification_status.append(
                        SessionMemoryItem(text=f"Run ended with terminal reason {reason}.", source=source)
                    )
            elif event_type == "tool_lifecycle":
                tool_call_id = str(payload.get("tool_call_id") or "")
                if not tool_call_id:
                    continue
                tool_name = str(payload.get("tool_name") or "unknown")
                status = str(payload.get("status") or "")
                current = tool_states.setdefault(
                    tool_call_id,
                    {
                        "tool_name": tool_name,
                        "requested": None,
                        "started": None,
                        "completed": None,
                        "failed": None,
                    },
                )
                current["tool_name"] = tool_name
                current[status] = {"event_id": event.event_id, "step": event.step}
                if status == "completed":
                    completed_work.append(
                        SessionMemoryItem(text=f"Completed tool {tool_name} ({tool_call_id}).", source=source)
                    )
                elif status == "failed":
                    failed_attempts.append(
                        SessionMemoryItem(text=f"Failed tool {tool_name} ({tool_call_id}).", source=source)
                    )

        todo_items, unresolved_verification = self._build_unresolved_items(tool_states)
        last = new_events[-1]
        return SessionMemory(
            current_goal=current_goal,
            completed_work=tuple(completed_work),
            key_decisions=tuple(key_decisions),
            failed_attempts=tuple(failed_attempts),
            todo_items=tuple(todo_items),
            verification_status=tuple([*verification_status, *unresolved_verification]),
            source=TranscriptEventRange(
                start_event_id=previous.source.start_event_id or new_events[0].event_id,
                end_event_id=last.event_id,
                start_step=previous.source.start_step if previous.event_count else new_events[0].step,
                end_step=last.step,
            ),
            event_count=previous.event_count + len(new_events),
            last_event_id=last.event_id,
            runtime_state={"tool_states": tool_states},
        )

    def _build_unresolved_items(
        self,
        tool_states: dict[str, dict[str, Any]],
    ) -> tuple[list[SessionMemoryItem], list[SessionMemoryItem]]:
        """从 tool_states 中找出所有未解决的工具调用。

        两种情况视为"未解决"：
        1. started 但既未 completed 也未 failed → 不确定操作（中断恢复场景）
        2. requested 但尚未 started → 待重规划的调用
        """
        todo_items: list[SessionMemoryItem] = []
        verification_status: list[SessionMemoryItem] = []
        for tool_call_id, tool_state in sorted(tool_states.items()):
            tool_name = str(tool_state.get("tool_name") or "unknown")
            started = tool_state.get("started")
            completed = tool_state.get("completed")
            failed = tool_state.get("failed")
            requested = tool_state.get("requested")
            # 已启动但未完成/未失败 → 不确定状态
            if started is not None and completed is None and failed is None:
                source = TranscriptEventRange(
                    start_event_id=(requested or started).get("event_id"),
                    end_event_id=started.get("event_id"),
                    start_step=int((requested or started).get("step") or 0),
                    end_step=int(started.get("step") or 0),
                )
                todo_items.append(
                    SessionMemoryItem(
                        text=f"Resolve uncertain action for {tool_name} ({tool_call_id}) before claiming completion.",
                        source=source,
                    )
                )
                verification_status.append(
                    SessionMemoryItem(
                        text=f"Uncertain tool action detected for {tool_name} ({tool_call_id}).",
                        source=source,
                    )
                )
            # 仅请求但未启动 → 待规划
            elif requested is not None and started is None and completed is None and failed is None:
                source = TranscriptEventRange(
                    start_event_id=requested.get("event_id"),
                    end_event_id=requested.get("event_id"),
                    start_step=int(requested.get("step") or 0),
                    end_step=int(requested.get("step") or 0),
                )
                todo_items.append(
                    SessionMemoryItem(
                        text=f"Replan pending tool call {tool_name} ({tool_call_id}).",
                        source=source,
                    )
                )
        return todo_items, verification_status


def render_session_memory(memory: SessionMemory, *, char_budget: int) -> tuple[str, int]:
    """将 SessionMemory 渲染为模型可读的 markdown 文本。

    在 ContextEngine.build_model_view() 中被调用，作为 system 消息注入模型上下文。
    受 char_budget 限制（默认 4000），超出则截断。

    Returns:
        (渲染后的文本, 实际字符数)
    """
    # 空记忆不渲染，节省上下文空间
    if memory.current_goal is None and not any(
        (
            memory.completed_work,
            memory.key_decisions,
            memory.failed_attempts,
            memory.todo_items,
            memory.verification_status,
        )
    ):
        return "", 0

    sections = [
        "## Session Memory",
        f"Source: transcript events {memory.source.start_event_id or 'unknown'}..{memory.source.end_event_id or 'unknown'}",
    ]
    if memory.current_goal is not None:
        sections.append(f"Current Goal: {memory.current_goal.text}")

    def _append_group(title: str, items: tuple[SessionMemoryItem, ...]) -> None:
        if not items:
            return
        sections.append(f"{title}:")
        for item in items:
            sections.append(f"- {item.text}")

    # 按重要性排列：Todo → Verification → Key Decisions → Failed → Completed
    _append_group("Todo", memory.todo_items)
    _append_group("Verification", memory.verification_status)
    _append_group("Key Decisions", memory.key_decisions)
    _append_group("Failed Attempts", memory.failed_attempts)
    _append_group("Completed Work", memory.completed_work)

    text = "\n".join(sections)
    if len(text) <= char_budget:
        return text, len(text)
    # 超出预算时截断，末尾加标记
    truncated = text[: max(0, char_budget - len("\n[truncated]"))].rstrip() + "\n[truncated]"
    return truncated, len(truncated)


__all__ = [
    "SESSION_MEMORY_SCHEMA_VERSION",
    "SessionMemory",
    "SessionMemoryDeriver",
    "SessionMemoryManager",
    "SessionMemoryItem",
    "TranscriptEventRange",
    "render_session_memory",
]


class SessionMemoryManager:
    """持有当前会话记忆并随 transcript 事件同步更新。

    使用方式：
    - manager.ingest_event(event) → 逐事件增量更新
    - manager.rebuild(events) → 全量重建（如从 transcript 恢复时）
    - manager.memory → 访问当前快照
    """

    def __init__(
        self,
        *,
        deriver: SessionMemoryDeriver | None = None,
        summary_refiner: Callable[[SessionMemory, SessionMemory | None, list["TranscriptEvent"]], SessionMemory] | None = None,
        on_update: Callable[[SessionMemory], None] | None = None,
    ):
        self.deriver = deriver or SessionMemoryDeriver()
        self.summary_refiner = summary_refiner
        self.on_update = on_update
        self.memory = SessionMemory()

    def ingest_event(self, event: "TranscriptEvent") -> SessionMemory:
        """逐事件增量更新：将单个 transcript 事件合并到当前记忆。"""
        previous = self.memory if self.memory.event_count > 0 else None
        self.memory = self.deriver.update(
            previous,
            [event],
            summary_refiner=self.summary_refiner,
        )
        if self.on_update is not None:
            self.on_update(self.memory)
        return self.memory

    def rebuild(self, events: Iterable["TranscriptEvent"]) -> SessionMemory:
        """全量重建：从事件列表重新构建记忆（如从 transcript 恢复会话时）。"""
        self.memory = self.deriver.rebuild(events)
        if self.on_update is not None:
            self.on_update(self.memory)
        return self.memory
