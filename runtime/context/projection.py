"""Read-time history projection."""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.context.compact_store import CompactStore
from runtime.history import Message


@dataclass(frozen=True)
class ProjectionResult:
    messages: list[Message]
    source_message_count: int
    projection_mode: str = "full_history"
    warnings: tuple[str, ...] = field(default_factory=tuple)
    compact_checkpoint_id: str | None = None


class ProjectionBuilder:
    """Builds the active history view without mutating the runtime log."""

    def __init__(self, compact_store: CompactStore | None = None):
        self.compact_store = compact_store

    def project(self, source_messages: list[Message]) -> ProjectionResult:
        """将原始消息列表和当前检查点组合成实际发给模型的历史视图：若存在检查点，则返回 [摘要消息] +
  原始消息[保留起始索引:]；否则返回完整历史。"""
        source = list(source_messages or [])
        checkpoint = self.compact_store.active_checkpoint if self.compact_store else None
        if not checkpoint:
            return ProjectionResult(
                messages=source,
                source_message_count=len(source),
                projection_mode="full_history",
                warnings=(),
            )

        retain_start_idx = min(max(checkpoint.retain_start_idx, 0), len(source))
        summary = Message(
            content=checkpoint.summary,
            role="summary",
            metadata={
                "checkpoint_id": checkpoint.id,
                "source_message_count": checkpoint.source_message_count,
                "messages_compacted": checkpoint.messages_compacted,
            },
        )
        return ProjectionResult(
            messages=[summary] + source[retain_start_idx:],
            source_message_count=len(source),
            projection_mode="compact_checkpoint",
            warnings=(),
            compact_checkpoint_id=checkpoint.id,
        )
