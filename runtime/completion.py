"""Completion gate and verification evidence for the runtime harness.

Completion Gate（完成门控）：在模型返回最终文本（无 tool_calls）时，
对完成质量进行校验。如果校验不通过，将反馈注入对话让模型修正。

核心流程：
1. build_completion_candidate() — 从模型输出构建候选
2. infer_completion_requirements() — 从用户输入推断验证需求
3. collect_verification_evidence() — 从历史中收集验证证据
4. CompletionVerifier.evaluate() — 判决 PASS / FAIL / UNVERIFIED
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


# "如果可能的话再验证" 的触发词 —— 匹配到则 allow_unverified=True
VERIFY_IF_POSSIBLE_PATTERNS = (
    "if possible",
    "if you can",
    "when possible",
    "如果可以",
    "尽量",
)

# 用户输入中的验证需求模式：(正则, 验证种类)
# 用于从用户消息中推断需要做什么验证
VERIFY_REQUIREMENT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpytest\b", "tests"),
    (r"\brun\s+(the\s+)?tests?\b", "tests"),
    (r"\btest\s+suite\b", "tests"),
    (r"\bunit\s+tests?\b", "tests"),
    (r"(运行|执行|跑)(一下)?测试", "tests"),
    (r"\blint\b", "lint"),
    (r"\btypecheck\b", "typecheck"),
    (r"类型检查", "typecheck"),
    (r"\bbuild\b", "build"),
    (r"(运行|执行)?构建", "build"),
)

# 验证命令分类模式：从实际执行的 Bash 命令中识别验证类型
VERIFY_COMMAND_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpytest\b", "tests"),
    (r"\b(?:npm|pnpm|yarn|bun)\s+(?:run\s+)?test(?::[\w-]+)?\b", "tests"),
    (r"\bgo\s+test\b", "tests"),
    (r"\bcargo\s+test\b", "tests"),
    (r"\bmake\s+test\b", "tests"),
    (r"\blint\b", "lint"),
    (r"\btypecheck\b", "typecheck"),
    (r"\bbuild\b", "build"),
)

# 会产生副作用的工具名 —— 执行这些工具后，之前的验证证据视为失效
MUTATING_TOOL_NAMES = {"Write", "Edit", "MultiEdit"}


class CompletionGateVerdict(str, Enum):
    """完成门控判决结果。"""
    PASS = "pass"            # 通过：验证证据充足，允许返回
    FAIL = "fail"            # 失败：缺少验证或存在未完成任务，需要模型修正
    UNVERIFIED = "unverified"  # 未验证：用户允许跳过，给出弱通过


@dataclass(frozen=True)
class CompletionCandidate:
    """模型返回的完成候选 —— 当模型输出无 tool_calls 时构建。"""
    final_text: str
    step: int
    response_meta: dict[str, Any]
    last_tool_name: str | None = None    # 最后一个工具名（用于判断是否刚完成操作）
    last_tool_status: str | None = None  # 最后一个工具的执行状态

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "final_text": self.final_text,
            "final_length": len(self.final_text),
            "step": self.step,
            "response_meta": dict(self.response_meta),
            "last_tool_name": self.last_tool_name,
            "last_tool_status": self.last_tool_status,
        }


@dataclass(frozen=True)
class CompletionRequirements:
    """从用户输入推断的完成要求。

    决定了 gate 需要检查什么：是否需要验证、验证哪些方面、
    是否允许跳过、是否有未完成的 Todo。
    """
    requires_verification: bool
    verification_kinds: tuple[str, ...] = ()      # 需要验证的种类列表，如 ("tests", "lint")
    allow_unverified: bool = False                 # 用户使用了 "if possible" 等弱约束措辞
    has_incomplete_todos: bool = False
    incomplete_todos: tuple[str, ...] = ()
    explicit_user_constraints: tuple[str, ...] = ()  # 用户明确要求的验证类型

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "requires_verification": self.requires_verification,
            "verification_kinds": list(self.verification_kinds),
            "allow_unverified": self.allow_unverified,
            "has_incomplete_todos": self.has_incomplete_todos,
            "incomplete_todos": list(self.incomplete_todos),
            "explicit_user_constraints": list(self.explicit_user_constraints),
        }


@dataclass(frozen=True)
class VerificationEvidence:
    """单条验证证据 —— 从历史中的 Bash 工具调用提取。

    valid 字段会被 collect_verification_evidence 动态更新：
    如果验证之后又有写操作（Write/Edit），之前的证据会被标记为 invalid。
    """
    requirement_id: str     # 如 "verification:tests"
    tool_name: str          # 产生证据的工具名（通常是 "Bash"）
    command: str | None     # 实际执行的命令
    status: str             # 工具执行状态：success / error
    step: int               # 发生时的步数
    valid: bool             # 证据是否仍然有效
    invalid_reason: str | None = None  # 失效原因，如 "modified_after_verification:5"

    def to_trace_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompletionGateResult:
    """门控判决结果。

    - PASS: 所有验证通过（或无需验证），模型可以返回
    - FAIL: 存在问题，blocking_feedback 会作为用户消息注入让模型修正
    - UNVERIFIED: 用户允许跳过验证，弱通过
    """
    verdict: CompletionGateVerdict
    reasons: tuple[str, ...] = ()
    blocking_feedback: str | None = None          # FAIL 时注入给模型的修正提示
    passed_evidence: tuple[VerificationEvidence, ...] = ()  # 通过验证的证据列表

    def to_trace_payload(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": list(self.reasons),
            "blocking_feedback": self.blocking_feedback,
            "passed_evidence": [e.to_trace_payload() for e in self.passed_evidence],
        }


class CompletionVerifier(Protocol):
    """验证器接口（Protocol，支持依赖注入替换为其他实现）。"""
    def evaluate(
        self,
        candidate: CompletionCandidate,
        requirements: CompletionRequirements,
        evidence: list[VerificationEvidence],
        history_messages: list[Any],
    ) -> CompletionGateResult: ...


class DeterministicCompletionVerifier:
    """Phase 2 默认验证器：纯确定性规则，不调用第二个 Agent。

    判决逻辑：
    1. 有未完成 Todo → FAIL
    2. 需要验证但缺少证据 → FAIL（除非 allow_unverified）
    3. 需要验证且证据全部有效 → 检查是否有 UNVERIFIED 标记
    4. 全部通过 → PASS
    """

    def evaluate(
        self,
        candidate: CompletionCandidate,
        requirements: CompletionRequirements,
        evidence: list[VerificationEvidence],
        history_messages: list[Any],
    ) -> CompletionGateResult:
        reasons: list[str] = []
        passed_evidence: list[VerificationEvidence] = []
        pending_unverified = False

        # 检查未完成 Todo
        if requirements.has_incomplete_todos:
            reasons.append("incomplete_todos")

        # 检查验证证据
        if requirements.requires_verification:
            for kind in requirements.verification_kinds:
                requirement_id = f"verification:{kind}"
                relevant = [item for item in evidence if item.requirement_id == requirement_id]
                valid = [item for item in relevant if item.valid and item.status == "success"]
                if valid:
                    passed_evidence.extend(valid)
                    continue
                if relevant and not valid:
                    # 有相关证据但全部无效（可能被后续写操作作废）
                    reasons.append(f"verification_invalid:{kind}")
                    continue
                if requirements.allow_unverified:
                    # 用户说 "if possible" → 缺少证据也不阻塞
                    pending_unverified = True
                else:
                    reasons.append(f"missing_verification_evidence:{kind}")

        if reasons:
            return CompletionGateResult(
                verdict=CompletionGateVerdict.FAIL,
                reasons=tuple(reasons),
                blocking_feedback=_build_blocking_feedback(requirements, reasons),
                passed_evidence=tuple(passed_evidence),
            )

        if pending_unverified:
            return CompletionGateResult(
                verdict=CompletionGateVerdict.UNVERIFIED,
                reasons=("verification_unverified",),
                passed_evidence=tuple(passed_evidence),
            )

        return CompletionGateResult(
            verdict=CompletionGateVerdict.PASS,
            passed_evidence=tuple(passed_evidence),
        )


def build_completion_candidate(
    *,
    final_text: str,
    step: int,
    response_meta: dict[str, Any] | None,
    history_messages: list[Any],
) -> CompletionCandidate:
    """从模型输出和历史消息构建 CompletionCandidate。

    从历史中反向查找最后一个 tool 角色消息，提取其工具名和执行状态，
    用于判断模型是否在完成工具操作后返回了最终文本。
    """
    last_tool_name = None
    last_tool_status = None
    for message in reversed(history_messages):
        if getattr(message, "role", None) != "tool":
            continue
        metadata = getattr(message, "metadata", {}) or {}
        last_tool_name = metadata.get("tool_name")
        parsed = _parse_tool_payload(getattr(message, "content", ""))
        if isinstance(parsed, dict):
            last_tool_status = parsed.get("status")
        break
    return CompletionCandidate(
        final_text=final_text,
        step=step,
        response_meta=dict(response_meta or {}),
        last_tool_name=last_tool_name,
        last_tool_status=last_tool_status,
    )


def infer_completion_requirements(
    *,
    user_input: str,
    history_messages: list[Any],
) -> CompletionRequirements:
    """从用户输入推断完成要求。

    1. 用 VERIFY_REQUIREMENT_PATTERNS 匹配用户输入，提取验证种类
    2. 检查是否有 "if possible" 等弱约束措辞 → allow_unverified
    3. 从历史中提取最新的 TodoWrite 结果，检查是否有未完成任务
    """
    normalized = (user_input or "").lower()
    explicit_constraints: list[str] = []
    verification_kinds: list[str] = []
    for pattern, kind in VERIFY_REQUIREMENT_PATTERNS:
        if re.search(pattern, normalized):
            if kind not in verification_kinds:
                verification_kinds.append(kind)
                explicit_constraints.append(kind)

    # 从历史中提取最新的 Todo 状态
    latest_todos = _extract_latest_todos(history_messages)
    incomplete_todos = tuple(
        item.get("content", "")
        for item in latest_todos
        if item.get("status") in {"pending", "in_progress"}
    )

    return CompletionRequirements(
        requires_verification=bool(verification_kinds),
        verification_kinds=tuple(verification_kinds),
        allow_unverified=bool(verification_kinds)
        and any(pattern in normalized for pattern in VERIFY_IF_POSSIBLE_PATTERNS),
        has_incomplete_todos=bool(incomplete_todos),
        incomplete_todos=incomplete_todos,
        explicit_user_constraints=tuple(explicit_constraints),
    )


def collect_verification_evidence(history_messages: list[Any]) -> list[VerificationEvidence]:
    """从历史消息中收集验证证据。

    遍历所有 tool 消息：
    1. 跟踪最后一次写操作（Write/Edit）的步数
    2. 对 Bash 调用：用 VERIFY_COMMAND_PATTERNS 分类命令类型
    3. 写操作之后的验证证据标记为 invalid（代码可能已变，旧测试结果不可信）
    """
    evidences: list[VerificationEvidence] = []
    latest_mutation_step = 0

    # 第一遍：收集证据 + 跟踪写操作
    for message in history_messages:
        if getattr(message, "role", None) != "tool":
            continue
        metadata = getattr(message, "metadata", {}) or {}
        tool_name = str(metadata.get("tool_name") or "")
        step = int(metadata.get("step") or 0)
        parsed = _parse_tool_payload(getattr(message, "content", ""))
        if not isinstance(parsed, dict):
            continue
        # 记录最后一次写操作的步数
        if tool_name in MUTATING_TOOL_NAMES and parsed.get("status") in {"success", "partial"}:
            latest_mutation_step = max(latest_mutation_step, step)
        if tool_name != "Bash":
            continue
        params_input = ((parsed.get("context") or {}).get("params_input") or {})
        if not isinstance(params_input, dict):
            continue
        command = params_input.get("command")
        if not isinstance(command, str):
            continue
        evidence_kind = _classify_verification_command(command)
        if evidence_kind is None:
            continue
        evidences.append(
            VerificationEvidence(
                requirement_id=f"verification:{evidence_kind}",
                tool_name=tool_name,
                command=command,
                status=str(parsed.get("status") or "unknown"),
                step=step,
                valid=str(parsed.get("status") or "") == "success",
            )
        )

    # 如果没有写操作，证据全部有效，直接返回
    if latest_mutation_step <= 0:
        return evidences

    # 第二遍：作废写操作之前的验证证据
    invalidated: list[VerificationEvidence] = []
    for evidence in evidences:
        if evidence.step < latest_mutation_step:
            invalidated.append(
                VerificationEvidence(
                    requirement_id=evidence.requirement_id,
                    tool_name=evidence.tool_name,
                    command=evidence.command,
                    status=evidence.status,
                    step=evidence.step,
                    valid=False,
                    invalid_reason=f"modified_after_verification:{latest_mutation_step}",
                )
            )
        else:
            invalidated.append(evidence)
    return invalidated


def _parse_tool_payload(raw: str) -> dict[str, Any] | None:
    """安全解析工具结果的 JSON 字符串。"""
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_latest_todos(history_messages: list[Any]) -> list[dict[str, Any]]:
    """从历史消息中反向查找最新一次的 TodoWrite 结果。"""
    for message in reversed(history_messages):
        if getattr(message, "role", None) != "tool":
            continue
        metadata = getattr(message, "metadata", {}) or {}
        if metadata.get("tool_name") != "TodoWrite":
            continue
        parsed = _parse_tool_payload(getattr(message, "content", ""))
        if not isinstance(parsed, dict):
            continue
        data = parsed.get("data") or {}
        todos = data.get("todos") if isinstance(data, dict) else None
        if isinstance(todos, list):
            return [item for item in todos if isinstance(item, dict)]
    return []


def _classify_verification_command(command: str) -> str | None:
    """将实际执行的 Bash 命令归类为验证类型：tests / lint / typecheck / build。"""
    normalized = command.lower()
    for pattern, kind in VERIFY_COMMAND_PATTERNS:
        if re.search(pattern, normalized):
            return kind
    return None


def _build_blocking_feedback(requirements: CompletionRequirements, reasons: list[str]) -> str:
    """构建 FAIL 时注入给模型的修正提示消息。

    消息以 <system-reminder> 包裹，解释为什么被阻塞以及模型应如何修正。
    """
    lines = ["<system-reminder>Completion blocked by runtime gate.</system-reminder>"]
    for reason in reasons:
        if reason == "incomplete_todos":
            if requirements.incomplete_todos:
                lines.append("Incomplete todos remain: " + "; ".join(requirements.incomplete_todos))
            else:
                lines.append("Incomplete todos remain.")
        elif reason.startswith("missing_verification_evidence:"):
            lines.append(
                f"Missing verification evidence for {reason.split(':', 1)[1]}. Run the required verification tool."
            )
        elif reason.startswith("verification_invalid:"):
            lines.append(
                f"Verification evidence for {reason.split(':', 1)[1]} is missing, failed, or stale."
            )
        else:
            lines.append(reason)
    return "\n".join(lines)


__all__ = [
    "CompletionCandidate",
    "CompletionGateResult",
    "CompletionGateVerdict",
    "CompletionRequirements",
    "CompletionVerifier",
    "DeterministicCompletionVerifier",
    "VerificationEvidence",
    "build_completion_candidate",
    "collect_verification_evidence",
    "infer_completion_requirements",
]
