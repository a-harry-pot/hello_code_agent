"""Trace 数据模型与持久化 — Agent 思考链路追踪系统"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# 数据模型
# ============================================================

@dataclass
class LLMCallRecord:
    """单次 LLM 调用的元数据"""
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    model: str


@dataclass
class StepRecord:
    """ReAct 单步的完整记录"""
    step: int
    thought: Optional[str]
    action: Optional[str]
    tool_name: Optional[str]
    tool_input: Optional[str]
    tool_output_length: Optional[int]
    observation_summary: Optional[str]
    tool_latency_ms: float
    llm_call: Optional[LLMCallRecord]
    timestamp: str


@dataclass
class TurnRecord:
    """一轮对话（用户提问→Agent 回答）的完整记录"""
    turn_index: int
    user_input: str
    start_time: str
    end_time: Optional[str] = None
    steps: List[StepRecord] = field(default_factory=list)
    final_answer: Optional[str] = None
    error: Optional[str] = None


@dataclass
class SessionTrace:
    """整个会话的追踪数据"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    turns: List[TurnRecord] = field(default_factory=list)


# ============================================================
# TraceStore — 管理追踪数据的收集与持久化
# ============================================================

class TraceStore:
    """追踪存储器

    在 Agent 运行过程中收集 Step/Turn 数据，
    会话结束时写入结构化 JSON 文件。
    """

    def __init__(self, traces_dir: Path, session_id: str, config_snapshot: Optional[Dict[str, Any]] = None):
        self._traces_dir = Path(traces_dir)
        self._traces_dir.mkdir(parents=True, exist_ok=True)

        self._session = SessionTrace(
            session_id=session_id,
            start_time=datetime.now().isoformat(),
            config_snapshot=config_snapshot or {},
        )
        self._current_turn: Optional[TurnRecord] = None
        self._turn_counter = 0

    # ---- session lifecycle ----

    @property
    def session(self) -> SessionTrace:
        return self._session

    def finalize(self) -> None:
        """结束会话并写入文件"""
        self._session.end_time = datetime.now().isoformat()
        self._write()

    # ---- turn lifecycle ----

    def start_turn(self, user_input: str) -> int:
        """开始新的一轮对话，返回 turn_index"""
        self._turn_counter += 1
        self._current_turn = TurnRecord(
            turn_index=self._turn_counter,
            user_input=user_input,
            start_time=datetime.now().isoformat(),
        )
        return self._turn_counter

    def end_turn(self, final_answer: Optional[str] = None, error: Optional[str] = None) -> None:
        """结束当前轮"""
        if self._current_turn is None:
            return
        self._current_turn.end_time = datetime.now().isoformat()
        self._current_turn.final_answer = final_answer
        self._current_turn.error = error
        self._session.turns.append(self._current_turn)
        self._current_turn = None
        # 每轮结束后增量写入
        self._write()

    # ---- step ----

    def add_step(self, record: StepRecord) -> None:
        """添加一个 ReAct 步骤到当前轮"""
        if self._current_turn is None:
            return
        self._current_turn.steps.append(record)

    # ---- persistence ----

    def _write(self) -> None:
        """将当前 session trace 写入 JSON 文件"""
        filepath = self._traces_dir / f"{self._session.session_id}.json"
        data = _session_trace_to_dict(self._session)
        filepath.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def flush(self) -> None:
        """强制写入（别名）"""
        self._write()


def _session_trace_to_dict(session: SessionTrace) -> Dict[str, Any]:
    """将 SessionTrace 转换为可序列化的字典"""
    return {
        "session_id": session.session_id,
        "start_time": session.start_time,
        "end_time": session.end_time,
        "config_snapshot": session.config_snapshot,
        "turns": [
            {
                "turn_index": t.turn_index,
                "user_input": t.user_input,
                "start_time": t.start_time,
                "end_time": t.end_time,
                "final_answer": t.final_answer,
                "error": t.error,
                "steps": [
                    {
                        "step": s.step,
                        "thought": s.thought,
                        "action": s.action,
                        "tool_name": s.tool_name,
                        "tool_input": s.tool_input,
                        "tool_output_length": s.tool_output_length,
                        "observation_summary": s.observation_summary,
                        "tool_latency_ms": s.tool_latency_ms,
                        "llm_call": {
                            "latency_ms": s.llm_call.latency_ms,
                            "prompt_tokens": s.llm_call.prompt_tokens,
                            "completion_tokens": s.llm_call.completion_tokens,
                            "model": s.llm_call.model,
                        }
                        if s.llm_call
                        else None,
                        "timestamp": s.timestamp,
                    }
                    for s in t.steps
                ],
            }
            for t in session.turns
        ],
    }
