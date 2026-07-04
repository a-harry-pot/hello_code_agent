"""会话日志持久化 — 结构化 JSONL + 人类可读文本双通道输出"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TextIO


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class SessionLogger:
    """每个 Agent 会话一个实例，管理 JSONL / 文本日志 / 控制台三路输出。

    使用方式::

        logger = SessionLogger(
            output_dir=".helloagents/logs",
            agent_name="my_agent",
            agent_type="react",
            model="deepseek-chat",
            provider="deepseek",
        )
        logger.console("Starting…")            # 控制台 + 文本日志
        logger.event("user_input", {...})       # JSONL + 文本日志摘要
        logger.close()
    """

    def __init__(
        self,
        output_dir: str | Path,
        agent_name: str,
        agent_type: str,
        model: str,
        provider: str,
        session_id: str | None = None,
        extra_meta: dict | None = None,
    ):
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.model = model
        self.provider = provider

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base = f"{ts}_{agent_type}_{self.session_id}"

        self.jsonl_path = self.output_dir / f"{base}.jsonl"
        self.text_path = self.output_dir / f"{base}.log"

        self._jsonl_file: TextIO = open(self.jsonl_path, "w", encoding="utf-8")
        self._text_file: TextIO = open(self.text_path, "w", encoding="utf-8")
        self._lock = threading.Lock()

        self._start_time = time.time()
        self._stats: dict[str, int] = {
            "total_steps": 0, "total_llm_calls": 0,
            "total_tool_calls": 0, "errors": 0,
        }

        self.event("session_start", {
            "agent_type": agent_type, "agent_name": agent_name,
            "model": model, "provider": provider,
            **(extra_meta or {}),
        })

    # ---- Public API ----

    def console(self, *args, sep: str = " ") -> None:
        """写入控制台（保留 ANSI）和文本日志（去除 ANSI）。

        接受多参数，行为与 print() 一致。
        """
        msg = sep.join(str(a) for a in args)
        print(*args, sep=sep)
        clean = _ANSI_RE.sub("", msg)
        self._text_file.write(clean + "\n")
        self._text_file.flush()

    def event(self, event_type: str, data: dict | None = None) -> None:
        """写入一条结构化事件到 JSONL，同时写一行摘要到文本日志。"""
        record = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "agent_type": self.agent_type,
            "agent_name": self.agent_name,
            "model": self.model,
            "provider": self.provider,
            "data": data or {},
        }
        with self._lock:
            self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._jsonl_file.flush()

            summary = _summarize(event_type, data or {})
            if summary:
                self._text_file.write(f"[{event_type}] {summary}\n")
                self._text_file.flush()

    def close(self, final_answer: str = "", success: bool = True, error: str | None = None) -> None:
        """写入 session_end 事件并关闭所有文件句柄。"""
        if self._jsonl_file.closed:
            return
        duration_ms = int((time.time() - self._start_time) * 1000)
        if final_answer:
            final_answer = final_answer[:500]
        self.event("session_end", {
            "total_duration_ms": duration_ms,
            "success": success,
            "error": error,
            "final_answer_preview": final_answer,
            **self._stats,
        })
        self._jsonl_file.close()
        self._text_file.close()

    # ---- Stats counters ----

    def inc_step(self) -> None:
        self._stats["total_steps"] += 1

    def inc_llm_call(self) -> None:
        self._stats["total_llm_calls"] += 1

    def inc_tool_call(self) -> None:
        self._stats["total_tool_calls"] += 1

    def inc_error(self) -> None:
        self._stats["errors"] += 1


class _NoopLogger:
    """哨兵：当未传入 SessionLogger 时，所有方法为空操作，Agent 代码无需 null 检查。"""

    def console(self, *args, sep: str = " ") -> None:
        print(*args, sep=sep)

    def event(self, event_type: str, data: dict | None = None) -> None:
        pass

    def close(self, final_answer: str = "", success: bool = True, error: str | None = None) -> None:
        pass

    def inc_step(self) -> None:
        pass

    def inc_llm_call(self) -> None:
        pass

    def inc_tool_call(self) -> None:
        pass

    def inc_error(self) -> None:
        pass


# ---- Internal helpers ----

def _summarize(event_type: str, data: dict) -> str:
    """为文本日志生成一行摘要。"""
    if event_type == "user_input":
        c = data.get("content", "")
        return f"len={data.get('content_length', len(c))} preview={c[:80]}"
    elif event_type == "llm_call":
        return (
            f"type={data.get('call_type','?')} "
            f"msgs={data.get('message_count','?')} "
            f"latency_ms={data.get('latency_ms','?')} "
            f"resp_len={data.get('response_length','?')} "
            f"stream={data.get('streaming', False)}"
        )
    elif event_type == "tool_call":
        return (
            f"tool={data.get('tool_name','?')} "
            f"ok={data.get('success','?')} "
            f"latency_ms={data.get('latency_ms','?')} "
            f"out_len={data.get('output_length','?')}"
        )
    elif event_type == "agent_thought":
        t = data.get("thought", "")
        return f"step={data.get('step','?')} thought={t[:100]}"
    elif event_type == "agent_action":
        return f"step={data.get('step','?')} action={data.get('action','')[:120]}"
    elif event_type == "agent_answer":
        a = data.get("answer", "")
        return f"steps={data.get('total_steps','?')} len={len(a)} preview={a[:100]}"
    elif event_type == "agent_observation":
        return f"step={data.get('step','?')} len={data.get('observation_length', 0)}"
    elif event_type == "plan_generated":
        return f"plan_len={data.get('plan_length', 0)}"
    elif event_type == "step_executed":
        return f"step={data.get('step_index','?')}/{data.get('step_total','?')}"
    elif event_type == "reflection":
        return f"iter={data.get('iteration','?')} phase={data.get('phase','?')}"
    elif event_type == "error":
        return f"ctx={data.get('context','?')} msg={data.get('error_message','')[:120]}"
    elif event_type in ("session_start", "session_end"):
        return ""
    return ""
