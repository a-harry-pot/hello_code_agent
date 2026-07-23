from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from agents.react_agent import ReActAgent
from core.config import Config
from core.llm import HelloAgentsLLM
from core.message import Message
from core.session_logger import SessionLogger
from context.builder import ContextBuilder, ContextConfig, ContextPacket
from tools.builtin.bash import BashTool
from tools.builtin.edit_file import EditTool
from tools.builtin.edit_file_multi import MultiEditTool
from tools.builtin.list_files import ListFilesTool
from tools.builtin.read_file import ReadTool
from tools.builtin.search_code import GrepTool
from tools.builtin.search_files_by_name import SearchFilesByNameTool
from tools.builtin.todo_write import TodoWriteTool
from tools.builtin.write_file import WriteTool
from tools.registry import ToolRegistry
from tools.builtin.note_tool import NoteTool
from tools.builtin.terminal_tool import TerminalTool
from tools.builtin.plan_tool import PlanTool
from tools.builtin.todo_tool import TodoTool
from tools.builtin.context_fetch_tool import ContextFetchTool



@dataclass
class CodeAgentPaths:
    """CodeAgent 路径配置类，集中管理所有相关目录路径"""
    repo_root: Path
    notes_dir: Path
    memory_dir: Path
    sessions_dir: Path
    logs_dir: Path

    @property
    def helloagents_dir(self) -> Path:
        """返回 .helloagents 目录路径"""
        return self.repo_root / ".helloagents"

    @property
    def prompts_dir(self) -> Path:
        """返回 prompts 目录路径"""
        return self.repo_root / "code_agent" / "prompts"

class CodeAgent:
    """
    类似 Claude Code/Codex 的 CLI 智能体：
    - 核心循环使用 ReActAgent。
    - ContextBuilder 负责拼接：系统提示词 + 最近对话 + 相关笔记 + 情景记忆。
    - 规划能力作为可选工具 (`plan`) 暴露给模型，模型可按需调用。
    """

    def __init__(self, repo_root: Path, llm: Optional[HelloAgentsLLM] = None, config: Optional[Config] = None):
        """
        初始化 CodeAgent

        Args:
            repo_root: 代码仓库根目录
            llm: LLM 实例
            config: 配置对象
        """
        repo_root=repo_root.resolve()
        self.config=config or Config.from_env()

        #初始化目录结构
        helloagents_dir=Path(self.config.helloagents_dir)
        state_root=helloagents_dir if helloagents_dir.is_absolute() else (repo_root/helloagents_dir)
        self.paths=CodeAgentPaths(
            repo_root=repo_root,
            notes_dir=state_root/"notes",
            memory_dir=state_root/"memory",
            sessions_dir=state_root/"sessions",
            logs_dir=state_root/"logs",
        )
        #确保所有必要目录存在
        self.paths.helloagents_dir.mkdir(parents=True, exist_ok=True)
        self.paths.notes_dir.mkdir(parents=True, exist_ok=True)
        self.paths.sessions_dir.mkdir(parents=True,exist_ok=True)
        #memory/logs 仅在需要时创建

        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.llm = llm or HelloAgentsLLM()

        # 初始化会话日志
        self.session_logger = SessionLogger(
            output_dir=self.paths.logs_dir,
            agent_name="code_agent",
            agent_type="react",
            model=self.llm.model or "unknown",
            provider=self.llm.provider or "unknown",
            session_id=self.session_id,
        )

        #初始化工具
        self.registry = ToolRegistry()
        self.registry.register_tool(
            ListFilesTool(project_root=self.paths.repo_root, working_dir=self.paths.repo_root)
        )
        self.registry.register_tool(SearchFilesByNameTool(project_root=self.paths.repo_root))
        self.registry.register_tool(GrepTool(project_root=self.paths.repo_root))
        self.registry.register_tool(ReadTool(project_root=self.paths.repo_root))
        self.registry.register_tool(WriteTool(project_root=self.paths.repo_root))
        self.registry.register_tool(EditTool(project_root=self.paths.repo_root))
        self.registry.register_tool(MultiEditTool(project_root=self.paths.repo_root))
        self.registry.register_tool(TodoWriteTool(project_root=self.paths.repo_root))
        self.registry.register_tool(BashTool(project_root=self.paths.repo_root))
        self.note_tool=NoteTool(workspace=str(self.paths.notes_dir))
        self.registry.register_tool(self.note_tool)

        # ReActAgent 的工具注册表
        # 核心工具：terminal, note, memory, plan
        # 扩展上下文工具：context_fetch（让模型按需获取更多证据）
        self.registry.register_tool(PlanTool(self.llm, prompt_path=str(self.paths.prompts_dir / "plan.md")))

        #注册上下文获取工具
        self.context_fetch_tool=ContextFetchTool(
            workspace=str(self.paths.repo_root),
            note_tool=self.note_tool,
            memory_tool=None,
            max_tokens_per_source=800,
            context_lines=5,
        )
        self.registry.register_tool(self.context_fetch_tool)

        #初始化上下文构建器（lazy_fetch=True：只构建保底上下文）
        # 初始化上下文构建器（lazy_fetch=True：只构建保底上下文）
        self.context_builder = ContextBuilder(
            memory_tool=None,
            rag_tool=None,
            config=ContextConfig(
                max_tokens=8000,
                reserve_ratio=0.15,
                max_history_turns=10,
                enable_compression=True,
                include_output_format=False,
                lazy_fetch=True,  # 按需探索模式
            ),
            llm=self.llm,
        )

        # 加载自定义 Prompt 并初始化 ReActAgent
        react_prompt = (self.paths.prompts_dir / "react.md").read_text(encoding="utf-8")
        summarize_prompt = (self.paths.prompts_dir / "summarize_observation.md").read_text(encoding="utf-8")

        def _summarize_observation(tool_name:str,tool_input:str,observation:str)->str:
            """使用LLM压缩工具输出（避免将巨大的原始输出放入Prompt）"""
            truncated=observation
            if len(truncated)>8000:
                truncated=truncated[:8000]+"\n...truncated...\n"
            user_msg=(
                f"Tool: {tool_name}\n"
                f"Input: {tool_input}\n\n"
                f"Output:\n{truncated}"
            )
            return self.llm.invoke(
                [
                    {"role": "system", "content": summarize_prompt},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=400,
            ) or ""

        self.react=ReActAgent(
            name="code_agent",
            llm=self.llm,
            tool_registry=self.registry,
            max_steps=20,
            custom_prompt=react_prompt,
            observation_summarizer=_summarize_observation,
            summarize_threshold_chars=1800,
            session_logger=self.session_logger,
        )

        base_system=(self.paths.prompts_dir/"system.md").read_text(encoding="utf-8")
        self.tools_reference_path = self.paths.prompts_dir / "tools.md"
        self.system_prompt = base_system
        self.history: List[Message] = []
        self.recent_tool_packets: List[ContextPacket] = []
        self.last_direct_reply: bool = False

    def _is_chitchat(self, text: str) -> bool:
        """判断是否为闲聊，避免不必要的工具调用"""
        t = (text or "").strip().lower()
        return t in {"hi", "hello", "hey", "yo", "你好", "您好", "在吗", "嗨", "哈喽"}

    def _is_history_query(self, text: str) -> bool:
        """判断是否为'回顾刚才说了什么'的元请求"""
        t = (text or "").strip().lower()
        patterns = [
            "说了什么",
            "刚才说了什么",
            "之前说了什么",
            "what did i say",
            "what did we say",
            "recap",
            "summary of conversation",
        ]
        return any(p in t for p in patterns)

    def _reply_with_recent_history(self,limit:int=6)->str:
        """生成最近对话的简要回顾"""
        #只取用户/助手消息
        items=[m for m in self.history if m.role in {"user","assistant"}][-limit*2:]
        if not items:
            return "目前还没有可回顾的对话历史。"
        lines=[]
        for m in items:
            role = "你" if m.role == "user" else "助手"
            lines.append(f"- {role}: {m.content}")
        return "下面是最近的对话回顾：\n" + "\n".join(lines)

    # 以下两个方法在 lazy_fetch 模式下不再主动调用，
    # 扩展上下文改由模型通过 context_fetch 工具按需获取。
    # 保留这些方法以支持 lazy_fetch=False 的传统模式。

    def _note_packets(self,query:str)->List[ContextPacket]:
        """检索相关笔记并封装为ContextPacket"""
        packets: List[ContextPacket] = []
        if self._is_chitchat(query):
            return packets

        try:
            #获取最近的阻碍（Blocker）
            blockers = self.note_tool.run({"action": "list", "note_type": "blocker", "limit": 2})
            if blockers and isinstance(blockers, str) and "暂无" not in blockers:
                packets.append(ContextPacket(content=f"[Notes:blocker]\n{blockers}", metadata={"source": "note"}))
            # 搜索相关笔记
            hits = self.note_tool.run({"action": "search", "query": query, "limit": 3})
            if hits and isinstance(hits, str) and "未找到" not in hits:
                packets.append(ContextPacket(content=f"[Notes:search]\n{hits}", metadata={"source": "note"}))
        except Exception:
            pass
        return packets

    def _memory_packets(self, query: str) -> List[ContextPacket]:
        """检索相关记忆并封装为 ContextPacket"""
        packets: List[ContextPacket] = []
        if self._is_chitchat(query):
            return packets
        try:
            hits = self.memory_tool.run(
                {"action": "search", "query": query, "memory_types": self.memory_tool.memory_types, "limit": 5, "min_importance": 0.0}
            )
            if hits and isinstance(hits, str) and "未找到" not in hits:
                packets.append(ContextPacket(content=f"[Memory]\n{hits}", metadata={"source": "memory"}))
        except Exception:
            pass
        return packets

    def _persist_session(self) -> None:
        """持久化当前会话到 JSON 文件"""
        p = self.paths.sessions_dir / f"{self.session_id}.json"
        data = {
            "session_id": self.session_id,
            "updated_at": datetime.now().isoformat(),
            "history": [
                {"role": m.role, "content": m.content, "timestamp": m.timestamp.isoformat()} for m in self.history[-50:]
            ],
        }
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def run_turn(self,user_input:str)->str:
        """
        执行一轮对话：
        1.收集上下文（笔记、记忆、最近工具输出）
        2.构建完整prompt
        3.运行React循环
        4.更新历史并持久化
        """

        #空输入：提示而不进入ReAct
        if not user_input.strip():
            return "请提供具体指令或问题。"

        # 闲聊/问候：直接回复，避免 ReAct 的严格格式解析失败，也避免无谓的工具调用。
        if self._is_chitchat(user_input):
            self.last_direct_reply = True
            reply = "你好！我是 Code Agent，可以帮你按需探索代码仓库、生成补丁并在确认后落盘。你想做什么？（例如：分析项目结构 / 搜索某个类 / 修复一个报错）"
            self.history.append(Message(content=user_input, role="user", timestamp=datetime.now()))
            self.history.append(Message(content=reply, role="assistant", timestamp=datetime.now()))
            if len(self.history) > 50:
                self.history=self.history[-50:]
            self._persist_session()
            return reply
        self.last_direct_reply = False

        # 元请求：回顾最近对话
        if self._is_history_query(user_input):
            self.last_direct_reply = True
            reply=self._reply_with_recent_history(limit=6)
            self.history.append(Message(content=user_input, role="user", timestamp=datetime.now()))
            self.history.append(Message(content=reply, role="assistant", timestamp=datetime.now()))
            if len(self.history) > 50:
                self.history=self.history[-50:]
            self._persist_session()
            return reply

        # 若检测到明显多步骤词汇，向模型追加轻量提示（不强制，只提高倾向）
        multistep_hint = ""
        multi_patterns = ["分步", "步骤", "三步", "计划", "改造", "完成后", "多步", "多步骤"]
        if any(p in user_input for p in multi_patterns):
            multistep_hint = "提示：本任务包含多个步骤，先用 todo 记录/更新，再执行；收尾用 todo list 汇总。"

        # 构建用户上下文（对话历史 + 工具摘要），拼入 user_input 尾巴
        history_lines = [
            f"[{m.role}] {m.content}"
            for m in self.history[-10:]  # 最近 10 条
        ] if self.history else []

        tool_summaries = []
        for packet in self.recent_tool_packets[-3:]:
            tool_summaries.append(packet.content)

        extra_parts = []
        if multistep_hint:
            extra_parts.append(multistep_hint)
        if history_lines:
            extra_parts.append("## 对话历史\n" + "\n".join(history_lines))
        if tool_summaries:
            extra_parts.append("## 上次工具结果摘要\n" + "\n".join(tool_summaries))

        full_input = user_input
        if extra_parts:
            full_input = user_input + "\n\n" + "\n\n".join(extra_parts)

        response = self.react.run(full_input, max_tokens=8000)

        #收集本轮工具的执行证据（已在 ReActAgent 内部摘要)
        try:
            tool_summaries:List[str]=[]
            todo_used=False
            todo_listed=False
            for item in getattr(self.react,"last_trace",[])[-6:]:
                summary=item.get("observation_summary")
                tname=item.get("tool_name")
                if tname=="todo":
                    todo_used=True
                    if "list" in str(item.get("tool_input","")):
                        todo_listed=True
                if summary:
                    tool_summaries.append(
                        f"[{item.get('tool_name')}] {item.get('tool_input')}\n{summary}"
                    )
            if tool_summaries:
                self.recent_tool_packets.append(
                    ContextPacket(
                        content="[Tool Evidence]\n" + "\n\n".join(tool_summaries),
                        metadata={"type": "tool_result", "source": "react"},
                    )
                )
                #保持缓冲区较小
                if len(self.recent_tool_packets) > 8:
                    self.recent_tool_packets = self.recent_tool_packets[-8:]
        except Exception:
            pass

        # 更新历史记录 (保留最近 50 条)
        self.history.append(Message(content=user_input, role="user", timestamp=datetime.now()))
        self.history.append(Message(content=response, role="assistant", timestamp=datetime.now()))
        if len(self.history) > 50:
            self.history = self.history[-50:]
        self._persist_session()
        try:
            if todo_used and not todo_listed:
                todo_snapshot = self.registry.execute_tool("todo", {"action": "list"})
                response = f"{response}\n\nTodo board:\n{todo_snapshot}"
        except Exception:
            pass
        return response
