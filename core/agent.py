"""Agent基类"""

from abc import ABC, abstractmethod
from typing import Optional
from .message import Message
from .llm import HelloAgentsLLM
from .config import Config
from .session_logger import SessionLogger, _NoopLogger


class Agent(ABC):
    """Agent基类"""

    def __init__(
            self,
            name: str,
            llm: HelloAgentsLLM,
            system_prompt: Optional[str] = None,
            config: Optional[Config] = None,
            session_logger: Optional[SessionLogger] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []
        self._log = session_logger or _NoopLogger()

    @abstractmethod
    def run(self,input_text:str,**kwargs)->str:
        """运行Agent"""
        pass

    def add_message(self, messages:Message):
        """添加消息到历史记录"""
        self._history.append(messages)

    def clear_history(self):
        """清空历史记录"""

    def get_history(self)->list[Message]:
        """获取历史记录"""
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name},provider={self.llm.provider})"

    def __repr__(self) ->str:
        return self.__str__()