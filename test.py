from typing import Dict, Any, List, Optional
import subprocess
import os
from pathlib import Path
import shlex
import re


ALLOWED_COMMANDS = {
        # 文件列表与信息
        'ls', 'dir', 'tree','dir',
        # 文件内容查看
        'cat', 'head', 'tail', 'less', 'more',
        # 文件搜索
        'find', 'grep', 'egrep', 'fgrep', 'rg',
        # 文本处理
        'wc', 'sort', 'uniq', 'cut', 'awk', 'sed',
        # shell 常见内建（用于管道/小脚本；仍受整体策略约束）
        'echo', 'printf',
        # 目录/文件创建（受路径沙箱约束）
        'mkdir',
        # 目录操作
        'pwd', 'cd',
        # 文件信息
        'file', 'stat', 'du', 'df',
        # 其他
        'which', 'whereis',
        # 版本控制（只读子命令会被进一步限制）
        'git',
    }

def _shell_all_commands_whitelisted(command: str) -> bool:
    """
    静态检查shell命令中的所有段是否都在白名单中（尽力而为的检查）

    此方法通过分割shell命令为多个段，然后检查每个段的第一个命令是否在白名单中。
    这是对shell命令的安全预检查，用于在未允许危险操作时确保命令的安全性。

    安全检查策略：
    1. 使用shell元字符分割命令为多个独立段
    2. 对每个段进行命令解析，提取基础命令
    3. 检查基础命令是否在白名单中
    4. 对git命令进行特殊处理，只允许安全的只读子命令

    Args:
        command: 要检查的完整shell命令字符串

    Returns:
        bool: 如果所有命令段都在白名单中则返回True，否则返回False

    Note:
        - 这是一个尽力而为的检查，不能保证100%准确
        - 对于复杂的shell语法，可能存在误判
        - git命令只允许status和diff子命令，其他子命令被视为危险操作
    """
    # 预处理：将换行符替换为空格，便于统一处理
    cmd = command.replace("\n", " ")

    # 分割命令为多个段，每个段包含一个独立的命令
    segments = _split_shell_segments(cmd)

    # 对每个命令段进行安全检查
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue  # 跳过空段

        try:
            # 使用shlex进行安全的命令分割，处理引号和转义
            argv = shlex.split(seg)
        except Exception:
            # 如果解析失败，认为不安全
            return False

        if not argv:
            continue  # 跳过空命令段

        # 获取基础命令（命令名的第一部分）
        base = argv[0]
        if base == "cmd":
            base=argv[2]

        # 检查基础命令是否在白名单中
        if base not in ALLOWED_COMMANDS:
            return False

        # 对git命令进行特殊处理
        # 只允许安全的只读子命令，其他子命令被视为危险操作
        if base == "git":
            if len(argv) < 2:
                return False  # git命令必须包含子命令
            if argv[1] not in {"status", "diff"}:
                return False  # 只允许status和diff子命令

    # 所有命令段都通过了安全检查
    return True

# --- shell parsing helpers (ignore operators inside quotes) ---
def _split_shell_segments(command: str) -> List[str]:
    """
    按管道/逻辑与/逻辑或/分号操作符分割shell命令，忽略引号内的操作符。
    返回分割后的段列表（不包含操作符）。

    这个方法用于分析复杂的shell命令，确保每个段都是安全的。

    Args:
        command: 要分割的shell命令字符串

    Returns:
        List[str]: 分割后的命令段列表
    """
    ops = ["||", "&&", "|", ";"]
    segs: List[str] = []
    buf: List[str] = []
    i = 0
    quote: Optional[str] = None
    while i < len(command):
        ch = command[i]
        if ch in {"'", '"'}:
            if quote is None:
                quote = ch
            elif quote == ch:
                quote = None
            buf.append(ch)
            i += 1
            continue
        if ch == "\\":
            buf.append(ch)
            if i + 1 < len(command):
                buf.append(command[i + 1])
                i += 2
            else:
                i += 1
            continue
        if quote is None:
            matched = False
            for op in ops:
                if command.startswith(op, i):
                    seg = "".join(buf).strip()
                    if seg:
                        segs.append(seg)
                    buf = []
                    i += len(op)
                    matched = True
                    break
            if matched:
                continue
        buf.append(ch)
        i += 1
    seg = "".join(buf).strip()
    if seg:
        segs.append(seg)
    return segs



command="cat terminal_tool.py"
current_dir="D:\ACode\HelloCodeAgent"
result = subprocess.run(
                command,
                shell=True,
                cwd=str(current_dir),
                capture_output=True,
                text=True,
                timeout=30,
                env=os.environ.copy(),
            )
output = (result.stdout or "") + (result.stderr or "")
print(_shell_all_commands_whitelisted(command))
print(output)