# 工具使用指南

遵循"明确何时用 / 何时不用 / 如何用 / 调用示例"，避免盲目调用。

## context_fetch
用途：聚合搜索（files/notes/memory/tests），自动摘要，控制预算。
- 何时用：需要"更多证据"时；搜类名/函数名/错误栈；需要相关笔记/记忆；比单独 note/memory 搜索更省步数。
- 何时不用：已经有足够证据；用户仅问对话历史。
- 调用示例：`context_fetch[{"sources":["files","notes"],"query":"ContextBuilder","paths":"context/**/*.py"}]`

## Bash
用途：执行 shell 命令，支持命令链（&&, ||, ;）和项目根目录内 cd。
- 何时用：构建/测试/运行命令，或其他工具不覆盖的系统工具。
- 何时不用：不要用 shell 做列表/搜索/读取文件（用 LS/Read/Grep/Glob）；禁止交互式命令（vim/nano/top/ssh 等）。
- 参数：command（必填）；directory（可选，默认 "."）；timeout_ms（可选，默认 120000，最大 600000）。
- 安全：沙箱限定项目根目录，禁止 sudo/su/mkfs/dd/rm -rf / 等危险模式；curl/wget 默认禁用。
- 调用示例：`Bash[{"command":"pytest tests"}]`
  - `Bash[{"command":"npm test","directory":"frontend"}]`

## note
用途：结构化笔记（action/decision/blocker/task_state 等），Markdown 持久化。
- 何时用：记录关键结论/风险/阻塞；补丁成功/失败总结；阶段小结。
- 何时不用：临时想法可先留在对话，不必频繁写笔记。
- 示例：`note[{"action":"create","title":"Patch applied","content":"...","note_type":"action","tags":["patch"]}]`

## memory
用途：情景记忆（SQLite），跨会话回忆"发生过什么"。默认不开自动写，需显式添加。
- 何时用：需要在未来回忆本次决策/阻塞/结论；会话结束前写小结；复用过往经验时可先 search。
- 何时不用：即时对话短期内容已有 history；信息尚不确定。
- 示例：
  - `memory[{"action":"add","memory_type":"episodic","content":"完成 hello.html 样式改造...","importance":0.7}]`
  - `memory[{"action":"search","query":"hello.html 样式","memory_types":["episodic"],"limit":5}]`

## plan
用途：显式规划工具，生成分步计划。
- 何时用：任务模糊或明显多步骤；用户要求出计划；执行前需要拆解。
- 何时不用：非常简单的一步任务。
- 示例：`plan[{"goal":"优化渲染性能，先梳理瓶颈再改"}]`

## Edit
用途：对已有文件做精确的单次文本替换。old_string 必须在文件中唯一出现。
- 何时用：修改文件中的单个位置（函数名、变量、配置值等）。
- 何时不用：新建文件用 write_file；同一文件多处修改用 edit_file_multi。
- 参数：path（必填）；old_string（必填，从 Read 输出精确复制）；new_string（必填）；dry_run（可选）。
- 重要：编辑前必须先 Read 文件；遇到 CONFLICT 重新 Read 再试。
- 调用示例：
  - `Edit[{"path":"src/utils.py","old_string":"def old_func(x):","new_string":"def new_func(x):"}]`

## MultiEdit
用途：对同一文件做多次独立修改，原子性批量应用。所有 old_string 均基于原始文件匹配。
- 何时用：同一文件需要多处修改，且修改互不依赖。
- 何时不用：单次修改用 edit_file；修改间有依赖则分开调用 edit_file。
- 参数：path（必填）；edits（数组，每项含 old_string/new_string）；dry_run（可选）。
- 重要：所有 old_string 必须基于原文件（非中间状态）；编辑区域不可重叠。
- 调用示例：
  - `MultiEdit[{"path":"src/config.py","edits":[{"old_string":"DEBUG=True","new_string":"DEBUG=False"},{"old_string":"LOG_LEVEL=\"INFO\"","new_string":"LOG_LEVEL=\"WARNING\""}]}]`

## LS
用途：列出目录内容，支持分页和隐藏文件开关。
- 何时用：探索目录结构、查看文件夹内容。
- 何时不用：搜索文件名用 search_files_by_name；搜索内容用 grep_tool。
- 参数：path（可选，默认 "."）；offset（可选，默认 0）；limit（可选，默认 100，最大 200）；include_hidden（可选）。
- 调用示例：`LS[{"path":"src","limit":50}]`

## Read
用途：读取文件内容，带行号（格式：`   1 | content`）。
- 何时用：查看文件内容以获取编辑上下文。
- 何时不用：禁止用 bash cat/less/head/tail 代替。
- 参数：path（必填）；start_line（可选，默认 1）；limit（可选，默认 500，最大 2000）。
- 调用示例：
  - `Read[{"path":"src/main.py"}]`
  - `Read[{"path":"src/main.py","start_line":101,"limit":100}]`

## Grep
用途：用正则表达式搜索文件内容，结果按修改时间排序（最新在前）。
- 何时用：搜索代码内容（类名、函数名、TODO、错误信息等）。
- 何时不用：搜索文件名用 search_files_by_name；浏览目录用 list_files。
- 参数：pattern（必填，正则）；path（可选，默认 "."）；include（可选，glob 过滤）；case_sensitive（可选，默认 false）；limit（可选，默认 100）。
- 调用示例：
  - `Grep[{"pattern":"TODO","include":"**/*.ts"}]`
  - `Grep[{"pattern":"class\\s+\\w+","path":"src"}]`

## Glob
用途：用 glob 模式按文件名查找文件（如 `**/*.ts`）。
- 何时用：按文件名/模式查找文件。
- 何时不用：搜索内容用 grep_tool；浏览目录用 list_files。
- 参数：pattern（必填，glob 非正则）；path（可选，默认 "."）；limit（可选，默认 50，最大 200）；include_hidden（可选）。
- 调用示例：
  - `Glob[{"pattern":"**/*.md","path":"."}]`
  - `Glob[{"pattern":"*.ts","path":"src"}]`

## TodoWrite
用途：多步骤任务跟踪。覆盖式更新（每次提交完整列表）。状态：pending | in_progress（最多 1 个）| completed | cancelled。
- 何时用：3 步以上或多文件/多特性；用户列出多项需求；跨回合/需确认的任务。
- 何时不用：单一步、琐碎或纯问答。
- 参数：summary（必填）；todos（数组，每项含 content 和 status）。
- 示例：
  - `TodoWrite[{"summary":"实现用户认证","todos":[{"content":"设计认证流程","status":"in_progress"},{"content":"创建登录接口","status":"pending"},{"content":"添加 JWT 校验","status":"pending"}]}]`

## Write
用途：创建新文件或完整覆写已有文件。必须提供完整内容，非补丁。
- 何时用：新建文件；完整替换已有文件内容。
- 何时不用：局部修改用 edit_file 或 edit_file_multi。
- 参数：path（必填）；content（必填，完整内容）；dry_run（可选）。
- 重要：覆盖已有文件前必须先 Read（框架自动注入冲突检测）；目录自动创建。
- 调用示例：
  - `Write[{"path":"src/helper.py","content":"def greet(name):\\n    return f'Hello, {name}!'\\n"}]`

## 重要提醒
- 先用已有上下文推理，不足再调用工具；避免无端多次搜索。
- 写/改文件必须通过 edit_file / edit_file_multi / write_file，禁止 cat > file / Here-Doc / 重定向写盘。
- 编辑文件前必须先 Read，修改后检查返回的 diff。
- todo_write 只保持 1 个 in_progress；完成立即标记；阻塞则标记为 cancelled 并新增任务。
- 工具响应遵循统一协议，顶层字段仅：status / data / text / error / stats / context。
