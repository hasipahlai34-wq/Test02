# Windows GBK 编码修复说明

## 问题背景

项目代码中包含大量 emoji 字符（🧠🤔📊 等）用于 CLI 输出。Windows 系统下 Python 默认使用 GBK 编码处理 stdout/stderr，无法编码这些 Unicode 字符，导致启动时抛出 `UnicodeEncodeError`：

```
UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f9e0' in position 0: illegal multibyte sequence
```

## 修复方案

在每个直接运行的入口脚本顶部，添加 UTF-8 编码强制逻辑。遵循两个原则：

1. **仅 Windows 生效** — 通过 `sys.platform == "win32"` 守卫，不影响 Linux/macOS
2. **静默降级** — `try/except` 包裹，管道/重定向场景下 `reconfigure()` 会抛 `OSError`，此时保持默认编码继续运行

### 修复代码模板

```python
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass  # 管道/重定向场景不支持 reconfigure，保持默认编码
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
```

`setdefault` 确保已有值不被覆盖（用户可能手动设置了 `PYTHONIOENCODING`）。

## 涉及文件

| 文件 | 行号 | 说明 |
|------|------|------|
| `main.py` | L23-L32 | 项目主入口，所有子命令（ui/ask/eval/serve/chat/ingest）的启动点 |
| `cli/review_pending.py` | L28-L35 | HITL 人工审核 CLI 工具，独立运行的审核管理入口 |

## 附加修复

### Streamlit 邮箱提示阻塞

Streamlit 首次运行时会阻塞等待用户输入邮箱，在无终端交互环境（后台进程/CI）中导致进程挂起。

**修复方式：** 创建 `.streamlit/config.toml`，设置：

```toml
[browser]
gatherUsageStats = false

[server]
headless = true
```

同时在 `main.py` 的 `cmd_ui()` 中设置环境变量 `STREAMLIT_SUPPRESS_ONBOARDING_EMAIL=1`（双重保险）。

## 扩展现有项目

如果将来新增需要直接 `python xxx.py` 运行的入口脚本，且其中包含 emoji 打印，按以下 checklist 操作：

1. [ ] 在 `import sys` 后、业务代码前插入上述「修复代码模板」
2. [ ] 确保脚本中已有 `import os`（如无则添加）
3. [ ] 运行 `python -m pytest tests/ -x -q` 确认 65 个测试全绿
4. [ ] 直接在终端运行该脚本，确认 emoji 正常显示

## 测试结果

```
65 passed, 1 warning in 5.03s
```

所有已有测试不受影响，因为修复代码仅在 stdout/stderr 上操作，不改变任何业务逻辑、API 行为或数据结构。

## 版本记录

| 日期 | 变更 |
|------|------|
| 2026-07-04 | 初始修复：`main.py` + `cli/review_pending.py` + `.streamlit/config.toml` |
| 2026-07-04 | Chat/Embedding Base URL 分离：`config/settings.py` + `src/models/embeddings.py` + `src/types.py` + `src/ingestion/indexer.py` + `.env.example` |
