"""
# ============================================================
# YAML Prompt 加载器 (模板版本化管理)
# ← WeKnora: internal/agent/prompts.go
#   - BuildSystemPrompt(): 动态变量插值 + 模板选择
#   - renderPromptPlaceholders(): {{variable}} 替换
#   - GetProgressiveRAGSystemPrompt(): 从 YAML 加载模板
# ============================================================

本模块实现:
- 从 YAML 文件加载 Prompt 模板
- Jinja2 风格变量插值 ({{ variable }})
- 模板缓存 (首次加载后缓存，避免重复 I/O)
- 多模板选择 (按 id 查找指定模板)

设计要点:
- 每个 YAML 文件包含一个 `templates` 列表，每个模板有唯一的 `id`
- `default: true` 标记默认模板
- 支持 `content` (系统提示词) 和 `user` (用户提示词) 两种模板
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from jinja2 import Template

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    单个 Prompt 模板
    ← WeKnora: prompts.go 中的模板结构
    """

    def __init__(self, template_id: str, data: dict[str, Any]):
        self.id = template_id
        self.name: str = data.get("name", template_id)
        self.description: str = data.get("description", "")
        self.is_default: bool = data.get("default", False)
        self.content: str = data.get("content", "")
        self.user: str = data.get("user", "")  # user 提示词 (可选)

    def render(self, **variables: Any) -> str:
        """
        使用 Jinja2 渲染 content 模板

        Args:
            **variables: 模板变量，如 query="xxx", contexts="yyy"

        Returns:
            渲染后的文本
        """
        template = Template(self.content)
        return template.render(**variables)

    def render_user(self, **variables: Any) -> str:
        """
        渲染 user 提示词模板 (如果存在)
        """
        if not self.user:
            return ""
        template = Template(self.user)
        return template.render(**variables)

    def __repr__(self) -> str:
        return f"PromptTemplate(id={self.id!r}, name={self.name!r})"


class PromptRegistry:
    """
    Prompt 模板注册表
    ← WeKnora: prompts.go 中的模板管理 + GetProgressiveRAGSystemPrompt()
    管理所有 YAML 文件中加载的 Prompt 模板
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        """
        Args:
            prompts_dir: Prompt 模板目录路径，默认从 settings 读取
        """
        settings = get_settings()
        self._prompts_dir = prompts_dir or settings.prompts_dir
        self._templates: dict[str, dict[str, PromptTemplate]] = {}  # {filename: {template_id: PromptTemplate}}
        self._loaded = False

    def _load_all(self) -> None:
        """加载所有 YAML 模板文件"""
        if self._loaded:
            return

        if not self._prompts_dir.exists():
            logger.warning("Prompt 模板目录不存在: %s", self._prompts_dir)
            self._loaded = True
            return

        for yaml_file in sorted(self._prompts_dir.glob("*.yaml")):
            try:
                self._load_file(yaml_file)
            except Exception as e:
                logger.error("加载 Prompt 模板失败: %s — %s", yaml_file.name, e)

        self._loaded = True
        logger.info("已加载 %d 个 Prompt 文件", len(self._templates))

    def _load_file(self, filepath: Path) -> None:
        """
        加载单个 YAML 模板文件
        ← WeKnora: prompts.go GetProgressiveRAGSystemPrompt() 读取 YAML
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "templates" not in data:
            logger.debug("跳过空的 Prompt 文件: %s", filepath.name)
            return

        file_templates: dict[str, PromptTemplate] = {}
        for tmpl_data in data["templates"]:
            tmpl_id = tmpl_data.get("id", "")
            if not tmpl_id:
                continue
            file_templates[tmpl_id] = PromptTemplate(tmpl_id, tmpl_data)

        if file_templates:
            self._templates[filepath.stem] = file_templates
            logger.debug("加载 Prompt: %s → %d 个模板", filepath.name, len(file_templates))

    # ----------------------------------------------------------------
    # 查询接口
    # ----------------------------------------------------------------

    def get(self, template_id: str, filename: Optional[str] = None) -> Optional[PromptTemplate]:
        """
        按 ID 查找模板

        Args:
            template_id: 模板 ID，如 "adaptive_rag_system"
            filename: 限定在哪个 YAML 文件中查找 (不含 .yaml 后缀)

        Returns:
            PromptTemplate 或 None
        """
        self._load_all()

        if filename:
            file_templates = self._templates.get(filename, {})
            return file_templates.get(template_id)

        # 在所有文件中查找
        for file_templates in self._templates.values():
            if template_id in file_templates:
                return file_templates[template_id]

        logger.warning("Prompt 模板未找到: %s", template_id)
        return None

    def get_default(self, filename: str) -> Optional[PromptTemplate]:
        """
        获取文件中的默认模板 (default: true)

        Args:
            filename: YAML 文件名 (不含 .yaml 后缀)

        Returns:
            默认的 PromptTemplate，如果没有标记 default 则返回第一个
        """
        self._load_all()

        file_templates = self._templates.get(filename, {})
        if not file_templates:
            return None

        for tmpl in file_templates.values():
            if tmpl.is_default:
                return tmpl

        # 没有标记 default，返回第一个
        return next(iter(file_templates.values()))

    def list_files(self) -> list[str]:
        """列出所有已加载的模板文件"""
        self._load_all()
        return sorted(self._templates.keys())

    def list_templates(self, filename: Optional[str] = None) -> list[PromptTemplate]:
        """
        列出模板

        Args:
            filename: 限定文件名，None 则列出所有

        Returns:
            模板列表
        """
        self._load_all()
        result: list[PromptTemplate] = []
        if filename:
            result.extend(self._templates.get(filename, {}).values())
        else:
            for file_templates in self._templates.values():
                result.extend(file_templates.values())
        return sorted(result, key=lambda t: (not t.is_default, t.id))


# ================================================================
# 便捷函数
# ================================================================


@lru_cache
def get_prompt_registry() -> PromptRegistry:
    """获取全局 Prompt 注册表单例"""
    return PromptRegistry()


def load_prompt(
    template_id: str,
    filename: str | None = None,
    **variables: Any,
) -> str:
    """
    便捷函数: 加载模板并渲染

    Args:
        template_id: 模板 ID
        filename: 限定 YAML 文件
        **variables: 渲染变量

    Returns:
        渲染后的 Prompt 文本

    Example:
        >>> prompt = load_prompt(
        ...     "adaptive_rag_system",
        ...     filename="system_prompt",
        ...     query="营收增长驱动因素",
        ...     contexts="...",
        ...     complexity="complex",
        ...     language="中文"
        ... )
    """
    registry = get_prompt_registry()
    template = registry.get(template_id, filename)
    if template is None:
        raise ValueError(f"Prompt 模板未找到: {template_id}")
    return template.render(**variables)


def load_prompt_with_default(
    filename: str,
    **variables: Any,
) -> str:
    """
    加载文件的默认模板并渲染

    Args:
        filename: YAML 文件名 (不含后缀)
        **variables: 渲染变量

    Returns:
        渲染后的 Prompt 文本
    """
    registry = get_prompt_registry()
    template = registry.get_default(filename)
    if template is None:
        raise ValueError(f"未找到默认模板: {filename}")
    return template.render(**variables)
