"""DefaultChatter 插件配置定义。"""

from __future__ import annotations

from typing import ClassVar, Literal

from src.core.components.base.config import BaseConfig, Field, SectionBase, config_section


class DefaultChatterConfig(BaseConfig):
    """DefaultChatter 配置。"""

    config_name: ClassVar[str] = "config"
    config_description: ClassVar[str] = "DefaultChatter 配置"

    @config_section("plugin")
    class PluginSection(SectionBase):
        """插件基础配置。"""

        @config_section("theme_guide")
        class ThemeGuideSection(SectionBase):
            """不同聊天类型的人设/语气引导。"""

            private: str = Field(
                default="你当前正处于“私下聊天（私聊）”环境中，你可以以更贴近一对一陪伴感的交流方式与用户互动，关注用户情绪并提供更直接、细腻的回应。",
                description="私聊场景的额外提示词",
            )
            group: str = Field(
                default="你当前正处于“群聊”环境中，你需要注意多人对话上下文，优先回应与当前话题强相关或明确提及你的内容，表达简洁自然。",
                description="群聊场景的额外提示词",
            )

        enabled: bool = Field(default=True, description="是否启用 DefaultChatter")
        mode: Literal["enhanced", "classical"] = Field(
            default="enhanced",
            description="执行模式: enhanced/classical",
        )
        theme_guide: ThemeGuideSection = Field(
            default_factory=ThemeGuideSection,
            description="按聊天类型区分的额外提示词",
        )

    plugin: PluginSection = Field(default_factory=PluginSection)
