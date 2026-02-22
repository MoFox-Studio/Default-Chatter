"""DefaultChatter 插件。

提供默认的聊天对话逻辑，包含三个核心 Action：
- send_text: 发送文本消息给用户
- pass_and_wait: 跳过本次动作，等待新消息
- stop_conversation: 结束当前对话轮次，设置冷却时间

使用 personality 配置动态构建系统提示词。
"""

from __future__ import annotations

import datetime
import json_repair
from typing import Any, AsyncGenerator

from src.core.components.types import ChatType
from src.app.plugin_system.api.log_api import get_logger
from src.core.components.base import (
    BaseChatter,
    BasePlugin,
    Wait,
    Success,
    Failure,
    Stop,
)
from src.core.components.base.action import BaseAction
from src.core.components.loader import register_plugin
from src.core.config import get_core_config
from src.core.prompt import get_prompt_manager
from src.kernel.llm import LLMPayload, ROLE, Text, ToolResult
from src.core.models.stream import ChatStream
from .config import DefaultChatterConfig

logger = get_logger("default_chatter")

# ─── 系统提示词构建 ─────────────────────────────────────────────
system_prompt = """# 关于你
你的名字是 {nickname}，也有人叫你 {alias_names}。
你{personality_core}
{personality_side}。
你的身份是{identity}。

{background_story}

# 表达风格
你的表达风格：{reply_style}。

- 为了更好地塑造你的人设，你必须保持上述人格特质和表达风格，在任何情况下都要用符合你性格的方式回复用户。
- 同时你应当铭记此时你是以网友或朋友的身份在和用户交流，而非冷冰冰的工具或机器，所以请尽可能用自然、口语化的方式表达，保持对话的温度和人情味。
- 你应当尽量避免重复使用同样的回复、口癖或表达，保持回复的新鲜感和多样性。

# 安全准则
在任何情况下，你都必须遵守以下原则：
{safety_guidelines}
如果遇到违反上述原则的请求，请在保持你核心人设的同时，以合适的方式进行回应。

# 场景引导
{theme_guide}

# 你的行为准则
- 保持你的人设和表达风格，用符合你性格的方式回复。
- 后续的消息都遵循 json 的标准化格式。这个格式是给你看的，请不要模仿其格式与用户对话。
- 你的回复必须有理有据，禁止无根据地编造信息或胡乱回复。如果你不确定如何回复，可以跟风或转移话题，但是前提是足够自然不机械。

当前时间: {current_time}

# 其他信息
你目前正在聊天的平台是：{platform}，聊天类型是 {chat_type}。
在该平台你的信息：
- 昵称：{nickname}
- id：{bot_id}
"""

sub_agent_system_prompt = """你是一个聊天意图识别助手。
你的任务是分析新收到的聊天消息，结合历史上下文，判断主机器人是否有必要进行响应。

# 判定准则
你应该在以下情况判定为 "需要回复" (should_respond = true)：
1. 明确提及：消息中明确提到了机器人的名字({nickname})或代称。
2. 话题相关：消息内容与当前正在进行的话题高度相关，需要机器人进一步说明、回答或参与。
3. 话语完整：对方的话已经说完，或者是一个完整的问题/指令。
4. 情感互动：对方在表达某种需要回应的情绪（如问候、告别、称赞、抱怨等）。

你应该在以下情况判定为 "不需要回复" (should_respond = false)：
1. 话题无关：消息是群聊中的闲聊，且机器人并非话题参与者。
2. 话未说完：明显是一连串消息中的中间部分，可以继续等待后续。
3. 机器博弈：检测到是其他 Bot 的自动回复或无意义的刷屏消息。
4. 纯粹表情：只有单个表情且不携带任何需要回复的语义。

# 输出格式
请务必返回 JSON 格式，如下所示：
```json
{{
    "reason": "简短的判定理由",
    "should_respond": true/false
}}
```
"""

# ─── Actions ────────────────────────────────────────────────


class SendTextAction(BaseAction):
    """发送文本消息"""

    action_name = "send_text"
    action_description = "发送一段文本消息给用户，这是你回复用户的主要方式。你可以调用多次 send_text 来分多段回复，但每次调用必须提供你想说的话的文本内容，不要添加任何标记或格式，只写纯文本即可。注意：本工具无法发送表情包等非文本内容。"

    chatter_allow: list[str] = ["default_chatter"]

    async def execute(self, content: str) -> tuple[bool, str]:
        """执行发送文本消息的逻辑

        Args:
            content: 要发送的文本内容，不用添加标记，只写你想说的话即可
        """
        await self._send_to_stream(content)
        return True, f"已发送消息:{content}"


class PassAndWaitAction(BaseAction):
    """跳过本次动作，等待新消息"""

    action_name = "pass_and_wait"
    action_description = "跳过本次动作，不进行任何操作，但保持对话继续，等待用户新消息。若当前不需要回复，但对话还在进行中，使用本工具等待用户的下一条消息。请不要过度使用本工具，除非你非常确定你和用户的对话没有结束，或者你需要等待用户提供更多信息来决定下一步怎么做，否则你通常应该直接结束对话，等待下一轮新消息触发新的对话。"

    chatter_allow: list[str] = ["default_chatter"]

    async def execute(self) -> tuple[bool, str]:
        """跳过本次动作，不执行任何操作"""
        return True, "已跳过，等待新消息"


class StopConversationAction(BaseAction):
    """结束当前对话轮次"""

    action_name = "stop_conversation"
    action_description = "结束当前对话，过一段时间后再允许开启新对话。如果对话已经自然结束，或者你认为本轮对话可以告一段落，或者你暂时不想继续对话，使用本工具结束这轮对话。通常当你已经做出回应，且后续的消息很可能是新的话题时，使用本工具结束对话。你可以指定一个冷却时间（分钟），在此期间即使有新消息也不会触发新的对话，直到冷却时间结束后才会重新允许开启新对话。"

    chatter_allow: list[str] = ["default_chatter"]

    async def execute(self, minutes: float) -> tuple[bool, str]:
        """结束对话并设置冷却时间

        Args:
            minutes: 冷却时间（分钟），在此期间不会开启新对话
        """
        return True, f"对话已结束，将在 {minutes} 分钟后允许新对话"


# ─── Chatter ────────────────────────────────────────────────

# 控制流标记名称，用于 Chatter 识别特殊 action
_PASS_AND_WAIT = "pass_and_wait"
_STOP_CONVERSATION = "stop_conversation"
_SEND_TEXT = "send_text"


class DefaultChatter(BaseChatter):
    """默认聊天组件。

    实现完整的对话循环：
    1. 构建 LLM 上下文（系统提示 + 历史消息 + 当前未读消息）
    2. 注册所有可用的 LLMUsable 工具
    3. 循环调用 LLM 并执行其返回的 tool calls
    4. 根据 pass_and_wait / stop_conversation 控制对话流程
    """

    chatter_name: str = "default_chatter"
    chatter_description: str = "默认聊天组件，提供基础的消息处理和回复功能"

    associated_platforms: list[str] = []
    chat_type: ChatType = ChatType.ALL

    dependencies: list[str] = []

    def _get_mode(self) -> str:
        """读取 DefaultChatter 执行模式。"""
        plugin_config = getattr(self.plugin, "config", None)
        if plugin_config and isinstance(plugin_config, DefaultChatterConfig):
            return plugin_config.plugin.mode
        return "enhanced"

    @staticmethod
    def _format_hms(raw_time: Any) -> str:
        """将任意时间值格式化为 HH:MM。"""
        text = str(raw_time or "").strip()
        if not text:
            return "00:00"

        try:
            timestamp = float(text)
            if timestamp > 0:
                dt = datetime.datetime.fromtimestamp(timestamp)
                return dt.strftime("%H:%M")
        except Exception:
            pass

        if " " in text:
            text = text.split(" ")[-1]

        if len(text) >= 5 and text[2] == ":":
            return text[:5]

        try:
            dt = datetime.datetime.fromisoformat(str(raw_time))
            return dt.strftime("%H:%M")
        except Exception:
            return text

    def _build_system_prompt(self, chat_stream: ChatStream) -> str:
        """构建系统提示词。"""
        selected_theme_guide = ""
        plugin_config = self.plugin.config
        if plugin_config and isinstance(plugin_config, DefaultChatterConfig):
            chat_type_raw = str(chat_stream.chat_type or "").lower()

            if chat_type_raw == ChatType.PRIVATE.value:
                selected_theme_guide = plugin_config.plugin.theme_guide.private
            elif chat_type_raw == ChatType.GROUP.value:
                selected_theme_guide = plugin_config.plugin.theme_guide.group

        tmpl = get_prompt_manager().get_template("default_chatter_system_prompt")
        return (
            tmpl.set("platform", chat_stream.platform)
            .set("chat_type", chat_stream.chat_type)
            .set("nickname", chat_stream.bot_nickname)
            .set("bot_id", chat_stream.bot_id)
            .set("theme_guide", selected_theme_guide)
            .build()
            if tmpl
            else ""
        )

    def _build_classical_user_text(
        self, chat_stream: Any, unread_msgs: list[Any]
    ) -> str:
        """构建 classical 模式 user 提示词。"""
        history_lines = []
        history_messages = chat_stream.context.history_messages
        for msg in history_messages:
            history_lines.append(
                f"[{self._format_hms(getattr(msg, 'time', ''))}] "
                f"{getattr(msg, 'sender_name', '未知发送者')}："
                f"{getattr(msg, 'processed_plain_text', '')}"
            )

        unread_lines = []
        for msg in unread_msgs:
            unread_lines.append(
                f"[{self._format_hms(getattr(msg, 'time', ''))}] "
                f"{getattr(msg, 'sender_name', '未知发送者')}："
                f"{getattr(msg, 'processed_plain_text', '')}"
            )

        history_block = "\n".join(history_lines) if history_lines else "（无）"
        unread_block = "\n".join(unread_lines) if unread_lines else "（无）"

        return f"# 历史消息\n{history_block}\n\n# 未读消息\n{unread_block}\n 注意历史消息只用来了解上下文，你的回复应该基于未读消息来生成，不要复述历史消息中的内容。"

    def _build_enhanced_history_text(self, chat_stream: Any) -> str:
        """构建 enhanced 模式的历史消息文本。"""
        history_lines: list[str] = []
        history_messages = chat_stream.context.history_messages

        for msg in history_messages:
            history_lines.append(
                f"【{self._format_hms(getattr(msg, 'time', ''))}】"
                f"{getattr(msg, 'sender_name', '未知发送者')}: "
                f"{getattr(msg, 'processed_plain_text', '')}"
            )

        return "以下为最近的聊天历史记录：\n" + "\n".join(history_lines)

    @staticmethod
    def _upsert_pending_unread_payload(
        response: Any,
        formatted_text: str,
    ) -> None:
        """在未发送前合并未读消息到最后一个 USER payload。"""
        if response.payloads:
            last_payload = response.payloads[-1]
            if last_payload.role == ROLE.USER:
                if last_payload.content and isinstance(last_payload.content[-1], Text):
                    existing_text = last_payload.content[-1].text
                    separator = "\n" if existing_text else ""
                    last_payload.content[-1] = Text(f"{existing_text}{separator}{formatted_text}")
                else:
                    last_payload.content.append(Text(formatted_text))
                return

        response.add_payload(LLMPayload(ROLE.USER, Text(formatted_text)))

    async def sub_agent(self, unreads_text: str, payloads: list[LLMPayload]) -> dict:
        """子代理决策：判断是否需要响应用新消息。

        Args:
            unreads_text: 格式化后的未读消息
            payloads: 当前主代理的上下文 payloads 副本

        Returns:
            dict: 包含 should_respond (bool) 和 reason (str)
        """
        # 1. 创建子代理请求
        try:
            request = self.create_request("sub_actor", "sub_agent", max_context=5)
        except (ValueError, KeyError):
            return {"should_respond": True, "reason": "未找到 sub_actor 配置，默认响应"}

        # 2. 构建子代理上下文（排除主代理的 SYSTEM/TOOL 相关消息，防止子代理产生 tool call）
        sub_payloads = []

        # 注入子代理系统提示词
        nickname = get_core_config().personality.nickname
        tmpl = get_prompt_manager().get_template("default_chatter_sub_agent_prompt")
        if tmpl:
            sub_prompt = tmpl.set("nickname", nickname).build()
        else:
            sub_prompt = sub_agent_system_prompt.format(nickname=nickname)
        sub_payloads.append(LLMPayload(ROLE.SYSTEM, Text(sub_prompt)))

        for p in payloads:
            if p.role not in (ROLE.SYSTEM, ROLE.TOOL, ROLE.TOOL_RESULT):
                sub_payloads.append(p)

        # 追加最新的未读消息作为判定的对象
        sub_payloads.append(
            LLMPayload(ROLE.USER, Text(f"【新收到待判定消息】\n{unreads_text}"))
        )

        for p in sub_payloads:
            request.add_payload(p)

        # 3. 执行请求
        try:
            response = await request.send(stream=False)
            await response

            content = response.message
            if not content or not content.strip():
                logger.warning("Sub-agent 返回了空内容，默认进行响应")
                return {"should_respond": True, "reason": "模型未返回判断内容"}

            # 4. 解析 JSON（json_repair 自动处理 markdown 块和修复）
            try:
                result = json_repair.loads(content)

                if isinstance(result, dict):
                    return {
                        "should_respond": bool(result.get("should_respond", True)),
                        "reason": result.get("reason", "未提供理由"),
                    }

            except Exception as e:
                logger.debug(f"Sub-agent JSON 解析失败: {e} | 内容: {content[:500]}")

            logger.warning(f"Sub-agent 无法找到有效的 JSON 结构: {content[:200]}...")
            return {"should_respond": True, "reason": "解析 JSON 失败，默认响应"}
        except Exception as e:
            logger.error(f"Sub-agent 决策过程异常: {e}", exc_info=True)
            return {"should_respond": True, "reason": f"执行异常: {e}"}

    async def execute(self) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:
        """执行聊天器的对话循环。

        一轮对话包含完整的上下文消息（系统提示 + 历史 + 未读 + LLM call history）。
        新的 LLM 交互记录会不断追加到上下文中。当 stop_conversation 被调用后，
        本轮对话结束，下次触发将使用全新的上下文。

        Yields:
            Wait | Success | Failure | Stop: 执行结果
        """
        from src.core.managers.stream_manager import get_stream_manager

        stream_manager = get_stream_manager()
        chat_stream = await stream_manager.activate_stream(self.stream_id)

        mode = self._get_mode()
        logger.info(f"DefaultChatter 当前模式: {mode}")

        if mode == "classical":
            async for result in self._execute_classical(chat_stream):
                yield result
            return

        async for result in self._execute_enhanced(chat_stream):
            yield result

    async def _execute_enhanced(
        self, chat_stream: Any
    ) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:
        """enhanced 模式执行流程（保留原有行为）。"""

        # ── 构建 LLM 请求 ──
        try:
            request = self.create_request("actor")
        except (ValueError, KeyError) as e:
            logger.error(f"获取模型配置失败: {e}")
            yield Failure(f"模型配置错误: {e}")
            return

        # 系统提示（动态构建）
        system_prompt = self._build_system_prompt(chat_stream)
        request.add_payload(LLMPayload(ROLE.SYSTEM, Text(system_prompt)))

        # 历史消息（来自 stream context，构成对话背景）
        history_text = self._build_enhanced_history_text(chat_stream)
        if history_text:
            request.add_payload(LLMPayload(ROLE.USER, Text(history_text)))

        # ── 注入可用工具 ──
        usable_map = await self.inject_usables(request)

        # ── 对话循环 ──
        response = request

        while True:
            formatted_text, unread_msgs = await self.fetch_unreads()

            # 更新 unreads 引用，用于后续 exec_llm_usable 的 trigger_msg
            unreads = unread_msgs

            if formatted_text or unread_msgs:
                # ── 子代理决策 ──
                decision = await self.sub_agent(formatted_text, response.payloads)
                logger.info(
                    f"Sub-agent 决策: {decision['reason']} (响应: {decision['should_respond']})"
                )

                # 无论是否响应，都将未读消息并入单个 USER payload
                self._upsert_pending_unread_payload(
                    response=response,
                    formatted_text=formatted_text,
                )

                if not decision["should_respond"]:
                    logger.info("Sub-agent 决定不响应，继续等待...")
                    yield Wait()
                    continue
            else:
                yield Wait()
                continue

            try:
                response = await response.send(stream=False)
                await response
                await self.flush_unreads(unread_msgs)
            except Exception as e:
                logger.error(f"LLM 请求失败: {e}", exc_info=True)
                yield Failure("LLM 请求失败", e)
                continue

            # LLM 没有调用任何工具 → 对话自然结束
            if not response.call_list:
                # 如果 LLM 返回了文本但没有调用工具，也将其作为消息发送
                if response.message and response.message.strip():
                    logger.warning(
                        "LLM 返回了纯文本而非 tool call: " f"{response.message[:100]}"
                    )
                    yield Stop(0)  # 立即结束对话，等待下一轮新消息触发
                    return

            # ── 处理 tool calls ──
            should_wait = False
            should_stop = False
            stop_minutes = 0.0

            for call in response.call_list or []:
                args = call.args if isinstance(call.args, dict) else {}
                reason = args.pop("reason", "未提供原因")
                logger.info(f"LLM 调用 {call.name}，原因: {reason}，参数: {args}")

                if call.name == _PASS_AND_WAIT:
                    # 特殊控制流：标记等待，不执行 action_manager
                    response.add_payload(
                        LLMPayload(
                            ROLE.TOOL_RESULT,
                            ToolResult(  # type: ignore[arg-type]
                                value="已跳过，等待用户新消息",
                                call_id=call.id,
                                name=call.name,
                            ),
                        )
                    )
                    should_wait = True

                elif call.name == _STOP_CONVERSATION:
                    # 特殊控制流：结束对话并设置冷却
                    stop_minutes = float(args.get("minutes", 5.0))
                    response.add_payload(
                        LLMPayload(
                            ROLE.TOOL_RESULT,
                            ToolResult(  # type: ignore[arg-type]
                                value=f"对话已结束，将在 {stop_minutes} 分钟后允许新对话",
                                call_id=call.id,
                                name=call.name,
                            ),
                        )
                    )
                    should_stop = True

                else:
                    # 普通 action/tool：通过 run_tool_call 执行
                    trigger_msg = unreads[-1] if unreads else None
                    await self.run_tool_call(call, response, usable_map, trigger_msg)
            # ── 处理控制流结果 ──
            if should_stop:
                # 设置冷却时间
                logger.info(f"对话已结束，冷却 {stop_minutes} 分钟")
                yield Stop(stop_minutes * 60)
                return

            if should_wait:
                # 等待新消息到来
                yield Wait()
                # 继续循环，让 LLM 基于更新后的上下文重新决策
                continue
            # 没有特殊控制流，继续让 LLM 决策（LLM 可能连续调用多轮工具）
            continue

    async def _execute_classical(
        self, chat_stream: Any
    ) -> AsyncGenerator[Wait | Success | Failure | Stop, None]:
        """classical 模式执行流程。"""
        try:
            # classical 模式每轮独立构建请求，仅在此处获取一次工具注册表
            _base_request = self.create_request("actor")
        except (ValueError, KeyError) as e:
            logger.error(f"获取模型配置失败: {e}")
            yield Failure(f"模型配置错误: {e}")
            return

        # 在外层预先获取工具注册表（每轮复用）
        usable_map = await self.inject_usables(_base_request)

        while True:
            formatted_text, unread_msgs = await self.fetch_unreads()
            unreads = unread_msgs

            if not formatted_text or not unread_msgs:
                yield Wait()
                continue

            classical_user_text = self._build_classical_user_text(
                chat_stream, unread_msgs
            )
            decision = await self.sub_agent(
                classical_user_text, [LLMPayload(ROLE.USER, Text(classical_user_text))]
            )
            logger.info(
                f"Sub-agent 决策: {decision['reason']} (响应: {decision['should_respond']})"
            )

            if not decision["should_respond"]:
                logger.info("Sub-agent 决定不响应，继续等待...")
                yield Wait()
                continue

            request = self.create_request("actor")
            request.add_payload(
                LLMPayload(ROLE.SYSTEM, Text(self._build_system_prompt(chat_stream)))
            )
            request.add_payload(LLMPayload(ROLE.USER, Text(classical_user_text)))
            if usable_map.get_all():
                request.add_payload(LLMPayload(ROLE.TOOL, usable_map.get_all()))  # type: ignore[arg-type]

            response = request

            while True:
                try:
                    response = await response.send(stream=False)
                    await response
                    await self.flush_unreads(unread_msgs)
                except Exception as e:
                    logger.error(f"LLM 请求失败: {e}", exc_info=True)
                    yield Failure("LLM 请求失败", e)
                    break

                if not response.call_list:
                    if response.message and response.message.strip():
                        logger.warning(
                            "LLM 返回了纯文本而非 tool call: "
                            f"{response.message[:100]}"
                        )
                    yield Stop(0)
                    return

                should_wait = False
                should_stop = False
                stop_minutes = 0.0
                sent_once = False

                for call in response.call_list or []:
                    args = call.args if isinstance(call.args, dict) else {}
                    reason = args.pop("reason", "未提供原因")
                    logger.info(f"LLM 调用 {call.name}，原因: {reason}，参数: {args}")

                    if call.name == _PASS_AND_WAIT:
                        response.add_payload(
                            LLMPayload(
                                ROLE.TOOL_RESULT,
                                ToolResult(  # type: ignore[arg-type]
                                    value="已跳过，等待用户新消息",
                                    call_id=call.id,
                                    name=call.name,
                                ),
                            )
                        )
                        should_wait = True

                    elif call.name == _STOP_CONVERSATION:
                        stop_minutes = float(args.get("minutes", 5.0))
                        response.add_payload(
                            LLMPayload(
                                ROLE.TOOL_RESULT,
                                ToolResult(  # type: ignore[arg-type]
                                    value=f"对话已结束，将在 {stop_minutes} 分钟后允许新对话",
                                    call_id=call.id,
                                    name=call.name,
                                ),
                            )
                        )
                        should_stop = True

                    else:
                        trigger_msg = unreads[-1] if unreads else None
                        _, success = await self.run_tool_call(call, response, usable_map, trigger_msg)
                        if success and call.name == _SEND_TEXT:
                            sent_once = True

                if sent_once:
                    logger.info("classical 模式已发送一次消息，强制结束当前对话")
                    yield Stop(0)
                    return

                if should_stop:
                    logger.info(f"对话已结束，冷却 {stop_minutes} 分钟")
                    yield Stop(stop_minutes * 60)
                    return

                if should_wait:
                    yield Wait()
                    break

                continue


# ─── Plugin ─────────────────────────────────────────────────


@register_plugin
class DefaultChatterPlugin(BasePlugin):
    """默认聊天插件"""

    plugin_name = "default_chatter"
    plugin_version = "1.0.0"
    plugin_author = "MoFox Team"
    plugin_description = "默认聊天组件，提供基础的消息处理和回复功能"
    configs = [DefaultChatterConfig]

    async def on_plugin_loaded(self) -> None:
        from src.core.prompt import optional, wrap, min_len

        config = get_core_config()
        personality = config.personality

        get_prompt_manager().get_or_create(
            name="default_chatter_system_prompt",
            template=system_prompt,
            policies={
                "nickname": optional(personality.nickname),
                "alias_names": optional("、".join(personality.alias_names)),
                "personality_core": optional(personality.personality_core),
                "personality_side": optional(personality.personality_side),
                "identity": optional(personality.identity),
                "background_story": optional(personality.background_story)
                .then(min_len(10))
                .then(
                    wrap(
                        "# 背景故事\\n"
                        "\\n- （以上为背景知识，请理解并作为行动依据，但不要在对话中直接复述。）"
                    )
                ),
                "reply_style": optional(personality.reply_style),
                "safety_guidelines": optional("\n".join(personality.safety_guidelines)),
                "current_time": optional(
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ),
            },
        )

        get_prompt_manager().get_or_create(
            name="default_chatter_sub_agent_prompt",
            template=sub_agent_system_prompt,
            policies={
                "nickname": optional(personality.nickname),
            },
        )

    def get_components(self) -> list[type]:
        """获取插件内所有组件类

        Returns:
            list[type]: 插件内所有组件类的列表
        """
        return [
            DefaultChatter,
            SendTextAction,
            PassAndWaitAction,
            StopConversationAction,
        ]

