from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage


@dataclass
class AgentEvent:
    event_type: str
    description: str
    tool: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    tool_call_id: str | None = None
    run_id: str | None = None
    namespace: tuple = ()


class LogParser:
    """
    Parser eventów LangGraph/Deep Agents z astream().

    Oczekiwany format:
        async for event in agent.astream(
            {"messages": prompt.messages},
            stream_mode=["tasks"],
            subgraphs=True,
        ):
            for parsed in parser.parse_chunk(event):
                logger.info(parsed)
    """

    def parse_chunk(self, chunk: Any) -> list[AgentEvent]:
        if not isinstance(chunk, tuple) or len(chunk) != 3:
            return []

        namespace, mode, task = chunk

        if mode != "tasks" or not isinstance(task, dict):
            return []

        namespace = tuple(namespace) if namespace else ()
        task_name = task.get("name")
        run_id = str(task["id"]) if task.get("id") else None

        output: list[AgentEvent] = []

        # Interesują nas wyłącznie eventy startowe lub końcowe,
        # zależnie od rodzaju noda.
        if task_name == "PatchToolCallsMiddleware.before_agent":
            output.extend(
                self._parse_human_message(task, namespace, run_id)
            )

        if task_name == "model":
            # Wynik modelu: wywołania narzędzi lub odpowiedź tekstowa.
            # Nie parsujemy input.messages, ponieważ to historia rozmowy
            # i może zawierać już wcześniej zalogowane tool_calls.
            if "result" in task:
                output.extend(
                    self._parse_model_result(task, namespace, run_id)
                )

        elif task_name == "tools":
            # Wynik narzędzia parsujemy wyłącznie z result.messages.
            if "result" in task:
                output.extend(
                    self._parse_tool_result(task, namespace, run_id)
                )

        return output

    def _parse_human_message(
        self,
        task: dict,
        namespace: tuple,
        run_id: str | None,
    ) -> list[AgentEvent]:
        # Middleware ma osobny event startu i zakończenia.
        # Event startowy zawiera input, końcowy result.
        if "input" not in task:
            return []

        output = []

        for msg in task.get("input", {}).get("messages", []):
            if not isinstance(msg, BaseMessage):
                continue

            if msg.type != "human":
                continue

            output.append(
                AgentEvent(
                    event_type="human_message",
                    description=msg.content or "",
                    run_id=run_id,
                    namespace=namespace,
                )
            )

        return output

    def _parse_model_result(
        self,
        task: dict,
        namespace: tuple,
        run_id: str | None,
    ) -> list[AgentEvent]:
        result = task.get("result") or {}
        output = []

        for msg in result.get("messages", []):
            if not isinstance(msg, BaseMessage):
                continue

            metadata = getattr(msg, "response_metadata", {}) or {}
            usage = metadata.get("token_usage", {}) or {}
            details = usage.get("completion_tokens_details", {}) or {}

            model_name = metadata.get("model_name")

            input_tokens = usage.get(
                "prompt_tokens",
                getattr(msg, "usage_metadata", {}).get("input_tokens", 0)
                if getattr(msg, "usage_metadata", None)
                else 0,
            )
            output_tokens = usage.get(
                "completion_tokens",
                getattr(msg, "usage_metadata", {}).get("output_tokens", 0)
                if getattr(msg, "usage_metadata", None)
                else 0,
            )
            total_tokens = usage.get(
                "total_tokens",
                getattr(msg, "usage_metadata", {}).get("total_tokens", 0)
                if getattr(msg, "usage_metadata", None)
                else 0,
            )
            reasoning_tokens = details.get(
                "reasoning_tokens",
                getattr(msg, "usage_metadata", {}).get(
                    "output_token_details", {}
                ).get("reasoning", 0)
                if getattr(msg, "usage_metadata", None)
                else 0,
            )

            # Model może zwrócić kilka wywołań narzędzi jednocześnie.
            for call in getattr(msg, "tool_calls", []) or []:
                args = call.get("args", {})
                args_str = ", ".join(f"{k}={v}" for k, v in args.items())

                tool_name = call.get("name", "")
                path = args.get("file_path")

                if path:
                    description = f"Wywołanie {path}"
                else:
                    description = f"Wywołanie narzędzia {tool_name} - {args_str}"

                output.append(
                    AgentEvent(
                        event_type="tool_call",
                        description=description,
                        tool=tool_name,
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        reasoning_tokens=reasoning_tokens,
                        tool_call_id=call.get("id"),
                        run_id=run_id,
                        namespace=namespace,
                    )
                )

            # Odpowiedź końcowa modelu, bez kolejnych tool_calls.
            if (
                msg.type == "ai"
                and msg.content
                and not getattr(msg, "tool_calls", [])
            ):
                output.append(
                    AgentEvent(
                        event_type="model_response",
                        description=msg.content,
                        model=model_name,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        reasoning_tokens=reasoning_tokens,
                        run_id=run_id,
                        namespace=namespace,
                    )
                )

        return output

    def _parse_tool_result(
        self,
        task: dict,
        namespace: tuple,
        run_id: str | None,
    ) -> list[AgentEvent]:
        result = task.get("result") or {}
        output = []

        for msg in result.get("messages", []):
            if not isinstance(msg, BaseMessage) or msg.type != "tool":
                continue

            output.append(
                AgentEvent(
                    event_type="tool_result",
                    description=msg.content or "",
                    tool=msg.name,
                    tool_call_id=getattr(msg, "tool_call_id", None),
                    run_id=run_id,
                    namespace=namespace,
                )
            )

        return output