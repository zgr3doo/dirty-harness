import operator
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

class Agent:

    def __init__(self, model, tools, system_prompt="", checkpointer=None):
        self.system = system_prompt

        # interrupt_before = ["action"]
        interrupt_before = []

        checkpointer = InMemorySaver()

        graph = StateGraph(State)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=interrupt_before
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: State):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: State):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        message = self.model.invoke(messages)
        extracted_message = getattr(message, "content", "")
        response_metadata = getattr(message, "response_metadata", "")
        token_usage = response_metadata.get("token_usage", {}).get("total_tokens", "")
        if extracted_message:
            print(f"Inferring model: [\n {extracted_message.strip()}\n] -- Tokens used:{token_usage}")
        else:
            print(f"Inferring model: + -- Tokens used:{token_usage}")
        return {'messages': [message]}

    def take_action(self, state: State):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:

            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

            extracted_args = t.get("args", {}).copy()
            extracted_args.pop("type", None)
            extracted_args.pop("content", None)
            tool_error = "Error:" in str(result)
            print(f"Calling: {t['name']} - {extracted_args} - Result: {"Error" if tool_error else "OK"}")

        return {'messages': results}