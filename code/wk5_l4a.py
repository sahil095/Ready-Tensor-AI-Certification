from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

# Set up your LLM - the brain of your agent
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# Define your agent's state - this is your agent's memory
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create your tools - your agent's capabilities
def get_tools():
    return [
        TavilySearchResults(max_results=3, search_depth="advanced")
    ]


# The LLM node - where your agent thinks and decides
def llm_node(state: State):
    """Your agent's brain - decides whether to use tools or respond."""
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)  # Give your agent access to tools
    
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# The tools node - where your agent takes action
def tools_node(state: State):
    """Your agent's hands - executes the chosen tools."""
    tools = get_tools()
    tool_registry = {tool.name: tool for tool in tools}
    
    last_message = state["messages"][-1]
    tool_messages = []
    
    # Execute each tool the agent requested
    for tool_call in last_message.tool_calls:
        tool = tool_registry[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        
        # Send the result back to the agent
        tool_messages.append(ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"]
        ))
    
    return {"messages": tool_messages}

# Decision function - should we use tools or finish?
def should_continue(state: State):
    """Decides whether to use tools or provide final answer."""
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"  # Agent wants to use tools
    return END  # Agent is ready to respond



# Build the complete workflow
def create_agent():
    graph = StateGraph(State)
    
    # Add the nodes
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)
    
    # Set the starting point
    graph.set_entry_point("llm")
    
    # Add the flow logic
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")  # After using tools, go back to thinking
    
    return graph.compile()


# Create and use your enhanced agent
agent = create_agent()

# Test it out!
initial_state = {
    "messages": [
        SystemMessage(content="You are a helpful assistant with access to web search. Use the search tool when you need current information."),
        HumanMessage(content="What's the latest news about AI developments in 2025?")
    ]
}

result = agent.invoke(initial_state)
print(result["messages"][-1].content)