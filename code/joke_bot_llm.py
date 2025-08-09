from typing import Annotated, List, Literal, Optional
from operator import add
from pydantic import BaseModel
from pyjokes import get_joke
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm_writer = ChatGroq(model="openai/gpt-oss-20b", temperature=0.7)
llm_critic = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)


class Joke(BaseModel):
    text: str
    category: str

class JokeState(BaseModel):
    # existing
    jokes: Annotated[List[Joke], add] = []
    jokes_choice: Literal["n", "c", "q"] = "n"  # next joke, change category, or quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False

    # new for writer–critic loop
    latest_joke: Optional[str] = None
    approved: bool = False
    retries: int = 0
    max_retries: int = 5


def show_menu(state: JokeState) -> dict:
    user_input = input("[n] Next  [c] Category  [q] Quit\n> ").strip().lower()
    return {"jokes_choice": user_input}


def writer(state: JokeState) -> dict:
    """
    Uses Groq LLM to generate a developer-themed joke based on category & language.
    """
    prompt_text = build_prompt("writer", state.category, state.language)
    # LangChain ChatGroq returns an AIMessage
    response = llm_writer.invoke(prompt_text)
    joke_text = (response.content or "").strip()
    return {"latest_joke": joke_text}


def critic(state: JokeState) -> dict:
    """
    Uses Groq LLM to judge the latest_joke. Sets approved True/False.
    Must return ONLY 'APPROVE' or 'REJECT'.
    """
    if not state.latest_joke:
        return {"approved": False}

    critic_prompt = (
        build_prompt("critic", state.category, state.language)
        + "\n\nJOKE:\n"
        + state.latest_joke
        + "\n\nAnswer with APPROVE or REJECT only."
    )
    response = llm_critic.invoke(critic_prompt)
    verdict = (response.content or "").strip().upper()
    # Be defensive: normalize any extra text
    if "APPROVE" in verdict and "REJECT" not in verdict:
        return {"approved": True}
    if "REJECT" in verdict and "APPROVE" not in verdict:
        return {"approved": False}
    # Fallback: if unclear, treat as reject
    return {"approved": False}

def retry_writer(state: JokeState) -> dict:
    # increment retry counter before looping back to writer
    return {"retries": state.retries + 1}

def show_final_joke(state: JokeState) -> dict:
    """
    Prints the approved joke (or fallback after max retries), appends to history,
    then resets evaluation fields.
    """
    if state.approved and state.latest_joke:
        final = state.latest_joke
    else:
        final = state.latest_joke or "Couldn't craft a good one right now—try again!"
    new_joke = Joke(text=final, category=state.category)
    print(new_joke)

    # Reset evaluation state for next cycle
    return {
        "jokes": [new_joke],
        "latest_joke": None,
        "approved": False,
        "retries": 0
    }


def fetch_joke(state: JokeState) -> dict:
    joke_text = get_joke(language=state.language, category=state.category)
    new_joke = Joke(text=joke_text, category=state.category)
    print(new_joke)
    return {"jokes": [new_joke]}


def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    selection = int(input("Select category [0=neutral, 1=chuck, 2=all]: ").strip())
    return {
        "category": categories[selection],
        "latest_joke": None,
        "approved": False,
        "retries": 0
    }


def exit_bot(state: JokeState) -> dict:
    return {"quit": True}


def route_writer_to_critic(state: JokeState) -> str:
    # Always go to critic after generating a joke
    return "critic"

def route_critic_next(state: JokeState) -> str:
    """
    If approved -> show_final_joke.
    If rejected and retries < max -> loop back to writer and increment retry counter.
    If rejected and at cap -> show_final_joke (fallback).
    """
    if state.approved:
        return "show_final_joke"
    if state.retries + 1 < state.max_retries:
        # bump retry count and try again
        state.retries += 1
        return "writer"
    # hit cap
    return "show_final_joke"


def route_choice(state: JokeState) -> str:
    if state.jokes_choice == "n":
        # now routes to writer (not direct joke function)
        return "writer"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"



def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    # existing nodes
    workflow.add_node("show_menu", show_menu)
    workflow.add_node("update_category", update_category)
    workflow.add_node("exit_bot", exit_bot)

    # new nodes
    workflow.add_node("writer", writer)
    workflow.add_node("critic", critic)
    workflow.add_node("show_final_joke", show_final_joke)

    workflow.set_entry_point("show_menu")

    # menu -> routes
    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "writer": "writer",
            "update_category": "update_category",
            "exit_bot": "exit_bot",
        }
    )

    # writer -> critic (always)
    workflow.add_conditional_edges(
        "writer",
        route_writer_to_critic,
        {"critic": "critic"}
    )

    # critic -> writer (retry) or -> show_final_joke (approved or cap)
    workflow.add_conditional_edges(
        "critic",
        route_critic_next,
        {
            "writer": "writer",
            "show_final_joke": "show_final_joke"
        }
    )

    # after showing a final joke, go back to menu
    workflow.add_edge("show_final_joke", "show_menu")

    # category change returns to menu
    workflow.add_edge("update_category", "show_menu")

    # exit
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


def build_prompt(role: str, category: str, language: str) -> str:
    if role == "writer":
        return (
            f"Write a short, clean developer-themed joke in {language}. "
            f"Category: {category}. Make it original, one or two lines max. "
            "Avoid profanity or offensive content."
        )
    if role == "critic":
        return (
            "You are a strict comedy critic for developer-themed jokes.\n"
            "Evaluate for clarity, originality, and being family-friendly.\n"
            "Reply with EXACTLY one word: APPROVE or REJECT."
        )
    return ""


def main():
    graph = build_joke_graph()
    final_state = graph.invoke(JokeState(), config={"recursion_limit": 200})
    # print(final_state)


if __name__ == "__main__":
    main()