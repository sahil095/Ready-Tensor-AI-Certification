from typing import Annotated, List, Literal
from operator import add
from pydantic import BaseModel
from pyjokes import get_joke
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph



# ===================
# Define State
# ===================

class Joke(BaseModel):
    text: str
    category: str

class JokeState(BaseModel):
    """
    Represents the evolving state of the joke bot.
    """
    jokes: Annotated[List[Joke], add] = []
    jokes_choice: Literal["n", "c", "q"] = "n" # next joke, change category, or quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False


# ===================
# Utilities
# ===================

def get_user_input(prompt: str) -> str:
    return input(prompt).strip().lower()

def print_joke(joke: Joke):
    """Print a joke with nice formatting."""
    # print(f"\n📂 CATEGORY: {joke.category.upper()}\n")
    print(f"\n😂 {joke.text}\n")
    print("=" * 60)

def print_menu_header(category: str, total_jokes: int):
    """Print a compact menu header."""
    print(f"🎭 Menu | Category: {category.upper()} | Jokes: {total_jokes}")
    print("-" * 50)

def print_category_menu():
    """Print a nicely formatted category selection menu."""
    print("📂" + "=" * 58 + "📂")
    print("    CATEGORY SELECTION")
    print("=" * 60)


# ===================
# Define Nodes
# ===================

def show_menu(state: JokeState) -> dict:
    user_input = input("[n] Next  [c] Category  [q] Quit\n> ").strip().lower()
    return {"jokes_choice": user_input}


def fetch_joke(state: JokeState) -> dict:
    joke_text = get_joke(language=state.language, category=state.category)
    new_joke = Joke(text=joke_text, category=state.category)
    print(new_joke)
    return {"jokes": [new_joke]}


def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    selection = int(input("Select category [0=neutral, 1=chuck, 2=all]: ").strip())
    return {"category": categories[selection]}


def exit_bot(state: JokeState) -> dict:
    return {"quit": True}



def route_choice(state: JokeState) -> str:
    """
    Router function to determine the next node based on user choice.
    Keys must match the target node names.
    """
    if state.jokes_choice == "n":
        return "fetch_joke"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"



# ===================
# Build Graph
# ===================

def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("fetch_joke", fetch_joke)
    workflow.add_node("update_category", update_category)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "fetch_joke": "fetch_joke",
            "update_category": "update_category",
            "exit_bot": "exit_bot",
        }
    )

    workflow.add_edge("fetch_joke", "show_menu")
    workflow.add_edge("update_category", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()


# ===================
# Main Entry
# ===================

def main():
    graph = build_joke_graph()
    final_state = graph.invoke(JokeState(), config={"recursion_limit": 100})
    # print(final_state)


main()