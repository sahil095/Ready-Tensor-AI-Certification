from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

load_dotenv()


def get_llm(model_name: str, temperature: float = 0.7) -> BaseChatModel:
    if model_name == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
    elif model_name == "openai/gpt-oss-20b":
        return ChatGroq(model="openai/gpt-oss-20b", temperature=temperature)
    else:
        raise ValueError(f"Unknown model name: {model_name}")