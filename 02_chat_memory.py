from typing import Dict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from configure import settings

# Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
DEFAULT_SESSION = "default"

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Guidelines:
- Answer accurately and clearly.
- Be concise unless more detail is requested.
- If you don't know something, say so.
- Do not make up facts."""

# Initialize LLM
llm = ChatGroq(
    model=MODEL_NAME,
    api_key=settings.GROQ_API_KEY.get_secret_value()
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Create Chain
chain = prompt | llm | StrOutputParser()

# Memory Store
store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

# Add Memory
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# Helper functions
def print_help() -> None:
    print(
        """
        Available Commands
        /help show commands
        /history Show conversation history
        /clear Clear current session memory
        /session Show current session
        /exit Exit chatbot"""
    )


# Chat Loop
print("=" * 50)
print("LangChain Chat Memory")
print("=" * 50)

# Define session_id
session_id = input("Enter Session ID (blank = default): ").strip()

if not session_id:
    session_id = DEFAULT_SESSION

print(f"Using session: {session_id}\n")
print("Type /help to see available commands.")

while True:

    question = input("You: ").strip()

    if not question:
        continue

    command = question.lower()

    if question in ("/exit", "exit"):
        print("Goodbye!")
        break
    elif command == "/help":
        print_help()
        continue
    elif command == "/session":
        print(f"Current Session: {session_id}")
        continue
    elif command == "/clear":
        store[session_id] = InMemoryChatMessageHistory()
        print("Memory cleared.")
        continue
    elif command == "/history":
        history = get_session_history(session_id)

        if not history.messages:
            print("No conversation history.")
            continue

        print("Conversation History")
        print("*" * 50)

        for message in history.messages:
            role = message.type.capitalize()
            print(f"{role}: {message.content}")

        print("-" * 50)
        continue

    try:
        response = chatbot.invoke(
            {"question": question},
            config={
                "configurable": {
                    "session_id": session_id
                }
            },
        )

        print(f"Bot: {response}")

    except KeyboardInterrupt:
        print("Inturrepted. Exiting...")
        break

    except Exception as e:
        print(f"Error: {e}")
