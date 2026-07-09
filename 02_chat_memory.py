from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from configure import settings

# Configuration
MODEL_NAME = "llama-3.3-70b-versatile"

# Initialize LLM
llm = ChatGroq(
    model=MODEL_NAME,
    api_key=settings.GROQ_API_KEY.get_secret_value()
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are AI assistant. Answer accurately and precisely.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# Create Chain
chain = prompt | llm | StrOutputParser()

# Memory Store
store = {}

def get_session_history(session_id: str):
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

# Chat Loop
print("=" * 50)
print("LangChain Chat Memory")
print("=" * 50)

# Define session_id
session_id = input("Enter Session ID: ").strip()

if not session_id:
    session_id = "default"

print(f"Using session: {session_id}\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    response = chatbot.invoke(
        {"question": question},
        config={
            "configurable": {
                "session_id": session_id
            }
        },
    )

    print(f"Bot: {response}\n")
