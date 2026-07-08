from dotenv import load_dotenv
import time

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from config import settings

# Load Environment Variables
load_dotenv()

API_KEY = settings.GROQ_API_KEY.get_secret_value()

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file.")

# Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
MAX_MESSAGES = 20

# Initialize Model
chat_model = ChatGroq(
    model=MODEL_NAME,
    api_key=API_KEY
)

# System Prompt
system_message = SystemMessage(
    content=(
        "You are a helpful, friendly and knowledgeable AI assistant."
        "Provide clear, concise, accurate and easy to understand answers."
    )
)

# Store Conversation
chat_history = [system_message]

# Helper Functions
def print_help():
    """Display available commands."""
    print("Available commands")
    print("****************************")
    print("/help Show all commands")
    print("/clear Clear chat history")
    print("/history Show chat history")
    print("/exit Exit chatbot")
    print("*****************************")

def show_history():
    """Display conversation history"""
    print("*****CHAT HISTORY*****")

    for message in chat_history:
        if isinstance(message, SystemMessage):
            continue
        if isinstance(message, HumanMessage):
            print(f"You: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"Bot: {message.content}")
    print("********************************")

def clear_history():
    """Reset Conversation"""
    global chat_history
    chat_history = [system_message]
    print("Chat history cleared")

def trim_history():
    """ Keep only the latest conversation while preserving the system prompt"""
    global chat_history

    if len(chat_history) > MAX_MESSAGES:
        chat_history = [system_message] + chat_history[-(MAX_MESSAGES - 1):]

def save_conversation(user_input, bot_response):
    """Save conversation to a text file"""

    with open("chat_history.txt", "a", encoding="utf-8") as file:
        file.write(f"You : {user_input}\n")
        file.write(f"Bot : {bot_response}\n")
        file.write("*" * 50 + "\n")

def chat(user_input):
    """Send message to LLM and return response"""

    global chat_history

    chat_history.append(HumanMessage(content=user_input))

    trim_history()

    print("Bot is thinking...\n")

    start_time = time.time()

    response_text = ""

    try:
        for chunk in chat_model.stream(chat_history):
            print(chunk.content, end="", flush=True)
            response_text += chunk.content

        print()

    except Exception as e:
        print(f"Error: {e}")
        return

    end_time = time.time()

    chat_history.append(AIMessage(content=response_text))

    save_conversation(user_input, response_text)

    print(f"Response Time: {end_time - start_time:.2f} seconds\n")

# Main Application
print("*" * 50)
print("LangChain + groq Chatbot")
print("*" * 50)
print("Type '/help' to see available commands.\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    if user_input.lower() in ["/exit", "exit"]:
        print("Goodbye!")
        break

    elif user_input.lower() == "/help":
        print_help()

    elif user_input.lower() == "/clear":
        clear_history()

    elif user_input.lower() == "/history":
        show_history()
    else:
        chat(user_input)



