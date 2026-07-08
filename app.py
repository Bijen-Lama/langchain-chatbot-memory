from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

chat_model = ChatGroq(model="llama-3.3-70b-versatile")

chat_history = []

print("Type 'exit' to quit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    response = chat_model.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    print("Bot: ", response.content)



