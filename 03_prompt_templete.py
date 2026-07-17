from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from configure import settings

import json
import time

# Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
COMMANDS = {
    "help",
    "change",
    "history",
    "save",
    "exit",
}

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
            """
            You are an expert {subject} teacher.

            Your goal is to teach beginners.
            
            Rules:
            - Explain simply.
            - Use examples whenever possible.
            - Avoid unnecessary jargon.
            - Keep answers clear and beginner friendly.
            """
        ),
        (
            "human",
            "{question}"
        ),
    ]
)

# Chain
chain = prompt | llm | StrOutputParser()

# Chat History

chat_history = []

# Helper Function

def show_help():
    print("""
    Available Commands
    help Show Menu
    change Change the subject
    history Show previous questions
    save Save chat history
    exit Quit the application""")

def save_history():
    with open("chat_history.json", "w", encoding="utf-8") as file:
        json.dump(chat_history, file, indent=4, ensure_ascii=False)

    print("Chat history saved as 'chat_history.json'")

def show_history():

    if not chat_history:
        print("No conversation History.")
        return

    print("Conversation History")
    print("-" * 50)

    for i, item in enumerate(chat_history, start=1):
        print(f"{i}. [{item['subject']}]")
        print(f"Q: {item['question']}")
        print(f"A: {item['response'][:120]}...")
        print(f"Time: {item['response_time']:.2f} sec")
        print()



def main():
    print("*" * 60)
    print("Prompt Template Demo")
    print("*" * 60)
    print(f"Type '{EXIT_COMMAND}' to quit.")

    # Get Subject
    while True:
        subject = input("Enter Subject: ").strip()

        if subject:
            break

        print("Subject cannot be empty.")

    while True:
        question = input("Question: ").strip()

        if question.lower() == EXIT_COMMAND:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            response = chain.invoke(
                {
                    "subject": subject,
                    "question": question
                }
            )

            print("-" * 60)
            print("AI: ")
            print(response)
            print("-" * 60)

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
