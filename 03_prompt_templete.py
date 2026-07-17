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

def generate_response(subject, question):

    last_error = None

    for attempt in range(2):

        try:

            start = time.perf_counter()

            response = chain.invoke(
                {
                    "subject": subject,
                    "question": question
                }
            )

            end = time.perf_counter()

            return response, end - start

        except Exception as e:
            last_error = e

            if attempt == 0:
                print("Request failed. Retrying...")
    raise last_error

# Main Program

def main():

    print("*" * 60)
    print("Prompt Template Demo")
    print("*" * 60)
    print(f"Type 'help' to see all available commands.")

    while True:

        subject = input("Enter Subject: ").strip()

        if subject:
            break

        print("Subject cannot be empty.")

    while True:

        question = input(f"[{subject}]Question: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        command = question.lower()

        if command == "exit":
            print("Goodbye!")
            break

        elif command == "help":
            show_help()
            continue

        elif command == "change":

            while True:
                new_subject = input("New Subject: ").strip()

                if new_subject:
                    subject = new_subject
                    print(f"Subject changed to: '{subject}'")
                    break

            continue

        elif command == "history":
            show_history()
            continue

        elif command == "save":
            save_history()
            continue

        try:

            response, response_time = generate_response(
                subject,
                question
            )

            print("-" * 60)
            print("AI: ")
            print(response)
            print("-" * 60)
            print(f"Response Time: {response_time:.2f} seconds")

            chat_history.append(
                {
                    "subject": subject,
                    "question": question,
                    "response": response,
                    "response_time": response_time

                }
            )

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
