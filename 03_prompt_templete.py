from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from configure import settings

# Configuration
MODEL_NAME = "llama-3.3-70b-versatile"
EXIT_COMMAND = "exit"

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
            You are expert {subject} teacher.
    
            Your goal is to teach beginners.
                - Explain concepts in simple language.
                - Use examples whenever possible.
                - Keep answers clear and easy to understand.
                - Avoid unnecessary jargon.
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
