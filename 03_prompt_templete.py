from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
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
            "You are expert {subject} teacher. "
            "Explain concepts in a simple and begineer friendly way."
        ),
        (
            "human",
            "{question}"
        ),
    ]
)

# Chain
chain = prompt | llm | StrOutputParser()

print("Prompt Template Demo")

subject = input("Enter Subject: ")

while True:
    question = input("Question: ")

    if question.lower() == "exit":
        break

    response = chain.invoke(
        {
            "subject": subject,
            "question": question
        }
    )

    print(f"AI: {response}")