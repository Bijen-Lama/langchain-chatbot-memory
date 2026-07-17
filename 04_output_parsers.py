from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from configure import settings

# ==========================
# Configuration
# ==========================

MODEL_NAME = "llama-3.3-70b-versatile"

# ==========================
# Initialize LLM
# ==========================

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=settings.GROQ_API_KEY.get_secret_value()
)

# ==========================
# Prompt Template
# ==========================

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant."
        ),
        (
            "human",
            "{question}"
        )
    ]
)

# ==========================
# Output Parser
# ==========================

parser = StrOutputParser()

# ==========================
# Create Chain
# ==========================

chain = prompt | llm | parser

# ==========================
# Chat Loop
# ==========================

print("=" * 50)
print("Output Parser Demo")
print("=" * 50)

while True:

    question = input("\nYou: ")

    if question.lower() == "exit":
        break

    response = chain.invoke(
        {
            "question": question
        }
    )

    print("\nType:", type(response))
    print("\nResponse:")
    print(response)