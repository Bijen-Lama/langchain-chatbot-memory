from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from configure import settings

MODEL_NAME = "llama-3.3-70-versatile"

llm = ChatGroq(
    model = MODEL_NAME,
    api_key = settings.GROQ_API_KEY.get_secret_value()
)



