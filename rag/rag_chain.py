from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.http import models
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.pydantic_v1 import BaseModel
# Python
from dotenv import load_dotenv
import os

# 通過 .env 讀取敏感資料
load_dotenv()  # ./env
api_key = os.getenv('API_KEY')
embedd_azure_deployment = os.getenv('EMBEDD_DEPLOYMENT_MODEL')
api_version = os.getenv('API_VERSION')
azure_endpoint = os.getenv('AZURE_ENPOINT')
file_path = os.getenv('FILE_PATH')
openai_model = os.getenv('OPENAI_MODEL')
embedding_model = AzureOpenAIEmbeddings(
    api_key=api_key,
    azure_deployment=embedd_azure_deployment,
    openai_api_version=api_version,
    azure_endpoint=azure_endpoint,
)

client = QdrantClient()
collection_name = "legalassistant"

qdrant = Qdrant(
    client,
    collection_name,
    embedding_model
)

retrieval = qdrant.as_retriever(search_kwargs={"k": 3})
# 呼叫 Azure 部署的 GPT-4 model
model = AzureChatOpenAI(
    api_key=api_key,
    openai_api_version=api_version,
    azure_deployment=openai_model,
    azure_endpoint=azure_endpoint,
    temperature=0,
    streaming=True
)
prompt = ChatPromptTemplate.from_template(
    """你是一位台灣的資深法律顧問，請回答依照 context 裡的資訊，使用臺灣繁體中文來回答問題:
<context>
{context}
</context>
Question: {input}
附帶說明是引用哪一條法規，並給出解決方案。
"""

)

documents = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retrieval, documents)
# Add typing for input


class Question(BaseModel):
    input: str


rag_chain = retrieval_chain.with_types(input_type=Question)
