from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

#load the hidden api key from the .env
api_key = os.getenv("CHAT_API_KEY")  

if api_key is None:
    raise ValueError("CHAT_API_KEY is not set in the .env file")

#define and load the llm
llm = init_chat_model("gpt-4o-2024-08-06", model_provider="openai")

#define embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#select and define the vector store
vector_store = InMemoryVectorStore(embeddings)

#indexing and loading the dataset for the RAG
#load the document
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://www.corpusdata.org/wikipedia.asp",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")