import bs4
import os
from dotenv import load_dotenv
from typing_extensions import List, TypedDict

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, StateGraph

# --- 1. Environment Setup and API Key Loading ---
load_dotenv()

# Load Google Gemini API Key
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key is None:
    print("🚨 The API key for Google Gemini (GOOGLE_API_KEY) is not found in the .env file. Please set it.")
    exit()

# Configure LangSmith Tracing (Optional - set LANGCHAIN_TRACING_V2=false to disable)
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false") # Defaults to false
if os.getenv("LANGCHAIN_TRACING_V2") == "true":
    print("✨ LangSmith tracing is enabled. Ensure LANGCHAIN_API_KEY and LANGCHAIN_PROJECT are set in .env.")
    langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
    langchain_project = os.getenv("LANGCHAIN_PROJECT")
    if langchain_api_key is None or langchain_project is None:
        print("⚠️ Warning: LANGCHAIN_API_KEY or LANGCHAIN_PROJECT missing for LangSmith. Tracing might fail.")

# Set User-Agent for web requests
user_agent = os.getenv("USER_AGENT")
if user_agent is None:
    print("⚠️ USER_AGENT environment variable not set. Using a default user agent for web requests.")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# --- 2. LLM and Embeddings Definition ---
print("🚀 Initializing LLM (Gemini-Pro) and Embeddings (embedding-001)...")
llm = init_chat_model("gemini-1.0-pro", model_provider="google_genai", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = InMemoryVectorStore(embeddings)
print("✅ LLM and Embeddings initialized.")

# --- 3. Data Loading and Indexing for RAG (using Project Gutenberg) ---
print("📚 Loading document from Project Gutenberg (Pride and Prejudice)...")
# Using a specific book from Project Gutenberg: Pride and Prejudice (HTML version)
gutenberg_url = "https://www.gutenberg.org/files/1342/1342-h/1342-h.htm"

loader = WebBaseLoader(
    web_paths=(gutenberg_url,),
    requests_kwargs={"headers": {"User-Agent": user_agent}},
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            # Targeting common tags for book content on Gutenberg HTML pages
            name=['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'body', 'pre', 'div']
        )
    ),
)

# Consume the iterator returned by lazy_load()
try:
    docs = list(loader.lazy_load())
except Exception as e:
    print(f"❌ Error loading document from {gutenberg_url}: {e}")
    print("Please check your internet connection and if the URL is accessible.")
    exit()

if not docs or not docs[0].page_content.strip(): # Check if docs is empty or content is just whitespace
    print(f"🚨 No substantial content loaded from {gutenberg_url}.")
    print("This might mean the URL is incorrect, the page is empty, or the CSS selectors are too restrictive.")
    exit()

print(f"Document loaded. Total raw characters: {len(docs[0].page_content)}")

print("✂️ Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
add_splits = text_splitter.split_documents(docs)

if not add_splits:
    print("🚨 No document chunks were created after splitting. The content might be too small or the splitter settings are off.")
    exit()

print(f"Successfully created {len(add_splits)} document chunks.")

print("📊 Indexing chunks into vector store...")
_ = vector_store.add_documents(documents=add_splits)
print("✅ Document chunks indexed.")

# --- 4. Define the RAG Prompt ---
prompt = ChatPromptTemplate.from_template(
    """You are an AI assistant. Use the following context to answer the question. If you don't know the answer, state that you don't have enough information.

Context:
{context}

Question: {question}
"""
)

# --- 5. Define the LangGraph Application State and Nodes ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    print("🔍 Retrieving relevant documents from the vector store...")
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    print("🧠 Generating answer with LLM based on retrieved context...")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"answer": response.content}

# --- 6. Compile and Run LangGraph Application ---
print("⚙️ Compiling LangGraph application...")
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.set_finish_point("generate")
graph = graph_builder.compile()
print("✅ LangGraph application compiled. Ready to answer questions! 🎉")

# --- 7. Test the Application ---
question = "Who are the main characters in Pride and Prejudice and what are their personalities like?"
print(f"\n❓ Question: {question}")

response = graph.invoke({"question": question})

print("\n--- Final Answer ---")
print(response["answer"])
print("--- End of RAG Process ---")