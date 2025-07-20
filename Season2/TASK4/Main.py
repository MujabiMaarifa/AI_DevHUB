import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["USER_AGENT"] = "MyNLPGeeksforgeeksRAG/1.0 (daudimujabi@gmail.com)"
#load the hidden api key from the .env
api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
   print("The api key for google germini is not found")
   exit()

# #define and load the llm
# llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
#define embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

#select and define the vector store
vector_store = InMemoryVectorStore(embeddings)

#define the document loader
loader = WebBaseLoader(
    web_paths = ("https://www.geeksforgeeks.org/nlp/nlp-custom-corpus/",),
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(
            class_=("article-page_flex", "main_wrapper", "text","post-content", "post-title", "post-header", "post-template-default") # the problem comes with the web base loader 
        )
    ),
)

#load the document using the loader extension
docs = loader.lazy_load()

#split the texts using the recursive text splitter character for easy and clear access of content of the defined document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 2000, chunk_overlap = 200
)
add_splits = text_splitter.split_documents(docs)

#index chunks
_= vector_store.add_documents(
    documents = add_splits
)

#define the prompt
prompt = hub.pull("rlm/rag-prompt")

# define the state for the application
class State(TypedDict) :
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question" : state["question"], "context" : docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

#compile app application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

print("\n✨ EASY NLP and Corpus✨")
print("Type your question below. Type 'exit' or 'quit' to end the conversation.\n")

if __name__ == "__main__":
    while True:
        user_input = input("You🔍: ").strip()
        greetings = {"hi", "hello", "hey", "how are you", "good morning", "good evening", "bonjour"}

        if user_input.lower() in greetings:
            print("Assistant🧠: Hello! How can I help you with NLP today?\n")
            continue
            
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Assistant🧠: Goodbye! Feel free to reach out for any queries...")
            break

        response = graph.invoke({"question": user_input})
        print(f"Assistant🧠: {response['answer']}\n")

