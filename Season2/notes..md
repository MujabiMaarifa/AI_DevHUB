#load the .env
# select the model to work with
**pip install -qU "langchain[mistralai]"**


# load the api keys
# select the model to work with and load the llm

# add the embeddings
***define and load the llm
llm = init_chat_model("gpt-4o-2024-08-06", model_provider="openai")
#define embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
***
# select vector store of the model to use
pip install -qU langchain-core