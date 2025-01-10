from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
import sys
import trafilatura
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from tavily import TavilyClient
from pyngrok import ngrok
import uvicorn

# Load environment variables
load_dotenv()

# API keys
mistral_api_key = os.getenv('MISTRAL_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')
ngrok_auth_token = os.getenv('NGROK_AUTH_TOKEN')

# Initialize FastAPI
app = FastAPI(title="Playdeas: RAG game recommender")

# Tavily Client Initialization
tavily_client = TavilyClient(api_key=tavily_api_key)

# Search service for Tavily integration
class SearchService:
    def web_search(self, query: str):
        try:
            results = []
            response = tavily_client.search(query, max_results=3)
            search_results = response.get("results", [])

            for result in search_results:
                downloaded = trafilatura.fetch_url(result.get("url"))
                content = trafilatura.extract(downloaded, include_comments=False)

                results.append(
                    Document(
                        page_content=content or "",
                        metadata={"title": result.get("title", ""), "url": result.get("url", "")}
                    )
                )

            return results
        except Exception as e:
            print(e)
            return []

# Helper functions
def initialize_llm(model_name: str):
    return OllamaLLM(model=model_name, base_url="http://ollama-container:11434", verbose=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Prompt Initialization
def change_prompt(template_file: str = "template.txt"):
    with open(template_file, "r") as file:
        template = file.read()
    os.write(sys.stdout.fileno(), f"Template: {template}\n".encode())
    return PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template_format='mustache', template=template)

# FastAPI Models
class ChatRequest(BaseModel):
    model: str
    question: str

class SearchRequest(BaseModel):
    query: str

class SearchResult(BaseModel):
    title: str
    url: str
    content_preview: str

# Initialize Vector Store and Retriever
vectorstore = FAISS.load_local('faiss_storage_1500_300',
                                embeddings=MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key),
                                allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_type='similarity_score_threshold',
                                     search_kwargs={'k': 20, 'score_threshold': .48})

# Initialize prompt and search service
prompt_template = change_prompt()
search_service = SearchService()

# Main API Routes
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Initialize the LLM based on the selected model
        os.write(sys.stdout.fileno(), f"Request: {request}\n".encode())
        llm = initialize_llm(request.model)

        # Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()} |
            prompt_template |
            llm |
            StrOutputParser()
        )

        # Generate response
        response = rag_chain.invoke(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

@app.post("/search", response_model=list[SearchResult])
async def search(request: SearchRequest):
    try:
        # Perform Tavily search
        search_results = search_service.web_search(request.query)
        if search_results:
            # Add results to vectorstore
            _ = vectorstore.add_documents(search_results)

            # Format results
            formatted_results = [
                SearchResult(
                    title=result.metadata['title'],
                    url=result.metadata['url'],
                    content_preview=result.page_content[:200]
                )
                for result in search_results
            ]
            return formatted_results
        else:
            return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing search: {e}")


@app.on_event("startup")
async def start_ngrok():
    # Check for existing tunnels to avoid creating multiple agents
    try:
        ngrok.set_auth_token(ngrok_auth_token)
        existing_tunnels = ngrok.get_tunnels()
        
        if not existing_tunnels:
            # Start a new tunnel only if no active tunnels are found
            public_url = ngrok.connect(8000, bind_tls=True)
            os.write(sys.stdout.fileno(), f"App started on URL: {public_url}\n".encode())
        else:
            public_url = existing_tunnels[0].public_url
            os.write(sys.stdout.fileno(), f"Reusing existing tunnel: {public_url}\n".encode())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing ngrok tunnel: {e}")



#use if want to test locally
'''
if __name__ == "__main__":
    # Start a tunnel to the FastAPI app
    public_url = ngrok.connect(8000, bind_tls=True)  # 8000 is the default port for FastAPI
    print(f"Ngrok tunnel: {public_url}")
    
    # Start the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''