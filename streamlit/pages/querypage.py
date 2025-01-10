import streamlit as st
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv
import os
import trafilatura
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
#from sentence_transformers import SentenceTransformer
from langchain import hub
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from tavily import TavilyClient

load_dotenv()

mistral_api_key = os.getenv('MISTRAL_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

tavily_client = TavilyClient(api_key=tavily_api_key)

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

# Helper function to initialize LLM
def initialize_llm(model_name: str):
    return OllamaLLM(model=model_name, base_url="http://ollama-conttainer:11434", verbose=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Helper function to handle sending prompts
def send_prompt(llm, prompt):
    response = llm.invoke(prompt)
    return response

def change_prompt():
    template = st.text_input('Paste your own template (optional)')
    prompt = PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template='', template_format='mustache')

    if len(template) != 0:
        prompt.template = template
    else:
        template = open("template.txt", "r").read()
        prompt.template = template

    return prompt

# Streamlit App
st.title("Chat with Ollama")

# Initialize session state for messages and selected model
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question!"}
    ]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "phi:latest"

# Model selection
available_models = ["phi:latest", "gemma2", "phi4"]  # Extend this list as needed
selected_model = st.selectbox("Choose a model:", available_models, index=0)

# Update the model in session state if it changes
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.messages.append(
        {"role": "assistant", "content": f"Switched to model: {selected_model}"}
    )

# Initialize the LLM based on the selected model
model = initialize_llm(st.session_state.selected_model)

vectorstore = FAISS.load_local('faiss_storage_1500_300',
                                embeddings=MistralAIEmbeddings(model="mistral-embed", api_key=mistral_api_key),
                                allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_type='similarity_score_threshold',
                                     search_kwargs={'k': 20, 'score_threshold': .48})

prompt = change_prompt()

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} |
    prompt |
    model |
    StrOutputParser()
)

# Integrate Tavily Search
search_service = SearchService()
user_query = st.text_input("Enter your query for Tavily search:")
if st.button("Search Tavily"):
    if user_query:
        with st.spinner("Fetching sources..."):
            search_results = search_service.web_search(user_query)
            if search_results:
                st.write("### Tavily Search Results")
                for result in search_results:
                    st.write(f"**Title:** {result.metadata['title']}")
                    st.write(f"**URL:** {result.metadata['url']}")
                    st.write(f"**Content Preview:** {result.page_content[:200]}...")
                    st.write("---")
                with st.spinner("Indexing results..."):
                    _ = vectorstore.add_documents(search_results)
                    st.success("Sources indexed successfully!")
            else:
                st.error("No results found.")

# Input for user prompt
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Generate a response if the last message is from the user
if st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = send_prompt(rag_chain, st.session_state.messages[-1]["content"])
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")
