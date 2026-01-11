import os
import requests
from typing import List, TypedDict
from dotenv import load_dotenv

# LangChain/LangGraph imports
try:
    from langchain_community.document_loaders import TextLoader
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Weaviate
    from langchain.text_splitter import CharacterTextSplitter
    from langgraph.graph import StateGraph, END
    import weaviate
    from weaviate.embedded import EmbeddedOptions
except ImportError:
    print("Error: Required LangChain/LangGraph/Weaviate components not found.")
    TextLoader = Document = ChatPromptTemplate = StrOutputParser = GoogleGenerativeAIEmbeddings = ChatGoogleGenerativeAI = CharacterTextSplitter = StateGraph = END = weaviate = EmbeddedOptions = None

# Load environment variables
load_dotenv()

# --- 1. Data Preparation ---
def prepare_vectorstore():
    if not weaviate:
        return None

    # Download sample data
    url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/how_to/state_of_the_union.txt"
    local_file = "state_of_the_union.txt"
    if not os.path.exists(local_file):
        print("Downloading sample data...")
        res = requests.get(url)
        with open(local_file, "w") as f:
            f.write(res.text)

    loader = TextLoader(local_file)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Note: Weaviate Embedded is useful for demos but requires specific setup
    try:
        client = weaviate.Client(embedded_options=EmbeddedOptions())
        vectorstore = Weaviate.from_documents(
            client=client,
            documents=chunks,
            embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            by_text=False
        )
        return vectorstore.as_retriever()
    except Exception as e:
        print(f"Vectorstore setup failed: {e}")
        return None

# --- 2. Graph Definition ---
class RAGGraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str

def retrieve_documents(state: RAGGraphState, retriever) -> RAGGraphState:
    print(f"--- RETRIEVING for: {state['question']} ---")
    docs = retriever.invoke(state["question"])
    return {"documents": docs, "question": state["question"], "generation": ""}

def generate_response(state: RAGGraphState, llm) -> RAGGraphState:
    print("--- GENERATING ---")
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "question": state["question"]})
    return {"question": state["question"], "documents": state["documents"], "generation": generation}

def build_rag_graph(retriever, llm):
    workflow = StateGraph(RAGGraphState)
    
    # Use lambda to pass the retriever/llm to nodes
    workflow.add_node("retrieve", lambda state: retrieve_documents(state, retriever))
    workflow.add_node("generate", lambda state: generate_response(state, llm))
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def setup_and_run_rag():
    if not all([TextLoader, StateGraph]):
        return

    retriever = prepare_vectorstore()
    if not retriever:
        print("Retriever not available. Ensure GOOGLE_API_KEY and Weaviate dependencies are correct.")
        return

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    app = build_rag_graph(retriever, llm)

    query = "What did the president say about Justice Breyer"
    print(f"\nQuery: {query}")
    result = app.invoke({"question": query})
    print(f"\nFinal Response:\n{result['generation']}")

if __name__ == "__main__":
    setup_and_run_rag()
