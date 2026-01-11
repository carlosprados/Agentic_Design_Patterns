import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.memory import InMemoryMemoryService, VertexAiRagMemoryService
except ImportError:
    print("Error: Google ADK memory components not found.")
    InMemoryMemoryService = VertexAiRagMemoryService = None

# Load environment variables
load_dotenv()

def demo_memory_services():
    """
    Demonstrates the initialization of different memory services in ADK.
    """
    print("--- ADK Memory Services Demo ---")

    # 1. InMemoryMemoryService - Volatile, for dev/test
    if InMemoryMemoryService:
        in_memory = InMemoryMemoryService()
        print("Initialized InMemoryMemoryService.")

    # 2. VertexAiRagMemoryService - Persistent, for production
    if VertexAiRagMemoryService:
        # Example configuration
        RAG_CORPUS_RESOURCE_NAME = os.getenv("RAG_CORPUS_RESOURCE_NAME", "projects/your-project/locations/us-central1/ragCorpora/your-corpus")
        
        try:
            vertex_memory = VertexAiRagMemoryService(
                rag_corpus=RAG_CORPUS_RESOURCE_NAME,
                similarity_top_k=5,
                vector_distance_threshold=0.7
            )
            print(f"Initialized VertexAiRagMemoryService with corpus: {RAG_CORPUS_RESOURCE_NAME}")
        except Exception as e:
            print(f"VertexAI Memory Service initialization skipped/failed: {e}")

if __name__ == "__main__":
    demo_memory_services()
