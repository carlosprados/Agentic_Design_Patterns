import os
from dotenv import load_dotenv

# Google ADK imports
try:
    from google.adk.sessions import InMemorySessionService, DatabaseSessionService, VertexAiSessionService
except ImportError:
    print("Error: Google ADK session components not found.")
    InMemorySessionService = DatabaseSessionService = VertexAiSessionService = None

# Load environment variables
load_dotenv()

def demo_session_services():
    """
    Demonstrates the initialization of different session management services in ADK.
    """
    print("--- ADK Session Services Demo ---")

    # 1. InMemorySessionService - Default, non-persistent
    if InMemorySessionService:
        in_memory = InMemorySessionService()
        print("Initialized InMemorySessionService.")

    # 2. DatabaseSessionService - Persistent via SQLAlchemy
    if DatabaseSessionService:
        db_url = os.getenv("DATABASE_URL", "sqlite:///./my_agent_data.db")
        try:
            db_session = DatabaseSessionService(db_url=db_url)
            print(f"Initialized DatabaseSessionService with URL: {db_url}")
        except Exception as e:
            print(f"Database Session Service initialization failed: {e}")

    # 3. VertexAiSessionService - Scalable managed sessions
    if VertexAiSessionService:
        project_id = os.getenv("GCP_PROJECT_ID", "your-project")
        location = os.getenv("GCP_LOCATION", "us-central1")
        try:
            vertex_session = VertexAiSessionService(project=project_id, location=location)
            print(f"Initialized VertexAiSessionService for project: {project_id}")
        except Exception as e:
            print(f"VertexAI Session Service initialization failed: {e}")

if __name__ == "__main__":
    demo_session_services()
