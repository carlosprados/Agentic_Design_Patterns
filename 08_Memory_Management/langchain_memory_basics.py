import os
from dotenv import load_dotenv

# LangChain imports
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.memory import ChatMessageHistory, ConversationBufferMemory
    from langchain.chains import LLMChain
    from langchain_core.prompts import (
        ChatPromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
except ImportError:
    print("Error: Required LangChain components not found.")
    ChatGoogleGenerativeAI = ChatMessageHistory = ConversationBufferMemory = LLMChain = None

# Load environment variables
load_dotenv()

def demo_langchain_memory():
    if not all([ChatGoogleGenerativeAI, ChatMessageHistory, ConversationBufferMemory, LLMChain]):
        print("Skipping LangChain memory demo due to missing dependencies.")
        return

    print("--- LangChain Memory Basics ---")

    # 1. Basic Chat Message History
    history = ChatMessageHistory()
    history.add_user_message("I'm heading to New York next week.")
    history.add_ai_message("Great! It's a fantastic city.")
    print(f"History messages: {history.messages}")

    # 2. Conversation Buffer Memory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # 3. Conversational Chain with Memory
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessagePromptTemplate.from_template("You are a friendly travel assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{question}")
            ]
        )
        
        conversation = LLMChain(llm=llm, prompt=prompt, memory=memory)

        print("\n--- Starting Conversation ---")
        response1 = conversation.predict(question="Hi, I'm Jane.")
        print(f"AI: {response1}")
        
        response2 = conversation.predict(question="Do you remember my name?")
        print(f"AI: {response2}")
        
    except Exception as e:
        print(f"LLM chain execution failed (ensure GOOGLE_API_KEY is set): {e}")

if __name__ == "__main__":
    demo_langchain_memory()
