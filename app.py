import streamlit as st
import os
from supabase import create_client

# 🎬 1. App Title and Layout Setup
st.set_page_config(page_title="Data Science Course AI", page_icon="🤖")
st.title("📚 Course Assistant Chatbot")
st.write("Ask me anything about the 20-Day Data Science Curriculum!")

# 🎬 2. Load API Keys Safely (Streamlit uses st.secrets locally)
# In your local project directory, create a folder called `.streamlit/secrets.toml`
# Add:
# GEMINI_API_KEY = "your_key"
# SUPABASE_URL = "your_url"
# SUPABASE_KEY = "your_key"

GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# Fallback environment setup for LangChain modules
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# 🎬 3. RAG Pipeline Initialization (Cached to run only ONCE, not on every click)
@st.cache_resource
def initialize_rag_pipeline():
    # Place all your imports, loading, FAISS building, and chain initialization here
    # from langchain_google_genai import ChatGoogleGenerativeAI...
    # return chain
    pass


# Call our cached function (uncomment when your imports are packed inside it)
# chain = initialize_rag_pipeline()
# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 🎬 4. The Dynamic UI Logic
# This turns your hardcoded query into a dynamic browser input field!
user_query = st.text_input("Enter your question here:", placeholder="e.g., What will I learn on Day 4?")

if user_query:
    with st.spinner("🤔 Querying Gemini pipeline..."):
        try:
            # 1. Fire the RAG chain dynamically
            # chain_response = chain.invoke({"query": user_query})
            # response_text = chain_response["result"]

            # Temporary dummy response text for layout visualization
            response_text = "Here is what you learn on that day..."

            # 2. Render response nicely in the UI
            st.markdown("### 🤖 Bot Response:")
            st.write(response_text)

            # 3. Log it dynamically to Supabase on the fly
            log_data = {
                "session_id": st.session_state.get("session_id", "streamlit_user_1"),
                "user_query": user_query,
                "bot_response": response_text
            }
            # supabase.table("chat_logs").insert(log_data).execute()
            st.caption("✨ Conversation saved to database successfully.")

        except Exception as e:
            st.error(f"An error occurred: {e}")