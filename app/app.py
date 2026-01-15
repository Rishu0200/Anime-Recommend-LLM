import streamlit as st
from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv

st.set_page_config(
    page_title="Anime Recommender", 
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

@st.cache_resource
def init_pipeline():
    """Initialize the recommendation pipeline once."""
    return AnimeRecommendationPipeline()

pipeline = init_pipeline()

## Main App Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🎌 Anime Recommender System")
    st.markdown("Enter your preferences below for personalized anime suggestions!")

with col2:
    st.markdown("### Tips")
    st.markdown("- 'lighthearted school anime'")
    st.markdown("- 'dark fantasy with strong female leads'")
    st.markdown("- 'mecha action series'")

# Input with validation
query = st.text_input(
    "Enter your anime preferences:", 
    placeholder="e.g., lighthearted anime with school settings",
    help="Describe genres, themes, or vibes you enjoy!"
)

if query:
    if len(query.strip()) < 3:
        st.warning("Please enter at least 3 characters for better recommendations!")
    else:
        with st.spinner("🔍 Finding your perfect anime..."):
            try:
                response = pipeline.recommend(query)
                answer_text = response["answer"] if isinstance(response, dict) else response
                st.markdown("### 🎥 Your Recommendations")
                st.markdown(answer_text)
                
                # Optional: Sidebar with debug info
                with st.sidebar:
                    st.markdown("### Debug Info")
                    if hasattr(response, 'source_count'):
                        st.metric("Documents Retrieved", response.get('source_count', 0))
                        
            except Exception as e:
                st.error(f"❌ Something went wrong: {str(e)}")
                st.info("Try a different query or check your pipeline setup.")

# Footer
st.markdown("---")