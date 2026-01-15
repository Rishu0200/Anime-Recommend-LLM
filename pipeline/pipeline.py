from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config_Anim.config import GROQ_API_KEY, MODEL_NAME, PROCESSED_CSV_PATH
from utils.logger import get_logger
from utils.custom_exception import CustomException
import os

logger = get_logger(__name__)

class AnimeRecommendationPipeline:
    def __init__(self, persist_dir: str = "chroma_db", csv_path: str = None):
        try:
            logger.info("Initializing Recommendation Pipeline")
            
            # Use provided path or config default
            csv_path = csv_path or PROCESSED_CSV_PATH
            if not csv_path or not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
            vector_builder = VectorStoreBuilder(csv_path=csv_path, persist_dir=persist_dir)
            
            if not os.path.exists(persist_dir):
                logger.info("Building vector store...")
                vector_builder.build_and_save_vectorstore()
            
            retriever = vector_builder.load_vector_store().as_retriever(
                search_type="similarity", search_kwargs={"k": 6}
            )
            
            self.recommender = AnimeRecommender(retriever, GROQ_API_KEY, MODEL_NAME)
            logger.info("Pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {str(e)}")
            raise CustomException("Error during pipeline initialization", e)


    
    def recommend(self, query: str) -> dict:  # Changed return type
        try:
            logger.info(f"Received query: {query}")
            recommendation = self.recommender.get_recommendation(query)
            logger.info("Recommendation generated successfully")
            return recommendation  # Returns dict with answer + context
        except Exception as e:
            logger.error(f"Failed to get recommendation: {str(e)}")
            raise CustomException("Error during recommendation generation", e)
