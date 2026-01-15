from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        self.llm = ChatGroq(api_key=api_key, model=model_name, temperature=0)
        self.prompt = get_anime_prompt()
        
        # Modern LCEL approach (non-deprecated)
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.qa_chain = create_retrieval_chain(retriever, doc_chain)

    def get_recommendation(self, query: str):
        result = self.qa_chain.invoke({"input": query})
        return {
            "answer": result["answer"],
            "context_docs": [doc.page_content for doc in result.get("context", [])],
            "source_count": len(result.get("context", []))
        }