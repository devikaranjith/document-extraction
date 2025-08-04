import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class VectorDatabase:
    def __init__(self, persist_directory: str = "./chroma_db"):
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="document_extractions",
            metadata={"hnsw:space": "cosine"}
        )

    def add_document_content(self, document_id: str, page_results: List[Dict[str, Any]]):
        """Add extracted document content to vector database as a single chunk"""
        all_texts = []
        all_visuals = []
        all_summaries = []

        for page_result in page_results:
            page_num = page_result['page_number']

            for text_item in page_result['text_extractions']:
                if text_item['confidence'] > 0.4:
                    all_texts.append(text_item['text'])

            for visual_item in page_result['visual_elements']:
                all_visuals.append(visual_item['description'])

            if page_result.get('page_summary'):
                all_summaries.append(page_result['page_summary'])

        # Join everything into one chunk
        full_content = ""
        if all_summaries:
            full_content += "Page Summaries:\n" + "\n".join(all_summaries) + "\n\n"
        if all_visuals:
            full_content += "Visual Elements:\n" + "\n".join(all_visuals) + "\n\n"
        if all_texts:
            full_content += "Text Content:\n" + "\n".join(all_texts)

        if full_content.strip():
            self.collection.add(
                documents=[full_content],
                metadatas=[{
                    'document_id': document_id,
                    'content_type': 'combined',
                    'confidence': 1.0
                }],
                ids=[f"{document_id}_combined"]
            )

    def search_similar_content(self, query: str, n_results: int = 5,
                             document_id: str = None) -> List[Dict[str, Any]]:
        """Search for similar content in the vector database"""
        where_clause = {}
        if document_id:
            where_clause['document_id'] = document_id
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i]
                })
        
        return formatted_results
    
    def get_document_content(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all content for a specific document"""
        results = self.collection.get(
            where={'document_id': document_id}
        )
        
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'])):
                formatted_results.append({
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'id': results['ids'][i]
                })
        
        return formatted_results
    
    def list_documents(self) -> List[str]:
        """List all document IDs in the database"""
        results = self.collection.get()
        document_ids = set()
        
        if results['metadatas']:
            for metadata in results['metadatas']:
                if 'document_id' in metadata:
                    document_ids.add(metadata['document_id'])
        
        return list(document_ids)
    
    def delete_document(self, document_id: str):
        """Delete all content for a specific document"""
        # Get all IDs for the document
        results = self.collection.get(
            where={'document_id': document_id}
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        documents = self.list_documents()
        
        return {
            'total_chunks': count,
            'total_documents': len(documents),
            'document_ids': documents
        }
