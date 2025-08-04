from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import List, Dict, Any
from vector_db import VectorDatabase

class RAGAgent:
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        
        # Initialize a lightweight LLM for CPU inference
        model_name = "microsoft/DialoGPT-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set device to CPUen
        self.device = torch.device("cpu")
        self.model.to(self.device)
        
        # Initialize text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # CPU
            max_length=512,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
    def retrieve_relevant_context(self, query: str, document_id: str = None, 
                                n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector database"""
        return self.vector_db.search_similar_content(
            query=query, 
            n_results=n_results, 
            document_id=document_id
        )
    #
    # def format_context(self, retrieved_results: List[Dict[str, Any]]) -> str:
    #     """Format retrieved context for the LLM"""
    #     if not retrieved_results:
    #         return "No relevant context found."
    #
    #     context_parts = []
    #     for i, result in enumerate(retrieved_results, 1):
    #         content = result['content']
    #         metadata = result['metadata']
    #
    #         context_part = f"Context {i}:\n"
    #         context_part += f"Content: {content}\n"
    #         context_part += f"Source: Page {metadata.get('page_number', 'Unknown')} "
    #         context_part += f"({metadata.get('content_type', 'unknown')} content)\n"
    #
    #         if metadata.get('confidence'):
    #             context_part += f"Confidence: {metadata['confidence']:.2f}\n"
    #
    #         context_parts.append(context_part)
    #
    #     return "\n".join(context_parts)

    def format_context(self, retrieved_results: List[Dict[str, Any]]) -> str:
        """Merge all retrieved chunks into one large context block"""
        if not retrieved_results:
            return "No relevant context found."

        # Just return the merged plain content
        return "\n".join([result['content'] for result in retrieved_results])

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using retrieved context"""
        # Create prompt with context and query
        prompt = f"""Answer the following question based on the document content below.

        Document Content:
        {context}

        Question: {query}
        Answer:"""

        try:
            # Generate response
            response = self.generator(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                truncation=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I couldn't generate a proper response based on the available context."
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def answer_query(self, query: str, document_id: str = None) -> Dict[str, Any]:
        """Main method to answer user queries using RAG"""
        # Step 1: Retrieve relevant context
        retrieved_results = self.retrieve_relevant_context(
            query=query, 
            document_id=document_id
        )
        
        # Step 2: Format context
        formatted_context = self.format_context(retrieved_results)
        
        # Step 3: Generate response
        response = self.generate_response(query, formatted_context)
        
        # Return comprehensive result
        return {
            'query': query,
            'answer': response,
            'retrieved_context': retrieved_results,
            'context_summary': formatted_context,
            'document_id': document_id
        }
    
    def get_document_summary(self, document_id: str) -> str:
        """Get a summary of a specific document"""
        document_content = self.vector_db.get_document_content(document_id)
        
        if not document_content:
            return f"No content found for document {document_id}"
        
        # Extract summaries and key content
        summaries = []
        text_content = []
        visual_content = []
        
        for item in document_content:
            metadata = item['metadata']
            content = item['content']
            
            if metadata.get('content_type') == 'summary':
                summaries.append(f"Page {metadata.get('page_number')}: {content}")
            elif metadata.get('content_type') == 'text':
                text_content.append(content)
            elif metadata.get('content_type') == 'visual':
                visual_content.append(content)
        
        summary_parts = []
        
        if summaries:
            summary_parts.append("Page Summaries:\n" + "\n".join(summaries))
        
        if visual_content:
            summary_parts.append("Visual Elements:\n" + "\n".join(visual_content))
        
        if text_content:
            # Take a sample of text content
            sample_text = " ".join(text_content[:5])
            summary_parts.append(f"Sample Text Content: {sample_text[:200]}...")
        
        return "\n\n".join(summary_parts) if summary_parts else "No content available for summary."
