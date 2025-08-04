from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import List
import uuid
import os
from document_processor import DocumentProcessor
from vector_db import VectorDatabase
from rag_agent import RAGAgent
from pdf2image import convert_from_bytes



app = FastAPI(title="Document Extraction & RAG API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
document_processor = DocumentProcessor()
vector_db = VectorDatabase()
rag_agent = RAGAgent(vector_db)

# Ensure upload dir exists
os.makedirs("uploads", exist_ok=True)

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(v) for v in obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@app.get("/")
async def root():
    return {"message": "Document Extraction & RAG API is running 🚀"}

@app.post("/upload-document")
async def upload_document(files: List[UploadFile] = File(...)):
    try:
        document_id = str(uuid.uuid4())
        images = []

        for file in files:
            filename = file.filename.lower()
            content = await file.read()

            if filename.endswith(".pdf"):
                # Convert each page of the PDF to an image
                pdf_images = convert_from_bytes(content)
                for img in pdf_images:
                    img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    images.append(img_np)

            elif filename.endswith((".png", ".jpg", ".jpeg")):
                nparr = np.frombuffer(content, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    raise HTTPException(status_code=400, detail=f"Failed to decode image: {file.filename}")
                images.append(image)

            else:
                raise HTTPException(status_code=415, detail=f"Unsupported file type: {file.filename}")

        if not images:
            raise HTTPException(status_code=400, detail="No valid images extracted from uploads.")

        # Step 1: Process pages through your custom processor
        page_results = document_processor.process_multi_page_document(images)

        # Step 2: Convert numpy types for JSON compatibility
        clean_results = convert_numpy_types(page_results)

        # Step 3: Store extracted results in vector DB
        vector_db.add_document_content(document_id, clean_results)

        return {
            "document_id": document_id,
            "pages_processed": len(clean_results),
            "status": "success",
            "message": f"Document processed with {len(clean_results)} pages."
        }

    except Exception as e:
        print(f"Upload Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_documents(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "").strip()
        document_id = data.get("document_id", None)

        if not query:
            raise HTTPException(status_code=400, detail=" Query cannot be empty")

        result = rag_agent.answer_query(query, document_id)

        return {
            "query": query,
            "answer": result['answer'],
            "context_used": len(result['retrieved_context']),
            "document_id": document_id,
            "retrieved_context": result['retrieved_context']
        }

    except Exception as e:
        print(f"Query Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/documents")
async def list_documents():
    try:
        stats = vector_db.get_collection_stats()
        return {
            "documents": stats['document_ids'],
            "total_documents": stats['total_documents'],
            "total_chunks": stats['total_chunks']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")


@app.get("/documents/{document_id}/summary")
async def get_document_summary(document_id: str):
    try:
        summary = rag_agent.get_document_summary(document_id)
        return {
            "document_id": document_id,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        vector_db.delete_document(document_id)
        return {
            "document_id": document_id,
            "status": "deleted",
            "message": " Document deleted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "document_processor": "ready",
            "vector_db": "ready",
            "rag_agent": "ready"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

