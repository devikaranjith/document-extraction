# Build a system to extract content from scanned multi-page documents

A comprehensive system for extracting content from scanned multi-page documents containing handwritten text (English & Malayalam) and visual elements, with a Retrieval-Augmented Generation (RAG) pipeline for intelligent querying.

# Installation & Setup
1.Clone and navigate to the project
2.Build and run with Docker Compose
  docker-compose up --build
3.Install system and python dependencies
4.Run the application

Terminal 1: Start FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000

Terminal 2: Start Streamlit
streamlit run streamlit_app.py --server.port 8501

# Usage
1. Upload Documents
Navigate to the "Upload" page in Streamlit
Select multiple image files (PNG, JPG, JPEG, TIFF, BMP)
Click "Process" to extract content
Note the generated Document ID for future queries
2. Query Documents
Go to "Query" page
Enter your question in natural language
Select a specific document or search all documents
Get AI-generated answers with source context
3. Manage Documents
View document statistics and summaries
Delete unwanted documents

OCR Engines
EasyOCR: Primary engine for handwritten text (English & Malayalam)
Tesseract: Secondary engine with language-specific configurations
Vision Model
BLIP: Salesforce's image captioning model for visual element analysis
Language Model
DialoGPT: Microsoft's conversational AI for response generation
CPU-optimized: Runs efficiently without GPU requirements
Vector Database
ChromaDB: Persistent vector storage with cosine similarity
Sentence Transformers: all-MiniLM-L6-v2 for embeddings

More fine tuneing and enhancements need to be done this is a simple overview of the project. 
To Do : 
Enhancement in extraction and extraction from malayalam text is 

Attaching some images of how app works
<img width="1776" height="1002" alt="Screenshot from 2025-08-04 22-15-03" src="https://github.com/user-attachments/assets/837421eb-c210-4b62-8cff-281d7a781aa3" />

<img width="898" height="620" alt="Screenshot from 2025-08-04 00-01-26" src="https://github.com/user-attachments/assets/45fb5d59-8a51-4528-a672-e09fb9f892eb" />



