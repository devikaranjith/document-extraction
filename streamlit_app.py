import streamlit as st
import requests
from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_bytes
import io

API_BASE_URL = "http://localhost:8000"

def main():
    st.title("Document Extraction and Query System")
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox("Select a page", ["Upload", "Query", "Manage"])

    if page == "Upload":
        upload_page()
    elif page == "Query":
        query_page()
    elif page == "Manage":
        management_page()

def upload_page():
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose document files (images or PDFs)",
        type=['png', 'jpg', 'jpeg', 'pdf'],
        accept_multiple_files=True,
        help="Upload scanned PDFs or images of a document"
    )

    if uploaded_files:
        st.subheader("Preview Uploaded Files")

        for i, uploaded_file in enumerate(uploaded_files):
            file_ext = uploaded_file.name.lower().split('.')[-1]

            if file_ext == 'pdf':
                try:
                    pdf_bytes = uploaded_file.read()
                    images = convert_from_bytes(pdf_bytes)
                    for j, img in enumerate(images):
                        st.image(img, caption=f"{uploaded_file.name} - Page {j + 1}", use_column_width=True)
                    uploaded_file.seek(0)  # Reset stream for upload
                except Exception as e:
                    st.warning(f"Could not preview PDF `{uploaded_file.name}`: {e}")
            else:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Image {i + 1}: {uploaded_file.name}", use_column_width=True)
                    uploaded_file.seek(0)
                except UnidentifiedImageError:
                    st.warning(f"Could not preview image: {uploaded_file.name}")

        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    # Prepare files for upload
                    files = []
                    for f in uploaded_files:
                        f.seek(0)
                        files.append(('files', (f.name, f.read(), f.type)))

                    response = requests.post(f"{API_BASE_URL}/upload-document", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.success("Document processed successfully.")
                        st.write(f"Document ID: `{result['document_id']}`")
                        st.write(f"Pages processed: {result['pages_processed']}")
                        st.session_state.setdefault('processed_documents', []).append(result['document_id'])
                    else:
                        st.error(f"Failed: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

def query_page():
    st.header("Query Documents")

    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        documents = response.json().get('documents', []) if response.status_code == 200 else []
    except:
        documents = []

    if not documents:
        st.warning("No documents found.")
        return

    query = st.text_input("Enter your query:")
    selected_doc = st.selectbox("Choose document", ["All"] + documents)

    if st.button("Ask") and query:
        with st.spinner("Fetching answer..."):
            try:
                params = {"query": query}
                if selected_doc != "All":
                    params["document_id"] = selected_doc

                response = requests.post(f"{API_BASE_URL}/query", json=params)
                result = response.json()

                if response.status_code == 200:
                    st.subheader("Answer")
                    st.write(result['answer'])

                    if result.get('retrieved_context'):
                        with st.expander("Context"):
                            for i, ctx in enumerate(result['retrieved_context'], 1):
                                st.markdown(f"**Context {i}:** {ctx['content']}")
                                metadata = ctx.get('metadata', {})
                                st.markdown(f"*Page:* {metadata.get('page_number', '?')}, *Type:* {metadata.get('content_type', '?')}, *Confidence:* {metadata.get('confidence', '?')}")
                                st.markdown("---")
                else:
                    st.error(result.get("detail", "Error during query"))
            except Exception as e:
                st.error(f"Query failed: {e}")

def management_page():
    st.header("Manage Documents")

    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code != 200:
            st.error("Failed to fetch documents.")
            return

        data = response.json()
        docs = data.get('documents', [])

        st.metric("Total Documents", data.get('total_documents', 0))
        st.metric("Content Chunks", data.get('total_chunks', 0))

        if docs:
            for doc_id in docs:
                with st.expander(f"{doc_id}"):
                    if st.button("View Summary", key=f"sum_{doc_id}"):
                        try:
                            res = requests.get(f"{API_BASE_URL}/documents/{doc_id}/summary")
                            summary = res.json().get("summary", "") if res.status_code == 200 else "Summary not available."
                            st.text_area("Summary", summary, height=150)
                        except Exception as e:
                            st.error(str(e))

                    if st.button("Delete", key=f"del_{doc_id}"):
                        try:
                            del_res = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
                            if del_res.status_code == 200:
                                st.success("Deleted.")
                                st.experimental_rerun()
                            else:
                                st.error("Delete failed.")
                        except Exception as e:
                            st.error(str(e))
        else:
            st.info("No documents available.")
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
