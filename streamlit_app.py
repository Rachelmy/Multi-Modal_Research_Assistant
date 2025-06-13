import streamlit as st
import base64
import io
import os
import shutil
from PIL import Image
from multimodal_rag import MultiModalRAG
from utils import is_image_data


def plt_img_base64(b64data):
    """
    Display base64-encoded image data
    """
    try:
        decoded_bytes = base64.b64decode(b64data)
        image = Image.open(io.BytesIO(decoded_bytes))
        # Display the image using st.image
        st.image(image, caption='Base64-encoded image')
    except Exception as e:
        st.write(f"Error displaying image: {e}")


def save_uploaded_file(uploaded_file):
    """
    Save uploaded file to disk and return the file path
    """
    try:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def delete_figures():
    """
    Delete all files in the figures directory
    """
    figures_dir = "./figures"
    if os.path.exists(figures_dir):
        try:
            shutil.rmtree(figures_dir)
            os.makedirs(figures_dir)  # Recreate the empty directory
            return True
        except Exception as e:
            st.error(f"Error deleting figures: {e}")
            return False
    return True


def main():
    st.title("Multi-Modal Research Assistant")
    
    # Initialize the RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MultiModalRAG()
    
    # PDF upload file container
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])
    
    # Check if PDF file is uploaded
    if pdf_file is not None:
        # Save uploaded file to disk
        pdf_path = save_uploaded_file(pdf_file)
        
        if pdf_path:
            # Initialize session state to store retriever data
            if 'pdf_loaded' not in st.session_state or st.session_state.get('current_pdf') != pdf_file.name:
                with st.spinner("Processing PDF... This may take a moment."):
                    try:
                        # Load PDF into RAG system
                        st.session_state.rag_system.load_pdf(pdf_path)
                        st.session_state.pdf_loaded = True
                        st.session_state.current_pdf = pdf_file.name
                        st.success("PDF processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")
                        st.session_state.pdf_loaded = False
    else:
        # If no PDF is uploaded (user clicked 'x'), delete figures and reset state
        if 'pdf_loaded' in st.session_state and st.session_state.pdf_loaded:
            if delete_figures():
                st.session_state.pdf_loaded = False
                st.session_state.current_pdf = None
                st.session_state.rag_system = MultiModalRAG()
    
    # User input section
    st.subheader("Ask a Question")
    user_input = st.text_input("Enter your question:")
    generate_button = st.button("Generate Answer")
    
    # Perform QA on button click
    if generate_button:
        if pdf_file is not None and st.session_state.get('pdf_loaded', False):
            if user_input.strip():
                with st.spinner("Generating answer..."):
                    try:
                        # Query the RAG system
                        result = st.session_state.rag_system.query(user_input)
                        
                        # Display intermediate results in an expandable section
                        with st.expander("Multi-Vector Retriever Results"):
                            st.write("**Retrieved documents based on your query:**")
                            for i, doc in enumerate(result['intermediate_docs']):
                                st.write(f"**Document {i+1}:**")
                                if is_image_data(doc):
                                    plt_img_base64(doc)
                                else:
                                    st.write(doc)
                                st.write("---")
                        
                        # Display final result
                        st.subheader("Answer")
                        st.write(result['answer'])
                        
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
            else:
                st.warning("Please enter a question.")
        else:
            st.warning("Please upload a PDF file first.")


if __name__ == "__main__":
    main()