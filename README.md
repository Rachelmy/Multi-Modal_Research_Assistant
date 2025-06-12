# Multi-Modal Research Assistant

Multi-Modal Research Assistant is a powerful multimodal Retrieval-Augmented Generation (RAG) application that enables intelligent querying of PDF documents containing text, tables, and images. Built with Google's Gemini AI, it provides accurate, context-aware answers by analyzing all types of content within your documents.

## 🌟 Features

- **Multimodal Understanding**: Processes text, tables, and images from PDF documents
- **Intelligent Summarization**: Automatically generates summaries of document elements for better retrieval
- **Visual Content Analysis**: Analyzes charts, graphs, and diagrams using vision models
- **Interactive Q&A**: Ask questions in natural language and get comprehensive answers
- **Streamlit Web Interface**: User-friendly web application for easy interaction

## 🏗️ Architecture

The application uses a sophisticated multimodal RAG pipeline:

1. **Document Processing**: Extracts text, tables, and images from PDFs
2. **Content Summarization**: Creates optimized summaries for each element type
3. **Vector Storage**: Stores summaries in a vector database for efficient retrieval
4. **Multimodal Retrieval**: Retrieves relevant content based on user queries
5. **AI Generation**: Uses Google Gemini to generate comprehensive answers

## 📋 Prerequisites

- Python 3.8+
- Google AI API key (for Gemini models)
- Poppler (for PDF processing)
- Tesseract OCR (for text extraction)

### System Dependencies


**macOS:**
```bash
brew install poppler tesseract
```


## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Multi-Modal_Research_Assistant
```

2. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```


## 🖥️ Usage

### Running the Streamlit App

1. **Start the application:**
```bash
streamlit run streamlit_app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload a PDF** using the file uploader

4. **Ask questions** about your document in the text input field

5. **View results** including retrieved content and AI-generated answers

### Using the Core Library

You can also use the core functionality programmatically:

```python
from multimodal_rag import MultiModalRAG

# Initialize the RAG system
rag = MultiModalRAG()

# Load a PDF document
rag.load_pdf("path/to/your/document.pdf")

# Query the document
result = rag.query("What are the main findings in this research?")

print("Answer:", result['answer'])
print("Retrieved documents:", len(result['intermediate_docs']))
```

## 📁 Project Structure

```
Multi-Modal_Research_Assistant/
├── streamlit_app.py          # Streamlit web interface
├── multimodal_rag.py         # Core RAG functionality
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (create this)
├── README.md                 # This file
└── figures/                  # Extracted images (auto-created)
```

## 🔧 Configuration


### Model Configuration

The application uses the following Google AI models:
- **Text Generation**: `gemini-1.5-pro-latest`
- **Embeddings**: `embedding-001`
- **Vision**: `gemini-1.5-pro-latest`

You can modify these in the respective files if needed.

## 🎯 Use Cases

- **Medical Reports**: Analyze reports with charts and tables
- **Research Paper Analysis**: Extract insights from academic papers with complex figures
- **Technical Documentation**: Query manuals and guides with diagrams
- **Scientific Literature**: Understand papers with experimental data and visualizations
- **Educational Materials**: Study textbooks with mixed content types
