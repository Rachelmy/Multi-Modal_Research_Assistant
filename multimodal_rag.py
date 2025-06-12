from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf  # required popplers and tesseract
import os
import base64
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils import *
import io

# load Gemini_model api key
load_dotenv()


def img_prompt_func(data_dict):
    """
    Join the context into a single string
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # Adding the text for analysis
    text_message = {
        "type": "text",
        "text": (
            "You are a reseracher tasking with providing factual answers from research papers.\n"
            "You will be given a mix of text, tables, and image(s) usually of charts or graphs.\n"
            "Use this information to provide answers related to the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """
    model_vision = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest",temperature=0, max_tokens=1024)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model_vision  # MM_LLM
        | StrOutputParser()
    )

    return chain


def data_loader(pdf_path):
    """
    Data loading and text, table and image summarisation 
    """
    # extract images from documents using unstructured library
    image_path = "./figures"
    pdf_elements = partition_pdf(
        pdf_path, # here pdf_path
        chunking_strategy="by_title",
        extract_images_in_pdf=True,
        infer_table_structure=True,
        max_characters=3000,
        new_after_n_chars=2800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=image_path
        )
    
    # extract tables and texts
    texts, tables = categorize_elements(pdf_elements)
    # Get text & table summaries
    text_summaries, table_summaries = generate_text_summaries(texts[0:19], tables, summarize_texts=True)
    # Image summaries
    img_base64_list, image_summaries = generate_img_summaries(image_path)

    return text_summaries, texts, table_summaries, tables, image_summaries, img_base64_list


def create_retriever(text_summaries, texts, table_summaries, tables, image_summaries, img_base64_list):
    """
    Storing the summaries and raw info in the vector and doc stores respectively for retrieval
    """
    # The vectorstore to use to index the summaries
    vectorstore = Chroma(
        collection_name="mm_rag_gemini",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), # embedding model  
    )

    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore,
        text_summaries,
        texts,
        table_summaries,
        tables,
        image_summaries,
        img_base64_list,
    )

    return retriever_multi_vector_img


class MultiModalRAG:
    """
    Main class for Multi-Modal RAG functionality
    """
    def __init__(self):
        self.retriever = None
    
    def load_pdf(self, pdf_path):
        """
        Load and process PDF file
        """
        pdf_data = data_loader(pdf_path)
        self.retriever = create_retriever(*pdf_data)
        return self.retriever
    
    def query(self, question, limit=1):
        """
        Query the RAG system
        """
        if self.retriever is None:
            raise ValueError("No PDF loaded. Please load a PDF first using load_pdf().")
        
        # Get intermediate results
        docs = self.retriever.get_relevant_documents(question, limit=limit)
        
        # Create RAG chain for final query
        chain = multi_modal_rag_chain(self.retriever)
        
        # Get final answer
        answer = chain.invoke(question)
        
        return {
            'answer': answer,
            'intermediate_docs': docs
        }