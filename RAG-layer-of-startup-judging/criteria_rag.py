import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env if it exists
load_dotenv()

def setup_rag(pdf_path):
    """
    Sets up a RAG system from a PDF file.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} was not found.")

    # 1. Load the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)

    # 3. Initialize OpenAI embeddings and create a FAISS vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # 4. Set up a RetrievalQA chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

def query_rag(qa_chain, question):
    """
    Queries the RAG system.
    """
    response = qa_chain.invoke({"query": question})
    return response["result"]

if __name__ == "__main__":
    # Path to the PDF file
    pdf_file = "RAG-layer-of-startup-judging/criteria_evaluate.pdf"
    
    try:
        # Check for API key
        if "OPENAI_API_KEY" not in os.environ:
            print("Error: OPENAI_API_KEY environment variable not set in .env or system environment.")
        else:
            print(f"Initializing Criteria RAG with {pdf_file}...")
            rag_system = setup_rag(pdf_file)
            
            sample_question = "What are the criteria for judging a startup proposal?"
            print(f"\nQuestion: {sample_question}")
            
            answer = query_rag(rag_system, sample_question)
            print(f"\nAnswer: {answer}")
            
    except Exception as e:
        print(f"An error occurred: {e}")
