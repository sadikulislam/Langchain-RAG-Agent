# import necessary libraries
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Chat Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto",
)


# Load environment variables and initialize RAG components
@st.cache_resource
def load_rag_pipeline():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        st.error("HF_TOKEN is not set in the environment variables.")
        st.stop()

    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    db_path = "vectorstore/db_faiss"
    embadding_model = "sentence-transformers/all-MiniLM-L6-v2"

    if not os.path.exists(db_path):
        st.error(f"Vector store not found at {db_path}. Please ensure the db is exist.")
        st.stop()

    endpoint = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)

    chat_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the question using only the provided context. "
                "If the answer is not in the context, say 'I don't know'. "
                "Start your answer directly, no small talk.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    # Load FAISS vector store and create retriever
    embeddings = HuggingFaceEmbeddings(model=embadding_model)
    vectorDB = FAISS.load_local(
        db_path, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorDB.as_retriever(search_kwargs={"k": 3})

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": chat_prompt},
    )
    return qa_chain, retriever


# Streamlit Application UI

st.title("RAG Chat Agent")
st.write("Ask questions based on the provided documents.")
# load the RAG pipeline
try:
    qa_chain, retriever = load_rag_pipeline()
except Exception as e:
    st.error(f"Error loading RAG pipeline: {e}")
    st.stop()

# Seession State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! This is the RAG Agent?"}
    ]
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i + 1}:**")
                    st.info(doc.page_content)

# Accept user input
prompt = st.chat_input("Enter your question here:")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(prompt)
            sources = retriever.invoke(prompt)
            st.markdown(answer["result"])

            # Display source documents if available
            if sources:
                with st.expander("Related Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source document {i + 1}:**")
                        st.info(doc.page_content)

            # Update session state with assistant response and sources
            assistant_message = {
                "role": "assistant",
                "content": answer["result"],
                "sources": sources,
            }
            st.session_state.messages.append(assistant_message)
