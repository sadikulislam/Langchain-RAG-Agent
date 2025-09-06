from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id=HF_REPO_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_new_tokens=512,
    )
)

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

# Load FAISS vector store
DB = "vectorstore/db_faiss"
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vectorDB = FAISS.load_local(DB, embeddings, allow_dangerous_deserialization=True)
retriever = vectorDB.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": chat_prompt},
)

query = input("Write Query Here: ")
answer = qa_chain.invoke(query)

print("RESULT: ", answer["result"])

print("SOURCE DOCUMENTS: ")
for doc in answer["source_documents"]:
    print(doc.page_content, "\n---\n")
