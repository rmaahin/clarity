import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import torch
from typing import List, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama
from langgraph.graph import StateGraph, START, END

VECTOR_STATE_PATH = '/mnt/vector_db/pandas_index'
DEFAULT_LLM_PATH = '/mnt/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
DEFAULT_EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llm = Llama(
    model_path = DEFAULT_LLM_PATH,
    n_gpu_layers=-1,
    n_ctx=2048,
    max_tokens=512,
    temperature=0.1,
    verbose=False
)

embedding_model = HuggingFaceEmbeddings(
    model_name = DEFAULT_EMBEDDING_MODEL_PATH,
    model_kwargs = {'device': device}
) 
vector_store = Chroma(
    persist_directory=VECTOR_STATE_PATH,
    embedding_function=embedding_model
)
retriever = vector_store.as_retriever()

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    is_relevant: bool
    answer: str

def retrieve_documents(state: GraphState) -> dict:
    print("Retrieving Relevant Documents...")
    question = state["question"]

    retrieved_documents = retriever.invoke(question)
    
    if not retrieved_documents:
        print("No relevant documents were retrieved. Please reframe your question and try again!")

    print(f"Documents Retrieved: {len(retrieved_documents)}")
    return {"documents": retrieved_documents}

def grade_documents(state: GraphState) -> dict:
    question = state["question"]
    retrieved_documents = state["documents"]

    grading_prompt = PromptTemplate.from_template(
        '''
        You are a relevance grader assessing the relevance between the retrieved document and user question.
        Here's the retrieved document:
        {context}
        Here's the user's question:
        {question}
        Is the document relevant to the question? Answer with a simple 'yes' or 'no'.
        '''

    )

    llm_relevant_docs = []

    for document in retrieved_documents:
        formatted_prompt = grading_prompt.format(
            context=document.page_content,
            question=question
        )

        response = llm(formatted_prompt, max_tokens=10)
        answer_text = response['choices'][0]['text'].lower().strip()

        if 'yes' in answer_text:
            llm_relevant_docs.append(document)
        
    if len(llm_relevant_docs) > 0:
        return {'documents': llm_relevant_docs, 'is_relevant': True}
    else:
        return {'documents': [], 'is_relevant': False}
    
def generate_answer(state: GraphState) -> dict:
    question = state['question']
    relevant_documents = state['documents']

    rag_prompt = PromptTemplate.from_template(
        '''
        You are a helpful assistant. Use the following context to answer the user's question. If you don't know the answer from the context, just say that you don't know.

        Context: {context}

        Question: {question}

        Answer: 
        '''
    )

    relevant_docs_merged = ''
    for document in relevant_documents:
        relevant_docs_merged += document.page_content

    rag_formatted_prompt = rag_prompt.format(
        context = relevant_docs_merged,
        question = question
    )

    response = llm(rag_formatted_prompt, max_tokens=512)
    final_answer = response['choices'][0]['text']

    return {'answer': final_answer}

def no_relevant_fallback(state: GraphState) -> dict:
    print('No relevant documents found!')
    return {'answer': "I'm sorry, but there are no relevant documents found to answer your question."}
    
def is_relevant_condition(state: GraphState) -> str:
    is_relevant = state['is_relevant']

    if is_relevant:
        return "generate_answer"
    else:
        return "no_relevant_fallback"

if __name__ == "__main__":
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("no_relevant_fallback", no_relevant_fallback)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        is_relevant_condition,
        {
            "generate_answer": "generate_answer",
            "no_relevant_fallback": "no_relevant_fallback"
        }
    )
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("no_relevant_fallback", END)

    app = workflow.compile()

    print("---RAG bot is ready. Go ahead and ask a question!---")

    while True:
        query = input("Ask a question (or 'exit' to quit): ")

        if query.lower() == 'exit':
            print("Exiting...")
            break

        try:
            final_state = app.invoke({'question': query})

            print("\n---Answer---")
            print(final_state['answer'])
            print("-----------------------\n")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.\n")