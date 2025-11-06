import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import os
import argparse
import torch
from typing import List, TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama
from langgraph.graph import StateGraph, END
from langchain_tavily import TavilySearch

VECTOR_STATE_PATH = '/mnt/vector_db/pandas_index'
DEFAULT_LLM_PATH = '/mnt/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    is_relevant: bool
    answer: str
    chat_history: List[str]

    next_agent: str

class RAGBot:

    '''
    A RAG bot that loads the relevant documents, scores the relevance of the documents to user's query and gives and answer.

    Args:
        model_path(str): The path to the model file in your local directory.
        vector_path(str): The path to the folder that contains the vector database embeddings.  
        embedding_model(str): Embedding model used to produce the vector database.
    '''

    def __init__(self, model_path: str, vector_path: str, embedding_model: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'\nUsing device: {self.device}')

        self.llm = Llama(
            model_path = model_path,
            n_gpu_layers=-1,
            n_ctx=2048,
            max_tokens=512,
            temperature=0.1,
            verbose=False
        )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name = embedding_model,
            model_kwargs = {'device': self.device}
        )

        self.vector_store = Chroma(
            persist_directory = vector_path,
            embedding_function = self.embedding_model
        )
        self.retriever = self.vector_store.as_retriever()

        # setting up the nodes
        workflow = StateGraph(GraphState)
        workflow.add_node("condense", self._memory_condense_question)
        workflow.add_node("bouncer", self._bouncer)
        workflow.add_node("offtopic_fallback", self._offtopic_fallback_agent)
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("grade_documents", self._grade_documents)
        workflow.add_node("web_search", self._web_search_agent)
        workflow.add_node("generate_answer", self._generate_answer)

        router_map = {
                "bouncer": "bouncer",
                "offtopic_fallback": "offtopic_fallback",
                "retrieve": "retrieve",
                "grade_documents": "grade_documents",
                "web_search": "web_search",
                "generate_answer": "generate_answer",
                "END": END
            }

        # setting the entry point
        workflow.set_entry_point("condense")

        workflow.add_conditional_edges("condense", self._router_function, router_map)
        workflow.add_conditional_edges("offtopic_fallback", self._router_function, router_map)
        workflow.add_conditional_edges("bouncer", self._router_function, router_map)
        workflow.add_conditional_edges("retrieve", self._router_function, router_map)
        workflow.add_conditional_edges("grade_documents", self._router_function, router_map)
        workflow.add_conditional_edges("generate_answer", self._router_function, router_map)
        workflow.add_conditional_edges("web_search", self._router_function, router_map)

        self.app = workflow.compile()

    def _memory_condense_question(self, state: GraphState) -> dict:
        '''
        Stores the chat history and condenses it in order to preserve it in the memory and connect it to the next question.
        '''
        question = state['question']
        chat_history = state['chat_history']

        if not chat_history:
            return {'question': question, 'next_agent': 'bouncer'}
           
        condense_prompt_template = PromptTemplate.from_template(
            """
            Given the following conversation history and a new question, rephrase the new question to be a standalone question.

            Chat History:
            {chat_history}

            New Question: {question}

            Standalone Question:
            """
        )

        formatted_history = "\n".join(chat_history)
        formatted_prompt = condense_prompt_template.format(
            chat_history = formatted_history,
            question = question
        )  

        condenser_llm_response = self.llm(formatted_prompt, max_tokens=200)
        condensed_question = condenser_llm_response['choices'][0]['text'].strip()

        print(f'\nCondensed question: {condensed_question}')

        return {'question': condensed_question, 'next_agent': 'bouncer'}
    
    def _bouncer(self, state: GraphState) -> dict:
        '''
        This function helps to check if user's question is actually relevant to the topic. Uses cosine similarity to measure how relevant the question is to the embeddings present in the vector store.
        '''

        question = state['question']
        results = self.vector_store.similarity_search_with_score(question, k=1)

        if not results:
            return {'next_agent': 'offtopic_fallback'}

        _, score = results[0]
        
        bouncer_threshold = 1.0
        # print(score)
        if score < bouncer_threshold:
            return {'next_agent': 'retrieve'}
        else:
            return {'next_agent': 'offtopic_fallback'}
        
    def _offtopic_fallback_agent(self, state: GraphState) -> dict:
        '''
        Responds to the user if they ask an irrelevant question.
        '''

        print('\n---OFFTOPIC ALERT!---')
        return {'answer': f"I'm a python libraries assistant. I can only answer questions related to Python libraries so please try again.", "next_agent": "END"}

    def _retrieve_documents(self, state: GraphState) -> dict:
        '''
        Takes a look at the question and retrieves the chunks that are most relevant to the question.
        '''

        print("\nRetrieving Relevant Documents...")
        question = state["question"]

        retrieved_documents = self.retriever.invoke(question)
        
        if not retrieved_documents:
            print("\nNo relevant documents were retrieved. Please reframe your question and try again!")

        # print(f"Documents Retrieved: {len(retrieved_documents)}")
        return {"documents": retrieved_documents, 'next_agent': 'grade_documents'}

    def _grade_documents(self, state: GraphState) -> dict:
        '''
        Considers the retrieved documenst by retrieve_documents() function and prompts an LLM to check if the chunks are actually relevant.
        Appends only the documents that deemed as relevant by the LLM to 'llm_relevant_docs' for the next step. 
        If no documents are deemed as actual matches by the LLM, returns an empty list for the next step.
        '''

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

            response = self.llm(formatted_prompt, max_tokens=10)
            answer_text = response['choices'][0]['text'].lower().strip()

            if 'yes' in answer_text:
                llm_relevant_docs.append(document)

        print(f'Relevant documents found: {len(llm_relevant_docs)}')
            
        if len(llm_relevant_docs) > 0:
            return {'documents': llm_relevant_docs, 'is_relevant': True, 'next_agent': 'generate_answer'}
        else:
            return {'documents': [], 'is_relevant': False, 'next_agent': 'web_search'}
        
    def _web_search_agent(self, state: GraphState) -> dict:
        '''
        If no relevant documents were found, this function is invoked in order to search the web and provide an answer to the query.
        '''

        print('---NO RELEVANT DOCUMENTS FOUND. SEARCHING THE WEB FOR ANSWERS!---')
        question = state['question']
        
        search_tool = TavilySearch(
            max_results = 5,
        )

        results = search_tool.invoke({'query': question})

        web_search_docs = []
        for result in results['results']:
            web_search_docs.append(Document(page_content = result['content'], metadata = {'source': result['url']}))
        
        return {'documents': web_search_docs, 'next_agent': 'generate_answer'}

    def _generate_answer(self, state: GraphState) -> dict:
        '''
        Prompts an LLM to generate an answer given a question and relevant chunks/bodies of data from the library's documentation.
        '''

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

        response = self.llm(rag_formatted_prompt, max_tokens=512)
        final_answer = response['choices'][0]['text']

        return {'answer': final_answer, 'next_agent': 'END'}

    # def _no_relevant_fallback(self, state: GraphState) -> dict:
    #     '''
    #     If no relevant documents were found, this node is invoked.    
    #     '''

    #     print('No relevant documents found!')
    #     return {'answer': "I'm sorry, but there are no relevant documents found to answer your question.", 'next_agent': 'END'}
        
    def _router_function(self, state: GraphState) -> str:
        '''
        This is the central router function for our multi-agent system. It reads the 'next_agent' key from the state and returns the string value.
        This way, the function acts as a traffic cop redirecting the graph to the next nodes.
        '''

        # print(f"ROUTING TO --> {state['next_agent']}")

        return state['next_agent']
        
    def rag_execute(self):

        '''
        Invokes the RAG pipeline defined in __init__ and starts a CLI for the user to ask queries.
        '''

        print("\n---RAG bot is ready. Go ahead and ask a question!---")
        chat_history = []

        while True:
            query = input("\nAsk a question (or 'exit' to quit): ")

            if query.lower() == 'exit':
                print("\nExiting...")
                break

            try:
                state_inputs = {'question': query, 'chat_history': chat_history, 'answer': ''}
                final_state = self.app.invoke(state_inputs)
                
                if len(final_state['answer'])!=0:
                    answer = final_state['answer'].strip()

                    # Updating the chat history
                    chat_history.append(f"Question: {query}")
                    chat_history.append(f"Answer: {answer}")

                    print("\n---Answer---\n")
                    print(answer)
                    print("\n-----------------------\n")

            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("\nPlease try again.\n")      

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-mp", "--model_path", type=str, help='The path to the LLM', default=DEFAULT_LLM_PATH)
    parser.add_argument("-v", "--vector_state_path", type=str, help='The path to the vector store', default=VECTOR_STATE_PATH)
    parser.add_argument("-ep", "--embedding_model_path", type=str, help='The file path to the embedding model.', default=DEFAULT_EMBEDDING_MODEL)
    args = parser.parse_args()

    bot = RAGBot(
        model_path = args.model_path,
        vector_path = args.vector_state_path,
        embedding_model_path = args.embedding_model_path    
    )

    bot.rag_execute()

    

    