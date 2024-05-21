from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain_openai import ChatOpenAI
from ingest import DocLoader
from langchain import hub
import langchain

# from langchain.memory import ConversationBufferWindowMemory
# from operator import itemgetter


class RAG:
    def __init__(self, user_id, file_path: str, api_key: str) -> None:
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        self.docloader = DocLoader(
            user_id=user_id, file_path=file_path, api_key=api_key
        )
        self.retriever = self.docloader()
        self.prompt = hub.pull("rlm/rag-prompt")
        self.parser = StrOutputParser()
        self.user_id = user_id

        # # Initialize the conversational memory
        # self.memory = ConversationBufferWindowMemory(
        #     k=5,  # The number of previous interactions to remember
        #     return_messages=True  # Whether to return the stored messages or not
        # )

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, query):
        langchain.debug = True
        self.rag_chain = (
            {
                "context": self.retriever | self.format_docs,
                "question": RunnablePassthrough(),
                # "chat_history": RunnableLambda(self.memory.load_memory_variables)
            }
            | self.prompt
            | self.llm
            | self.parser
        )
        return self.rag_chain.invoke(query)
    
    def generate_response(self,openai_api_key, query_text, chat_history):
        print(chat_history)
        print("Inner API Key--------------",openai_api_key)
        print(openai_api_key)
        # Set up history-aware retriever
        llm = ChatOpenAI(api_key=openai_api_key)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, self.retriever, contextualize_q_prompt
        )

        # Set up the QA system prompt
        qa_system_prompt = (
            "You are an assistant for question-answering tasks. Use "
            "the following pieces of retrieved context to answer the "
            "question. If you don't know the answer, just say that you "
            "don't know. Use three sentences maximum and keep the answer "
            "concise."
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create the question-answer chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Invoke the chain with the query and chat history
        response = rag_chain.invoke({"input": query_text, "chat_history": chat_history})
        print(response)
        return response["answer"]

    def delete_db(self):
        self.docloader.delete_db()
