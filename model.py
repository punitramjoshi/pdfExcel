from ingest import DocLoader
import langchain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.schema import StrOutputParser

# from langchain.memory import ConversationBufferWindowMemory
# from operator import itemgetter


class RAG:
    def __init__(self, user_id, file_path: str, api_key:str) -> None:
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        self.docloader = DocLoader(user_id=user_id, file_path=file_path)
        self.vectorsearch, self.retriever = self.docloader()
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

    def delete_db(self):
        self.docloader.delete_db()
