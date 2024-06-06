from data.ingest import DocLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
from langchain.schema import StrOutputParser


class RAG:
    def __init__(self, api_key) -> None:
        self.api_key = api_key
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        self.embedding_function = OpenAIEmbeddings(api_key=api_key)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.parser = StrOutputParser()

    def load_db(self, file_path, user_id):
        try:
            self.delete_db(user_id)
        except:
            pass
        self.docloader = DocLoader(
            user_id=user_id, file_path=file_path, api_key=self.api_key
        )
        self.vectorsearch = self.docloader()

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, user_id, query):
        vectorstore = Chroma(
            persist_directory="./chromadb", embedding_function=self.embedding_function
        )
        print(vectorstore.get())
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.2, "filter": {"user_id": user_id}},
        )
        self.rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | self.parser
        )
        return self.rag_chain.invoke(query)

    def delete_db(self, user_id, persist_directory: str = "./chromadb"):
        self.pdfsearch: Chroma = Chroma(persist_directory=persist_directory)
        self.pdfsearch._collection.delete(where={"user_id": user_id})
