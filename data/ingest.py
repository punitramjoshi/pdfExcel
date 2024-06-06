from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
import os


class DocLoader:
    def __init__(self, user_id, api_key, file_path, persist_dir="./chromadb") -> None:
        self.user_id = user_id
        self.file_path = file_path
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=500,
            separators=[
                "\n\n",
                "\n",
                " ",
                ".",
                ",",
                "",
            ],
        )
        try:
            if os.path.exists(persist_dir):
                self.persist_directory = persist_dir
            else:
                os.mkdir(persist_dir)
                self.persist_directory = persist_dir
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Database Directory not found: {persist_dir}"
            ) from e
        except Exception as e:
            raise ValueError(f"Error during data ingestion: {e}") from e

    def ingest_pdf(self):
        self.delete_db(user_id=self.user_id)
        self.document_list: list[Document] = list()
        print(self.file_path)
        if ".pdf" in self.file_path:
            self.loader = PyPDFLoader(self.file_path)
        self.document_list.extend(
            self.loader.load_and_split(text_splitter=self.text_splitter)
        )
        for pdf_data in self.document_list:
            pdf_data.metadata = {"user_id": self.user_id}

        self.pdfsearch: Chroma = Chroma.from_documents(
            self.document_list,
            self.embeddings,
            persist_directory=self.persist_directory,
        )

    def delete_db(self, user_id):
        self.pdfsearch: Chroma = Chroma(persist_directory="./chromadb")
        self.pdfsearch._collection.delete(where={"user_id": user_id})

    def __call__(self):
        self.ingest_pdf()
