import streamlit as st
from model import RAG
from dotenv import load_dotenv
from excel_model import excel_invoke
import pandas as pd

load_dotenv()

st.title("Chatbot with Document Upload and Retrieval")
api_key = st.text_input("OpenAI API Key:-", key="api_key")
if "messages" not in st.session_state:
    st.session_state.messages = []


# uploaded_file = st.file_uploader("Upload PDF Document", type="pdf", accept_multiple_files=True)
uploaded_file = st.file_uploader(
    "Upload an Excel/PDF file", type=["xlsx", "xls", "pdf"]
)

if uploaded_file and api_key:
    file_details = {"filetype": uploaded_file.type}

    try:
        rag_chain.delete_db()
        # st.session_state.messages = []

    except:
        pass
    # Checking file type based on MIME type

    if uploaded_file.type == "application/pdf":
        user_id = st.text_input("Enter User ID:", key="user_id")
        if user_id:
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            rag_chain = RAG(
                user_id=user_id, file_path="uploaded_file.pdf", api_key=api_key
            )
            delete_button = st.button("Delete Database (if existing)")
            clear_chat_history = st.button("Clear Chat")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            chat_history = []
            if delete_button:
                rag_chain.delete_db()
                st.success("Database entries deleted for user ID: " + user_id)
            
            if clear_chat_history:
                chat_history = []

            if prompt := st.chat_input("What is up?"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    answer = rag_chain.generate_response(openai_api_key=api_key,query_text=st.session_state.messages[-1]["content"], chat_history=chat_history)
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

    elif uploaded_file.type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]:

        df = pd.read_excel(uploaded_file)
        st.write("DataFrame Preview:")
        st.write(df.head())

        query = st.text_input("Enter your query")

        if query:

            # Process the uploaded file and query
            output = excel_invoke(df, query, api_key=api_key)

            # Display the result
            st.write("Query Result:")
            st.write(output)

    else:
        st.write("The uploaded file is neither a PDF nor an Excel file.")
