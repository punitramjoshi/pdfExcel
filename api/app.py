import os
import sys

# Add the project directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from data.model import RAG
from dotenv import load_dotenv
from data.excel_model import ExcelBot
import pandas as pd
import langchain

langchain.debug=True

load_dotenv()

st.title("Chat with Excel and PDF")
api_key = st.text_input("OpenAI API Key:-", key="api_key")

if "messages" not in st.session_state:
    st.session_state.messages = []


uploaded_file = st.file_uploader(
    "Upload an Excel/CSV/PDF file", type=["xlsx", "xls", "pdf", "csv"]
)

if uploaded_file and api_key:
    file_details = {"filetype": uploaded_file.type}

    try:
        rag_chain.delete_db()
        # st.session_state.messages = []

    except:
        pass
    # Checking file type based on MIME type
    try:
        if uploaded_file.type == "application/pdf":
            rag_model = RAG(api_key)
            user_id = st.text_input("Enter User ID:", key="user_id")
            if user_id:
                with open("uploaded_file.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())

                rag_model.load_db("uploaded_file.pdf", user_id)
                st.success("Database loaded successfully!")
                delete_button = st.button("Delete Database (if existing)")
                clear_chat_history = st.button("Clear Chat")

                if delete_button:
                    rag_model.delete_db(user_id)
                    st.success("Database deleted successfully!")

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if clear_chat_history:
                    st.session_state.messages = []

                if prompt := st.chat_input("What is up?"):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        answer = rag_model.invoke(user_id=user_id, query=prompt)
                        st.markdown(answer)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer}
                        )

        elif uploaded_file.type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "text/csv"
        ]:
            if uploaded_file.type=="text/csv":
                temp_file = "uploaded_file.csv"
            else:
                temp_file = "uploaded_file.xlsx"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if ".csv" in temp_file:
                df = pd.read_csv(uploaded_file)
                st.write("DataFrame Preview:")
                st.write(df.head())
                excelbot = ExcelBot(file_path=temp_file, api_key=api_key)
            else:
                sheet_name = st.text_input("Sheet name (e.g., Master_Sheet) or Number(e.g., 3)", value=0)
                print(sheet_name)
                if sheet_name:
                    try:
                        sheet_name = int(sheet_name)
                    except:
                        pass
                    # df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                    # st.write(df.head())
                    print("Sheet name is ", sheet_name, "Sheet name type is ", type(sheet_name))
                    excelbot = ExcelBot(file_path=temp_file, api_key=api_key, sheet_name=sheet_name)
                    st.write("DataFrame Preview:")
                    st.write(excelbot.clean_df.head())

            query = st.text_input("Enter your query")
            if query:

                # Process the uploaded file and query
                output = excelbot.excel_invoke(query)

                # Display the result
                st.write("Query Result:")
                st.write(output)

        else:
            st.write("The uploaded file is neither a PDF nor an Excel file.")
    except Exception as e:
        st.write(e)