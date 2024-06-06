import os
import sys

# Add the project directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from flask import Flask, request, jsonify
from data.model import RAG
from data.excel_model import ExcelBot
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

app = Flask(__name__)
api_key = os.getenv("OPENAI_API_KEY")
rag_model = RAG(api_key)


@app.route("/load_db", methods=["POST"])
def load_db():
    try:
        data = request.get_json()
        file_path = data["file_path"]
        user_id = data["user_id"]
        rag_model.load_db(user_id=user_id, file_path=file_path)
        return jsonify({"detail": "Database Loaded Successfully."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/pdf_chat", methods=["POST"])
def invoke():
    try:
        data = request.get_json()
        query = data["query"]
        user_id = data["user_id"]
        response = rag_model.invoke(user_id, query)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_db", methods=["DELETE"])
def delete_db():
    try:
        data = request.get_json()
        user_id = data["user_id"]
        rag_model.delete_db(user_id)
        return jsonify({"detail": "Database deleted successfully."})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/excel_chat", methods=["POST"])
def excel_chat():
    try:
        data = request.get_json()
        file_path = data["file_path"]
        query = data["query"]
        excelbot = ExcelBot(file_path=file_path, api_key=api_key)
        response = excelbot.excel_invoke(query)
        # os.remove(file_path)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
