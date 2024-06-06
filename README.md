# Documentation for Streamlit App

## Command to run in the root folder

```
streamlit run api/app.py
```

# API Documentation for Flask API

This API allows interaction with a RAG model and an ExcelBot for various tasks such as loading a database, querying PDFs, deleting databases, and querying Excel files.

## Base URL

```
http://<your-server-domain>
```

## Endpoints

### 1. Load Database

#### Endpoint

```
POST /load_db
```

#### Description

Loads a database from a specified file path for a given user.

#### Request

- **Headers:** `Content-Type: application/json`
- **Body:**

```json
{
    "file_path": "path/to/your/file",
    "user_id": "unique_user_id"
}
```

#### Response

- **Success (200):**

```json
{
    "detail": "Database Loaded Successfully."
}
```

- **Error (500):**

```json
{
    "error": "Error message"
}
```

#### Example Curl Request

```bash
curl -X POST http://<your-server-domain>/load_db -H "Content-Type: application/json" -d '{"file_path": "path/to/your/file", "user_id": "unique_user_id"}'
```

### 2. PDF Chat

#### Endpoint

```
POST /pdf_chat
```

#### Description

Queries the RAG model with a user query and returns the response.

#### Request

- **Headers:** `Content-Type: application/json`
- **Body:**

```json
{
    "query": "your_query_here",
    "user_id": "unique_user_id"
}
```

#### Response

- **Success (200):**

```json
{
    "response": "model_response_here"
}
```

- **Error (500):**

```json
{
    "error": "Error message"
}
```

#### Example Curl Request

```bash
curl -X POST http://<your-server-domain>/pdf_chat -H "Content-Type: application/json" -d '{"query": "your_query_here", "user_id": "unique_user_id"}'
```

### 3. Delete Database

#### Endpoint

```
DELETE /delete_db
```

#### Description

Deletes the database associated with the given user ID.

#### Request

- **Headers:** `Content-Type: application/json`
- **Body:**

```json
{
    "user_id": "unique_user_id"
}
```

#### Response

- **Success (200):**

```json
{
    "detail": "Database deleted successfully."
}
```

- **Error (500):**

```json
{
    "error": "Error message"
}
```

#### Example Curl Request

```bash
curl -X DELETE http://<your-server-domain>/delete_db -H "Content-Type: application/json" -d '{"user_id": "unique_user_id"}'
```

### 4. Excel Chat

#### Endpoint

```
POST /excel_chat
```

#### Description

Processes a query on an Excel file and returns the response. The file is deleted after processing.

#### Request

- **Headers:** `Content-Type: application/json`
- **Body:**

```json
{
    "file_path": "path/to/your/file",
    "query": "your_query_here"
}
```

#### Response

- **Success (200):**

```json
{
    "response": "excel_response_here"
}
```

- **Error (500):**

```json
{
    "error": "Error message"
}
```

#### Example Curl Request

```bash
curl -X POST http://<your-server-domain>/excel_chat -H "Content-Type: application/json" -d '{"file_path": "path/to/your/file", "query": "your_query_here"}'
```

## Error Handling

Each endpoint returns a JSON object with an `error` key containing a message if an error occurs, and an HTTP status code of 500. Ensure that the request body is correctly formatted and all required fields are provided to avoid errors.

## Environment Variables

- `OPENAI_API_KEY`: The API key for accessing OpenAI services.

## Running the Application

To run the Flask application, use the following command:

```bash
python app.py
```

Make sure to set the environment variables in a `.env` file or export them before running the application.