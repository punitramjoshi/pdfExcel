import pandas as pd
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import langchain
from dotenv import load_dotenv

load_dotenv()

langchain.debug = True


# Function to generate pandas code
def generate_pandas_code(query: str) -> str:
    """Generate pandas code based on the query."""
    # Implement logic to generate pandas code here
    # For demonstration purposes, return a simple code snippet
    return f"result = df.head()  # Generated code for query: {query}"


# Function to execute pandas code
def execute_pandas_code(df: pd.DataFrame, code: str) -> str:
    """Execute the given pandas code and return the result."""
    try:
        exec_locals = {"df": df}
        exec(code, {}, exec_locals)
        return exec_locals.get("result", "Code executed successfully")
    except Exception as e:
        return str(e)


# Create the prompt template using ReAct format
react_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant skilled in using pandas with ReAct."),
        ("user", "{input}"),
        ("assistant", "Let's think step by step."),
        (
            "assistant",
            "First, I need to understand what data manipulation is required.",
        ),
        (
            "assistant",
            "Based on the query and the first five rows of the dataframe, I will generate the appropriate pandas code.",
        ),
        (
            "assistant",
            "I will remember that the rows are only an example of the dataframe, not the whole dataframe",
        ),
        ("assistant", "Next, I will execute this code and return the result."),
    ]
)


# Function to create the agent with ReAct technique
def create_react_agent(df: pd.DataFrame, api_key: str):
    # Define the OpenAI model
    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        agent_type="tool-calling",
        include_df_in_prompt=True,
        prompt_template=react_prompt_template,
    )
    return agent


def excel_invoke(df: pd.DataFrame, query: str, api_key: str):
    # Create the agent using the ReAct technique
    agent = create_react_agent(df, api_key)

    # Generate reasoning steps and action (pandas code) iteratively
    response = agent.invoke({"input": query})
    return response["output"]
