import pandas as pd
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import langchain
from typing import Union

load_dotenv()

langchain.debug = True


class ExcelBot:
    def __init__(
        self, file_path: str, api_key: str, sheet: Union[str, int] = 0
    ) -> None:
        self.df: pd.DataFrame = self.load_excel_skip_empty_rows(
            file_path, sheet_name=sheet
        )
        self.refine_llm = ChatOpenAI(model="gpt-4o", temperature=0.01, api_key=api_key)
        self.chat_llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)
        self.llm_with_tools = self.chat_llm.bind_tools(
            [self.generate_pandas_code, self.execute_pandas_code]
        )
        self.column_value_pairs, self.column_list, self.sample_data = (
            self.create_metadata()
        )
        self.system_prompt = """
    #### Context:
    You are a query refinement system designed to interpret and refine user queries based on data from an Excel file. You will receive:
    1. Metadata about the data in the Excel file.
    2. The first five rows of the dataframe.
    3. A natural language query from the user.

    #### Metadata:
    - **Columns:** A list of all column names in the dataframe.
    - **Categorical Columns-Values Pairs:** A dictionary where the keys are column names with less than 15 unique values, and the values are lists of those unique values.
    - **First Five Rows of Data:** The first five rows as sample of the data. Output from df.head()

    #### Your Task:
    1. **Understand the User Query:** Interpret the user's natural language query in the context of the provided metadata and data sample.
    2. **Consider Metadata:**
    - Use the list of columns to understand the scope of the data.
    - Use the unique values dictionary to identify categorical columns with limited unique values that might be relevant for the query.
    3. **Refine the Query:** Translate the natural language query into a refined, structured query that clearly specifies:
    - Relevant columns to consider.
    - Any filters or conditions based on the unique values.
    - The type of information or analysis the user is seeking.

    #### Output Format:
    Your response should be a refined query in a structured format that can be directly used by the next component to retrieve or analyze the data. The format should be clear and unambiguous.

    #### Examples:
    - **User Query:** "What are the sales figures for each region?"
    **Refined Query:** "Select 'Region', 'Sales' columns. Get sales figures for each unique value in 'Region'."

    - **User Query:** "Show me the details of the products with sales above 1000 units."
    **Refined Query:** "Filter rows where 'Sales' > 1000. Select all columns for the filtered rows."

    #### Constraints:
    - Ensure clarity and specificity in the refined query.
    - Include any necessary conditions or filters explicitly.
    - Consider all relevant columns and unique values in the metadata.
    - Consider some external data like Zip Codes, State Abbreviations, Symbolic state names, etc and refine the query including this data, ONLY IF REQUIRED.
    - ADHERE TO THE FORMAT OF OUTPUT SHOWN IN THE ABOVE EXAMPLES. DO NOT add any instruction or any extra text to the output.
    
    #### Begin Processing:
    Refine the user query based on the above instructions.
    """
        self.human_prompt = (
            f"""
    #### Metadata:
    Columns: {self.column_list}
    Column-Value Pairs: {self.column_value_pairs}
    Sample Data: {self.sample_data}
    """
            + """
    #### User Query:
    """
        )
        self.prompt_template = "System:\n" + self.system_prompt + self.human_prompt

    def load_excel_skip_empty_rows(
        self, file_path: str, sheet_name: Union[str, int] = 0
    ) -> pd.DataFrame:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Drop initial empty rows
        df = df.dropna(how="all").reset_index(drop=True)

        # Set the new column names
        df.columns = df.iloc[0]

        # Drop the row used for column names
        df = df[1:].reset_index(drop=True)

        return df

    def create_metadata(self):
        # Dictionary to hold columns with 15 or fewer unique values
        column_value_pairs = {}

        # Iterate through each column in the DataFrame
        for column in self.df.columns:
            unique_values = self.df[column].nunique()
            if unique_values <= 15:
                column_value_pairs[column] = self.df[column].unique().tolist()

        return str(column_value_pairs), str(list(self.df.columns)), str(self.df.head())

    def refine_query(self, query):
        prompted_query = self.prompt_template + query + "#### Refined Query:"
        self.refined_query = self.refine_llm.invoke(prompted_query)
        return self.refined_query.content

    # Function to generate pandas code
    def generate_pandas_code(self, query: str) -> str:
        """Generate pandas code based on the query."""
        # Implement logic to generate pandas code here
        # For demonstration purposes, return a simple code snippet
        return f"result = df.head()  # Generated code for query: {query}"

    # Function to execute pandas code
    def execute_pandas_code(self, code: str) -> str:
        """Execute the given pandas code and return the result."""
        try:
            exec_locals = {"df": self.df}
            exec(code, {}, exec_locals)
            return exec_locals.get("result", "Code executed successfully")
        except Exception as e:
            return str(e)

    def is_query_valid(self, refined_query: str) -> bool:
        # Check if the refined query references columns in the DataFrame
        for column in self.df.columns:
            if column in refined_query:
                return True
        return False

    def excel_invoke(self, query: str):
        # Create the agent using the ReAct technique
        refined_query = self.refine_query(query)
        print(refined_query)

        # Check if the refined query is valid
        if not self.is_query_valid(refined_query):
            return "I don't know the answer to that question."

        MODIFIED_PREFIX = """
        You are working with a pandas dataframe in Python. The name of the dataframe is `df`. You will be given the first five rows of the dataframe for reference. Do not include the sample data in the output code."""

        FUNCTIONS_WITH_DF="""
        This is the result of `print(df.head())`:
        {df_head}"""
        agent = create_pandas_dataframe_agent(
            llm=self.llm_with_tools,
            df=self.df,
            agent_type="tool-calling",
            include_df_in_prompt=True,
            prefix=MODIFIED_PREFIX,
            suffix=FUNCTIONS_WITH_DF
        )

        # Generate reasoning steps and action (pandas code) iteratively
        response = agent.invoke({"input": refined_query})
        return response["output"]
