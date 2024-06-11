import pandas as pd
from langchain_experimental.agents.agent_toolkits.pandas.base import (
    create_pandas_dataframe_agent,
)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import langchain
from typing import Union
from langchain.agents.agent_types import AgentType
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.helpers.openai_info import get_openai_callback
import pandas as pd
import re

load_dotenv()

langchain.debug = True


class ExcelBot:
    def __init__(
        self, file_path: str, api_key: str, sheet_name: Union[str, int] = 0
    ) -> None:
        self.df:pd.DataFrame = self.load_excel_file(file_path, sheet_name=sheet_name)
        self.clean_df:pd.DataFrame = self.clean_dataframe_columns(self.df)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
        self.column_value_pairs, self.column_list, self.sample_data = (
            self.create_metadata()
        )
        self.smart_df = SmartDataframe(self.clean_df,config={"llm": self.llm, "conversational": False})
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
    - Convert the US state names given in the query into the corresponding zip codes, if the data is containing zip codes.

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

    def load_excel_file(self, file_path: str, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        # Load the file to inspect the first few rows
        if ".csv" in file_path:
            temp_df = pd.read_csv(file_path, header=None)
        else:
            temp_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        
        # Drop initial empty rows
        temp_df = temp_df.dropna(how="all").reset_index(drop=True)

        # Check if the first two rows contain strings to determine if it has a multi-level header
        if len(temp_df) > 1:
            first_row_is_str = all(isinstance(i, str) or pd.isna(i) for i in temp_df.iloc[0])
            second_row_is_str = all(isinstance(i, str) or pd.isna(i) for i in temp_df.iloc[1])
        else:
            first_row_is_str = False
            second_row_is_str = False
        
        if first_row_is_str and second_row_is_str:
            print("first,secondrow")
            if ".csv" in file_path:
                df = pd.read_csv(file_path, header=[0, 1])
            else:
                # If the first two rows are strings (or NaNs in the first row), it's a multi-level header
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=[0, 1])
            # Flatten the multi-level column headers into a single level
            df.columns = [f"{str(col[0])} {col[1]}".strip() if pd.notna(col[0]) else col[1] for col in df.columns.values]
        else:
            # If not, use the first row as the header
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

        # Drop initial empty rows (redundant if already done above, but keeping for completeness)
        df = df.dropna(how="all").reset_index(drop=True)

        return df

    def clean_dataframe_columns(self,df:pd.DataFrame) -> pd.DataFrame:
        def clean_column_name(col):
            # Use regex to remove 'Unnamed: ..._level_...' patterns
            cleaned_col = re.sub(r'Unnamed: \d+_level_\d+', '', col).strip()
            return cleaned_col if cleaned_col else 'Unnamed'
        
        # Apply the cleaning function to each column name
        cleaned_columns = [clean_column_name(col) for col in df.columns]
        
        # Rename the columns in the DataFrame
        df.columns = cleaned_columns
        
        return df

    def create_metadata(self):
        # Dictionary to hold columns with 15 or fewer unique values
        column_value_pairs = {}

        # Iterate through each column in the DataFrame
        for column in self.clean_df.columns:
            unique_values = self.clean_df[column].nunique()
            if unique_values <= 15:
                column_value_pairs[column] = self.clean_df[column].unique().tolist()

        return str(column_value_pairs), str(list(self.clean_df.columns)), str(self.clean_df.head())

    def refine_query(self, query):
        prompted_query = self.prompt_template + query + "#### Refined Query:"
        self.refined_query = self.llm.invoke(prompted_query)
        return self.refined_query.content

    def is_query_valid(self, refined_query: str) -> bool:
        # Check if the refined query references columns in the DataFrame
        for column in self.clean_df.columns:
            if column.strip() in refined_query:
                return True
        return False

    def excel_invoke(self, query:str):
        refined_query = self.refine_query(query)
        print('inexcelinvoke')
        # result = self.is_query_valid(refined_query)
        
        # if not result:
        #     return "I don't know the answer to that question."
        response = self.smart_df.chat(refined_query)
        print(response)
        return response
    # def excel_invoke(self, query: str):
    #     # Create the agent using the ReAct technique
    #     refined_query = self.refine_query(query)
    #     print(refined_query)

    #     result = self.is_query_valid(refined_query)
    #     print("This is the resut",result)
    #     # Check if the refined query is valid
    #     if not result:
    #         return "I don't know the answer to that question."
        
    #     PREFIX = """
    #     You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Remember, the final output should include answer in natural language, not a Python code.
    #     """
    #     agent = create_pandas_dataframe_agent(
    #         llm=self.llm,
    #         df=self.df,
    #         agent_type=AgentType.OPENAI_FUNCTIONS,
    #         include_df_in_prompt=True,
    #         prefix=PREFIX,
    #         agent_executor_kwargs={"handle_parsing_errors": True}
    #     )

    #     # Generate reasoning steps and action (pandas code) iteratively
    #     response = agent.invoke({"input": refined_query})
    #     return response["output"]
