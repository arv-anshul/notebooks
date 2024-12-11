# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-experimental",
#     "langchain-ollama",
#     "pandas",
#     "rich",
#     "tabulate",
# ]
# ///

from operator import itemgetter
from typing import Any

import pandas as pd
import rich
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_experimental.tools import PythonAstREPLTool
from langchain_ollama import ChatOllama

LLM_MODEL = "mistral:7b"

print("Downloading dataset...")
matches = pd.read_csv(
    # TODO: if you are running this multiple times then you should read csv file
    # from local path. it will be faster...
    "https://www.kaggle.com/api/v1/datasets/download/patrickb1912/ipl-complete-dataset-20082020/matches.csv",
    usecols=[
        "season",
        "city",
        "player_of_match",
        "winner",
        "team1",
        "team2",
        "target_runs",
    ],  # type: ignore
)
print("Dataset downloaded.")


def code_executor(code: str) -> Any:
    pandas_repl = PythonAstREPLTool(globals={"df": matches, "pd": pd})
    exec_result = pandas_repl.invoke(code)
    return exec_result


code_gen_prompt = ChatPromptTemplate.from_template(
    """
    For given pandas dataframe head you have to generate a python that to solve the
    user query in way that the executed code result will passed to new LLM to generate a
    natural language response for the user.

    DataFrame Head:
    {df_head}

    User Query: {query}

    - JUST GIVE ME THE PANDAS CODE
    - WITHOUT USING "```" OR ANY TEXT.
    - YOU HAVE ONLY `df` AND `pd` VARIABLE TO DEAL WITH.
    - CODE MUST NOT ANY VARIABLE ASSISGNMENT.
    - GENERATE A CONCISE AND OPTIMIZED CODE.
    - CODE MUST BE IN ONE LINE
    (RETURN ONE LINE CODE) Pandas Code:
    """,
)


llm = ChatOllama(model=LLM_MODEL, temperature=0.3)

code_gen_chain = (
    code_gen_prompt | llm | StrOutputParser() | str.strip
    # we can add pandas_repl here to run the generated code and get the result
    # but restricting it due for debugging purpose only
    # | pandas_repl.run
)

response_gen_prompt = ChatPromptTemplate.from_template(
    """
    Generate a nice markdown text response, focused around IPL (Indian Premier League)
    tournament by analysing user query and query's response.

    User Query: {query}
    Query Result: {query_result}

    - Don't include any numbers around IPL or anything beacause it maybe changed
      or misleading to user.
    - Only use numbers from Query Result.
    - Keep the response concise and professional.
    - If the code excution throws any error just return a small apollogy response.
    """,
)

# It is possible to shorten this chain by removing unusual nodes
# but I'm not removing due to debugging
response_gen_chain = (
    RunnableParallel(
        query=itemgetter("query"),
        code=code_gen_chain,
    )
    | {
        "query": itemgetter("query"),
        "code": itemgetter("code"),
        "query_result": lambda x: code_executor(x["code"]),
    }
    | {
        "code": itemgetter("code"),
        "query_result": itemgetter("query_result"),
        "response": response_gen_prompt | llm | StrOutputParser(),
    }
)

response = response_gen_chain.invoke(
    {
        # "query": "how many rows are there?",
        # "query": "how many matches were played in whole IPL?",
        "query": "how many matches were played by daredevils?",
        # "query": "how many matches were played by daredevils (include rebranded team name too)?",
        # "query": "venue where most no. of matches were played.",
        # "query": "which team is won with largest margin?",
        "df_head": matches.head().to_markdown(),
    },
)

rich.print(response)
