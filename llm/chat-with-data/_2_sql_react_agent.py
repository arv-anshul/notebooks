# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-ollama",
#     "langgraph",
#     "polars",
# ]
# ///

from typing import Any

import polars as pl
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent

llm = ChatOllama(model="mistral:7b", temperature=0)

# Reading DataFrame using polars
matches = pl.scan_csv(
    "https://www.kaggle.com/api/v1/datasets/download/patrickb1912/ipl-complete-dataset-20082020/matches.csv",
    ignore_errors=True,
)
sql_ctx = pl.SQLContext(matches=matches, eager=True)

system_message = f"""
    For given (first 5 rows) of SQL table (`matches`) you have to generate a SQL that
    to solves the user query in way that the executed query result will passed to new
    LLM to generate a natural language response for the user.

    DataFrame Head:
    {matches.head().collect().write_csv(separator="|")}

    - JUST GIVE ME THE SQL QUERY
    - WITHOUT USING "```" OR ANY TEXT.
    - GENERATE A CONCISE AND OPTIMIZED QUERY.
    - ALSO EXECUTE THE QUERY USING PROVIDED TOOL.
    - USE `LIMIT` TO LIMIT THE QUERY RESULT.
    - ALSO TRY TO HANDLE NORMAL CONVERSTION WITHOUT ANY TOOL CALLING.
    SQL Query:
    """


def sql_executor(sql: str) -> Any:
    """Execute SQL query using this tool."""
    return sql_ctx.execute(sql)


tools = ToolNode([sql_executor], name=sql_executor.__name__)

agent = create_react_agent(
    llm,
    tools,
    state_modifier=system_message,
    checkpointer=MemorySaver(),
)


def print_stream(agent, inputs):
    config = {"configurable": {"thread_id": "thread-1"}}
    for s in agent.stream(inputs, config, stream_mode="values"):
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


if __name__ == "__main__":
    import sys

    # run without specifying cli args
    # print_stream(agent, {"messages": "How many matches played by CSK?"})

    # read messages from cli arguments
    for msg in sys.argv[1:]:
        print_stream(agent, {"messages": msg})
