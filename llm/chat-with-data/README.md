# Chat with IPL Dataset

As I am learning some applications of LLMs, I am also trying to built a **Chat With Data** application with LangChain
and LangGraph like tools using LLM.

## Scripts

### `_1_query_pandas_df.py`

Query your pandas dataframe using LLM.

#### Usage

```bash
uv run _1_query_pandas_df.py
```

#### Approach

- Used IPL dataset from kaggle. (https://kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- Used Ollama to interact with LLMs.
- Used `PythonAstREPLTool` to excecute generated pandas code.
- The resultant answer then passes to another LLM to form a better response for user.

### `_2_sql_react_agent.py`

Query csv file using SQL statements.

#### Usage

```bash
uv run _2_sql_react_agent.py \
  "How many matches played by CSK?" \
  "How many they won out of them?" \
```

#### Process

- Used IPL dataset from kaggle. (https://kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020)
- Polars library to query a `pl.LazyFrame` object with SQL statements using `pl.SQLContext` object.
  - See docs: https://docs.pola.rs/user-guide/sql/intro
- Used [`create_react_agent`](https://langchain-ai.github.io/langgraph/reference/prebuilt/)
  from LangGraph library for this task.
- Tried to implement memory feature.
