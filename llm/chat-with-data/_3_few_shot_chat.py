# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "langchain-chroma==0.1.4",
#     "langchain-core==0.3.24",
#     "langchain-ollama==0.2.1",
#     "marimo",
#     "polars==1.17.1",
# ]
# ///

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
    # Chat With IPL Data

    Inspired from Blog:
    [**Mastering Natural Language to SQL with LangChain | NL2SQL**](https://blog.futuresmart.ai/mastering-natural-language-to-sql-with-langchain-nl2sql)
    """,
    ).center()


@app.cell
def _(mo):
    mo.md(
        f"""
    ## Application Components

    1. **LLM:** Using `OllamaChat` class to interact with local models.
    2. **SQL Query Generator _(Node)_:** To generate a SQL query using **user's questions** to get required data from dataset after seeing dataset schema.
    3. **SQL Query Executor _(Tool)_:** To execute the generated **SQL query** on the available dataset. For this, we are using [`polars.SQLContext`](https://docs.pola.rs/user-guide/sql/intro/) object.
    4. **Final Response Generator _(Prompt)_:** For LLM to generate a better response by processing **user question and SQL query result**.

    ### Additional Components

    1. **Memory & VectorDatabase:** To store **user's questions and generated SQL queries** which will used to feed the LLM in later user questions from which LLM can understand the question and generate a better SQL query for the asked question.
    2. **FewShotChatPromptTemplate:** To avail the features of previous point we need to use this object. [See docs](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate.html).

    {mo.image("llm/chat-with-data/assets/_3_diagram.png", "diagram")}
        """,
    )


@app.cell
def _():
    import shutil
    from operator import itemgetter

    import polars as pl
    from langchain_chroma import Chroma
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import (
        ChatPromptTemplate,
        FewShotChatMessagePromptTemplate,
        PromptTemplate,
    )
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    import marimo as mo

    return (
        ChatOllama,
        ChatPromptTemplate,
        Chroma,
        FewShotChatMessagePromptTemplate,
        OllamaEmbeddings,
        PromptTemplate,
        RunnableLambda,
        RunnablePassthrough,
        SemanticSimilarityExampleSelector,
        StrOutputParser,
        itemgetter,
        mo,
        pl,
        shutil,
    )


@app.cell
def _(pl):
    matches = pl.scan_csv(
        "https://www.kaggle.com/api/v1/datasets/download/patrickb1912/ipl-complete-dataset-20082020/matches.csv",
        ignore_errors=True,
    ).select(
        "season",
        "city",
        "venue",
        "player_of_match",
        "team1",
        "team2",
        "toss_winner",
        "toss_decision",
        "target_runs",
        "winner",
    )

    sql_ctx = pl.SQLContext(matches=matches, eager=True)
    return matches, sql_ctx


@app.cell
def _(matches, mo):
    mo.accordion(
        {"Take a glimpse of IPL Matches table.": matches.head(10).collect()},
        lazy=True,
    )


@app.cell
def _():
    examples = [
        {
            "question": "How many matches were played in IPL season 2007?",
            "query": "SELECT COUNT(*) as 'Match Count' FROM matches WHERE season LIKE '2007/08';",
        },
        {
            "question": "How many matches daredevils won?",
            "query": "SELECT COUNT(*) as 'Winning Count' FROM matches WHERE winner = 'Delhi Daredevils';",
        },
        {
            "question": "How many matches KKR played in whole IPL?",
            "query": "SELECT COUNT(*) as 'Match Count' FROM matches WHERE team1 = 'Kolkata Knight Riders' OR team2 = 'Kolkata Knight Riders';",
        },
        {
            "question": "How many times Virat Kohli became POTM?",
            "query": "SELECT COUNT(*) as 'POTM Count' FROM matches WHERE player_of_match = 'V Kohli';",
        },
        {
            "question": "How many times Dhoni became POTM? Compare with Kohli.",
            "query": "SELECT COUNT(CASE WHEN player_of_match = 'MS Dhoni' THEN 1 END) as 'Dhoni POTM Count', COUNT(CASE WHEN player_of_match = 'V Kohli' THEN 1 END) as 'Kohli POTM Count' FROM matches;",
        },
    ]
    return (examples,)


@app.cell
def _(
    Chroma,
    OllamaEmbeddings,
    SemanticSimilarityExampleSelector,
    examples,
    shutil,
):
    chroma_vectorstore_dir = "chromadb.arv"

    # Delete all chromadb files before insterting examples
    shutil.rmtree(chroma_vectorstore_dir, True)

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OllamaEmbeddings(model="mistral:7b"),
        Chroma,
        k=2,
        input_keys=["question"],
        # vectorstore class kwargs
        persist_directory=chroma_vectorstore_dir,
    )
    return chroma_vectorstore_dir, example_selector


@app.cell
def _(
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    example_selector,
):
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages(
            [("user", "{question}"), ("ai", "{query}")],
        ),
        example_selector=example_selector,
        input_variables=["question"],
    )
    return (few_shot_prompt,)


@app.cell
def _(mo):
    model_selector = mo.ui.dropdown(["llama3:8b", "mistral:7b"], "mistral:7b")
    return (model_selector,)


@app.cell
def _(ChatOllama, model_selector):
    llm = ChatOllama(model=model_selector.value, temperature=0, verbose=True)
    return (llm,)


@app.cell
def _(pl, sql_ctx):
    from polars.exceptions import PolarsError

    def sql_executor(_sql: str) -> str:
        try:
            result = sql_ctx.execute(_sql)
        except PolarsError as e:
            return "Got an error:\n" + str(e)
        with pl.Config(tbl_formatting="MARKDOWN", tbl_hide_dataframe_shape=True):
            print("After executing SQL Query:\n", result)
            return str(result)

    return PolarsError, sql_executor


@app.cell
def _(ChatPromptTemplate, StrOutputParser, few_shot_prompt, llm):
    _system_message = """
    For given (first 5 rows) of SQL table (`matches`) you have to generate a SQL that to solves the user question.

    DataFrame Head:
    {df_head}

    - JUST GIVE ME THE SQL QUERY
    - WITHOUT USING "```" OR MARKDOWN FORMATTING.
    - ALWAYS USE `LIMIT` TO RESTRICT THE RESULT TO TOP 7.
    - GENERATE A CONCISE AND OPTIMIZED QUERY AFTER SEEING EXAMPLE PROMPTS.

    (NO TEXT, ONLY) SQL Query:"""

    def refine_sql_query(s: str) -> str:
        print("Generated SQL Query:", s)
        return s.strip().split(";\n", 1)[0]

    prompt = ChatPromptTemplate.from_messages(
        [("system", _system_message), few_shot_prompt, ("user", "{question}")],
    )

    # I have tried `sqlcoder:7b` model but didn't got response as good as base LLMs
    # sqlcoder = ChatOllama(model="sqlcoder:7b", temperature=0, verbose=True)
    # I think that required different type of prompt or else...

    generate_query = prompt | llm | StrOutputParser() | refine_sql_query
    return generate_query, prompt, refine_sql_query


@app.cell
def _(PromptTemplate, StrOutputParser, llm):
    answer_prompt = PromptTemplate.from_template(
        """
    Generate a response for user according to their question around IPL data.
    If SQL result is a error, then just tell that you don't know the answer.

    Question: {question}
    SQL Result:
    {result}

    (CONCISE) Answer:""",
    )
    rephrase_answer = answer_prompt | llm | StrOutputParser() | str.strip
    return answer_prompt, rephrase_answer


@app.cell
def _(
    RunnableLambda,
    RunnablePassthrough,
    generate_query,
    itemgetter,
    rephrase_answer,
    sql_executor,
):
    final_chain = (
        RunnablePassthrough.assign(query=generate_query)
        .assign(result=itemgetter("query") | RunnableLambda(sql_executor))
        .assign(output=rephrase_answer)
    )
    return (final_chain,)


@app.cell
def _(mo, model_selector):
    question_input_widget = mo.ui.text(
        placeholder="Query IPL data in natural language...",
        full_width=True,
    )

    # Who has taken most potm in whole IPL seasons?
    mo.md(f"""
    **Select Model:** {model_selector}

    **Ask Question:** _(press enter)_ {question_input_widget}
    """).callout().style(min_width="800px").center()
    return (question_input_widget,)


@app.cell
def _(final_chain, matches, mo, question_input_widget):
    # This stop the exceution if the widget has no value
    mo.stop(not question_input_widget.value)

    with mo.status.spinner("Processing..."):
        response = final_chain.invoke(
            {
                "question": question_input_widget.value,
                "df_head": matches.head().collect().write_csv(separator="|"),
            },
        )
    return (response,)


@app.cell
def _(mo, response):
    mo.md(
        """
    ## Processing Details

    - **Your Question:** {question}
    - **SQL Query generated by LLM to tackle this question:**

    ```sql
    {query}
    ```

    - **Result after execution of SQL Query:**

    ```md
    {result}
    ```

    ### Final Response

    {output}
    """.format(**response),
    )


if __name__ == "__main__":
    app.run()
