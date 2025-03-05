# Sentiment Analysis

Simple scripts to perform bulk sentiment analysis using an ensemble approach

## [`classify_db.py`](./classify_db.py)

This script expects one CLI arg that is a path to a JSON file with the following schema:

```JSON
{
    "query": "string; a SQL query selecting an ID and a text column",
    "fetch_size": "int; row count of output dataframes",
    "run_id": "string; unique identifier for the run",
    "output_parent_dir": "string; parent directory in which to create run folder",
    "id_col": "string; ID column in DB",
    "text_col":"string; text column in DB"
}
```