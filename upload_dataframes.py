import os, json, sys
import psycopg2 as ps
import polars as pl
from getpass import getpass
from tqdm import tqdm
from dotenv import load_dotenv
from psycopg2.extras import execute_values

# Load environment variables
load_dotenv()

# Connect to database
db_name = os.getenv("DB_NAME") if os.getenv("DB_NAME") is not None else input("DB Name: ")
username = os.getenv("DB_UNAME") if os.getenv("DB_UNAME") is not None else input("Username: ")
password = os.getenv("DB_PWD") if os.getenv("DB_PWD") is not None else getpass()
pg_conn = ps.connect(dbname=db_name, user=username, password=password)


def main(config):
    # Loop through data in feather files
    for f in tqdm(
        os.listdir(os.getenv("DATA_DIR") + config["output_parent_dir"] + "/" + config["run_id"])
    ):
        df = (
            pl.read_ipc(
                os.getenv("DATA_DIR")
                + config["output_parent_dir"]
                + "/"
                + config["run_id"]
                + "/"
                + f
            )
            .rename(
                {
                    "vader_classification": "vader_class",
                    "cardiff_classification": "cardiff_class",
                    "siebert_classification": "siebert_class",
                    "bertweet_classification": "bertweet_class",
                }
            )
            .with_columns(pl.lit(config["run_id"]).alias("run_id"))
        )

        # Update boolean column in main table
        with pg_conn.cursor() as cursor:
            try:
                ids = tuple(df[config["id_col"]].to_list())
                table_name = config["query"].split(" from ")[1].split(" where ")[0]
                cursor.execute(
                    "update "
                    + table_name
                    + " set sentiment_complete = true where "
                    + config["id_col"]
                    + " in %s",
                    (ids,),
                )
                pg_conn.commit()
            except (Exception, ps.DatabaseError) as error:
                print(error)
                pg_conn.rollback()
                exit()

        # Tuple upload via psycopg2
        tuples = [tuple(x) for x in df.to_numpy()]
        cols = ",".join(list(df.columns))
        with pg_conn.cursor() as cursor:
            # In the future, we may choose a different "ON CONFLICT" action
            query = (
                "insert into sentiment_analysis(%s) values " % cols + "%s on conflict do nothing"
            )

            try:
                execute_values(cursor, query, tuples)
                pg_conn.commit()
            except ps.DatabaseError as error:
                print("[POSTGRESQL ERROR]", error)
                pg_conn.rollback()
                exit()


if __name__ == "__main__":
    with open(sys.argv[1], "r") as infile:
        run_config = json.load(infile)
    assert run_config["id_col"] in run_config["query"]
    assert run_config["text_col"] in run_config["query"]
    main(run_config)
