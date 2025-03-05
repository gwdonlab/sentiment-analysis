import os, uuid, json, sys
import datetime as dt
import psycopg2 as ps
import polars as pl
from getpass import getpass
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
from torch.cuda import empty_cache
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv()

# Machine/code parameters
CUDA_DEVICE = "cuda:0"
URL_REGEX = r"https?://\S+|www\.\S+"
GPU_BATCH_SIZE = 1080

# Connect to database
db_name = os.getenv("DB_NAME") if os.getenv("DB_NAME") is not None else input("DB Name: ")
username = os.getenv("DB_UNAME") if os.getenv("DB_UNAME") is not None else input("Username: ")
password = os.getenv("DB_PWD") if os.getenv("DB_PWD") is not None else getpass()
pg_conn = ps.connect(dbname=db_name, user=username, password=password)


def main(config):
    # Read database values that haven't yet been classified
    query = config["query"]

    if not os.path.exists(
        os.getenv("DATA_DIR") + config["output_parent_dir"] + "/" + config["run_id"]
    ):
        # Prep output directory
        os.mkdir(os.getenv("DATA_DIR") + config["output_parent_dir"] + "/" + config["run_id"])
        skip_these = ()
    else:
        # Skip previously-classified data for this run
        df_list = [
            pl.read_ipc(f, columns=config["id_col"])
            for f in os.listdir(
                os.getenv("DATA_DIR") + config["output_parent_dir"] + "/" + config["run_id"]
            )
        ]
        skip_these = tuple(pl.concat(df_list)[config["id_col"]].unique()) if len(df_list) else ()
        if len(skip_these):
            query += " and " + config["id_col"] + " not in %s"

    # We use the same VADER object for all data
    vader_analyzer = SentimentIntensityAnalyzer()

    # Ensure all tags are the same
    tag_standardizer = {"neu": "neutral", "pos": "positive", "neg": "negative"}

    with pg_conn.cursor() as cursor:
        # Get first batch to start iteration
        cursor.execute(query, (skip_these,)) if len(skip_these) else cursor.execute(query)
        results = cursor.fetchmany(config["fetch_size"])

        # Stop iterating when there are no more database results
        while len(results) > 0:

            df = pl.DataFrame(
                results, schema=[config["id_col"], config["text_col"]], orient="row"
            ).with_columns(pl.col(config["text_col"]).str.replace_all(URL_REGEX, "").alias("text"))
            ds = Dataset.from_polars(df.drop(config["text_col"]))

            # Classify rows
            v_class, v_score = [], []
            for row in tqdm(results, desc="VADER"):
                v_results = vader_analyzer.polarity_scores(row[-1])
                if v_results["compound"] >= 0.05:
                    v_class.append("positive")
                elif v_results["compound"] <= -0.05:
                    v_class.append("negative")
                else:
                    v_class.append("neutral")
                v_score.append(v_results["compound"])

            # Run CardiffNLP's sentiment analyzer on this database batch
            cardiff_analyzer = pipeline(
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                max_length=512,
                truncation=True,
                device=CUDA_DEVICE,
            )
            c_class, c_score = [], []
            for result in tqdm(
                cardiff_analyzer(KeyDataset(ds, "text"), batch_size=GPU_BATCH_SIZE),
                total=len(ds),
                desc="CardiffNLP",
            ):
                c_class.append(result["label"].lower())
                c_score.append(result["score"])
            del cardiff_analyzer
            empty_cache()

            # Run Siebert's sentiment analyzer on this database batch
            siebert_analyzer = pipeline(
                model="siebert/sentiment-roberta-large-english",
                max_length=512,
                truncation=True,
                device=CUDA_DEVICE,
            )
            s_class, s_score = [], []
            for result in tqdm(
                siebert_analyzer(KeyDataset(ds, "text"), batch_size=GPU_BATCH_SIZE),
                total=len(ds),
                desc="Siebert",
            ):
                s_class.append(result["label"].lower())
                s_score.append(result["score"])
            del siebert_analyzer
            empty_cache()

            # Run finiteautomata's BERTweet sentiment analyzer on this database batch
            bertweet_analyzer = pipeline(
                model="finiteautomata/bertweet-base-sentiment-analysis",
                max_length=128,
                truncation=True,
                device=CUDA_DEVICE,
            )
            f_class, f_score = [], []
            for result in tqdm(
                bertweet_analyzer(KeyDataset(ds, "text"), batch_size=GPU_BATCH_SIZE),
                total=len(ds),
                desc="BERTweet",
            ):
                f_class.append(tag_standardizer[result["label"].lower()])
                f_score.append(result["score"])
            del bertweet_analyzer
            empty_cache()

            # Output results
            df.with_columns(
                vader_classification=pl.Series(v_class),
                vader_score=pl.Series(v_score),
                cardiff_classification=pl.Series(c_class),
                cardiff_score=pl.Series(c_score),
                siebert_classification=pl.Series(s_class),
                siebert_score=pl.Series(s_score),
                bertweet_classification=pl.Series(f_class),
                bertweet_score=pl.Series(f_score),
            ).with_columns(
                pl.struct(
                    "vader_classification",
                    "cardiff_classification",
                    "siebert_classification",
                    "bertweet_classification",
                )
                .map_elements(final_score, return_dtype=str)
                .alias("majority_class"),
                pl.lit(dt.datetime.now()).alias("predicted_at"),
            ).drop(
                [config["text_col"], "text"]
            ).write_ipc(
                os.getenv("DATA_DIR")
                + config["output_parent_dir"]
                + "/"
                + config["run_id"]
                + "/sentiments_"
                + str(uuid.uuid4())
                + ".ft"
            )

            del df

            # If this returns nothing, the loop will break
            results = cursor.fetchmany(config["fetch_size"])


def final_score(row):
    """
    Helper function to pool classifications per row
    The row's classification is the majority
    """
    c = Counter(
        [
            row["vader_classification"],
            row["cardiff_classification"],
            row["siebert_classification"],
            row["bertweet_classification"],
        ]
    )
    if c.most_common(1)[0][1] > 1:
        return c.most_common(1)[0][0]
    else:
        # TODO: Improve logic for this:
        # With four classifiers and three outputs, this should be impossible
        return "TIE"


if __name__ == "__main__":
    with open(sys.argv[1], "r") as infile:
        run_config = json.load(infile)
    assert run_config["id_col"] in run_config["query"]
    assert run_config["text_col"] in run_config["query"]
    main(run_config)
