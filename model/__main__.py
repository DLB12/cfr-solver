import logging
import pathlib

import polars as pl
import sklearn
from constants import RANDOM_SEED
from models import train_models
from utils import evaluate_performance

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


def extract_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assumptions:
    - No ordering among suits, e.g. spades is NOT preferred over hearts.
    - Card numbers: int[1, 52] -> [hearts, diamonds, clubs, spades] * [int[2, 14]]

    Findings:
    - Pure card numbers or suits & ranks do not offer predictive power.
    """
    df_features = df[:, :-3]

    # 1. compute suits and ranks
    df_cards = df_features.drop("History")

    suit_and_rank_exprs = []
    for col in df_cards.columns:
        # create suit column
        suit_col = f"{col}_suit"
        suit_expr = (
            pl.when(pl.col(col) == 0)
            .then(0)
            .otherwise((pl.col(col) - 1) // 13)
            .alias(suit_col)
        )

        # create rank column
        rank_col = f"{col}_rank"
        rank_expr = (
            pl.when(pl.col(col) == 0)
            .then(0)
            .otherwise((((pl.col(col) - 2) % 13) + 2))
            .alias(rank_col)
        )

        suit_and_rank_exprs.extend([suit_expr, rank_expr])

    df_suit_and_rank = df_cards.select(suit_and_rank_exprs)

    # 2. compute hole card interaction features
    df_interaction_features = df_suit_and_rank.select(
        [
            # ordering among ranks
            pl.max_horizontal("C1_rank", "C2_rank").alias("high_rank"),
            pl.min_horizontal("C1_rank", "C2_rank").alias("low_rank"),
            # rank gap
            (pl.col("C1_rank") - pl.col("C2_rank")).abs().alias("rank_gap"),
            # equal ranks
            (pl.col("C1_rank") == pl.col("C2_rank")).cast(pl.UInt8).alias("is_pair"),
            # same suit
            (
                (pl.col("C1_suit") == pl.col("C2_suit"))
                & (pl.col("C1_rank") > 0)
                & (pl.col("C2_rank") > 0)
            )
            .cast(pl.UInt8)
            .alias("is_suited"),
            # connected
            ((pl.col("C1_rank") - pl.col("C2_rank")).abs() == 1)
            .cast(pl.UInt8)
            .alias("is_connected"),
            # helpers for calculating Chen score
            (
                (pl.col("C1_rank") == pl.col("C2_rank")).cast(pl.UInt8)
                * pl.max_horizontal("C1_rank", "C2_rank")
                * 2
            ).alias("chen_pair_component"),
            (
                (pl.col("C1_rank") != pl.col("C2_rank")).cast(pl.UInt8)
                * (
                    pl.max_horizontal("C1_rank", "C2_rank")
                    + pl.min_horizontal("C1_rank", "C2_rank") / 3
                )
            ).alias("chen_nonpair_component"),
        ]
    )

    # Chen score (for evaluating starting hand)
    df_interaction_features = df_interaction_features.with_columns(
        [
            (
                pl.when(pl.col("is_pair") == 1)
                .then(pl.col("chen_pair_component"))
                .otherwise(pl.col("chen_nonpair_component"))
                + pl.col("is_suited") * 2
                + pl.when(pl.col("rank_gap") == 1)
                .then(1)
                .when(pl.col("rank_gap") == 2)
                .then(0.5)
                .otherwise(0)
            ).alias("chen_score")
        ]
    )

    print(df_interaction_features.head())
    return df_interaction_features


def extract_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    Assumptions:
    - Labels are real numbers in the interval [0, 1].
    - Sum of labels of each sample should be 1 - since they are probabilities.
    """
    return df[:, -3:]


def main() -> None:
    logger.info("=" * 120)
    logger.info("Initializing CFR Prediction Model")
    logger.info("=" * 120)

    # hard-coded data path
    data_path = pathlib.Path(".") / "strategy_output_frozen.csv"
    df_input = pl.read_csv(data_path)
    logger.info(f"Running on raw dataset of shape {df_input.shape}")

    # feature engineering
    x_all = extract_features(df_input)
    y_all = extract_labels(df_input)

    # train and test models
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x_all, y_all, test_size=0.2, random_state=RANDOM_SEED
    )
    models = train_models(x_train, y_train)
    for model in models:
        logger.info("=" * 60)
        logger.info(f"Result Metrics for {model.get_name()}")
        y_pred = model.predict(x_test)
        metrics = evaluate_performance(y_test, y_pred)
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name} = {metric_value}")

    logger.info("=" * 120)
    logger.info("Terminating CFR Prediction Model")
    logger.info("=" * 120)


if __name__ == "__main__":
    main()
