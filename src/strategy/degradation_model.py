import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = Path("data/raw")


def load_all_laps():
    """
    Load laps from all seasons & races
    """
    all_laps = []

    for season_dir in RAW_DATA_DIR.iterdir():
        if not season_dir.is_dir():
            continue

        if not season_dir.name.isdigit():
            continue  # skip fastf1_cache

        for race_dir in season_dir.iterdir():
            race_session = race_dir / "R"
            laps_path = race_session / "laps.parquet"

            if not laps_path.exists():
                continue

            laps = pd.read_parquet(laps_path)
            laps["season"] = int(season_dir.name)
            laps["race_name"] = race_dir.name

            all_laps.append(laps)

    if not all_laps:
        raise ValueError("No lap data found")

    return pd.concat(all_laps, ignore_index=True)



def prepare_degradation_data(laps: pd.DataFrame) -> pd.DataFrame:
    laps = laps.copy()

    # Convert lap time to seconds
    laps["LapTimeSeconds"] = (
        laps["LapTime"]
        .dt.total_seconds()
        .astype(float)
    )

    # Remove invalid laps
    laps = laps.dropna(
        subset=["LapTimeSeconds", "Compound", "Stint"]
    )

    # Lap number within each stint
    laps["lap_in_stint"] = (
        laps.groupby(["Driver", "Stint"])
        .cumcount() + 1
    )

    return laps


def train_degradation_models(laps: pd.DataFrame):
    """
    Train one regression model per tire compound
    """
    models = {}

    for compound in ["SOFT", "MEDIUM", "HARD"]:
        dfc = laps[laps["Compound"] == compound]

        if dfc.empty:
            logger.warning(f"No data for compound: {compound}")
            continue

        X = dfc[["lap_in_stint"]]
        y = dfc["LapTimeSeconds"]

        model = LinearRegression()
        model.fit(X, y)

        models[compound] = {
            "model": model,
            "intercept": model.intercept_,
            "slope": model.coef_[0],
            "n_samples": len(dfc)
        }

        logger.info(
            f"{compound} | samples={len(dfc)} | "
            f"slope={model.coef_[0]:.4f}s/lap"
        )

    return models



def build_degradation_models():
    """
    End-to-end training pipeline
    """
    logger.info("Loading lap data...")
    laps = load_all_laps()

    logger.info("Preparing degradation features...")
    laps = prepare_degradation_data(laps)

    logger.info("Training degradation models...")
    models = train_degradation_models(laps)

    return models



if __name__ == "__main__":
    models = build_degradation_models()

    for comp, info in models.items():
        print(
            f"{comp}: intercept={info['intercept']:.2f}, "
            f"slope={info['slope']:.4f}"
        )
