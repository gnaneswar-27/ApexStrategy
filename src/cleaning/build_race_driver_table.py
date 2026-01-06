import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed/race_driver_level")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_race_sessions():
    """
    Generator that yields:
    season, race_name, laps_df, results_df
    """
    for season_dir in RAW_DATA_DIR.iterdir():
        if not season_dir.is_dir():
            continue

        if not season_dir.name.isdigit():
            logger.info(f"Skipping non-season folder: {season_dir.name}")
            continue

        season = int(season_dir.name)


        for race_dir in season_dir.iterdir():
            race_name = race_dir.name
            race_session = race_dir / "R"

            laps_path = race_session / "laps.parquet"
            results_path = race_session / "results.parquet"

            if not laps_path.exists() or not results_path.exists():
                logger.warning(f"Missing data for {season} | {race_name}")
                continue

            laps = pd.read_parquet(laps_path)
            results = pd.read_parquet(results_path)

            yield season, race_name, laps, results


def build_lap_features(laps: pd.DataFrame) -> pd.DataFrame:
    laps = laps.copy()

    laps["LapTimeSeconds"] = (
        laps["LapTime"]
        .dt.total_seconds()
        .astype(float)
    )

    laps = laps.dropna(subset=["LapTimeSeconds"])

    agg = laps.groupby("Driver").agg(
        avg_lap_time=("LapTimeSeconds", "mean"),
        best_lap_time=("LapTimeSeconds", "min"),
        lap_count=("LapNumber", "count"),
        stints_used=("Stint", "nunique"),
        used_soft=("Compound", lambda x: int("SOFT" in set(x))),
        used_medium=("Compound", lambda x: int("MEDIUM" in set(x))),
        used_hard=("Compound", lambda x: int("HARD" in set(x))),
    ).reset_index()

    return agg



def build_results_features(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()

    results = results[
        [
            "DriverId",
            "Abbreviation",
            "FullName",
            "TeamName",
            "GridPosition",
            "Position",
            "ClassifiedPosition",
            "Points",
            "Status",
            "Laps"
        ]
    ]

    results.rename(
        columns={
            "Abbreviation": "Driver",
            "DriverId": "driverId",
            "Abbreviation": "Driver",
            "FullName": "driver_name",
            "TeamName": "team_name",
            "GridPosition": "grid_position",
            "Position": "position_raw",
            "Points": "points",
            "Laps": "laps_completed"
        },
        inplace=True
    )

    # Robust finish position
    results["finish_position"] = pd.to_numeric(
        results["ClassifiedPosition"],
        errors="coerce"
    )

    results["finish_position"] = results["finish_position"].fillna(
        results["position_raw"]
    )

    results["dnf"] = results["finish_position"].isna().astype(int)

    results.drop(
        columns=["ClassifiedPosition", "position_raw"],
        inplace=True
    )

    return results




def main():
    all_rows = []

    for season, race_name, laps, results in load_race_sessions():
        logger.info(f"Processing {season} | {race_name}")

        lap_features = build_lap_features(laps)
        result_features = build_results_features(results)

        df = lap_features.merge(
            result_features,
            on="Driver",
            how="inner"
        )



        df["season"] = season
        df["race_name"] = race_name

        all_rows.append(df)

    final_df = pd.concat(all_rows, ignore_index=True)

    output_path = OUTPUT_DIR / "race_driver_features.parquet"
    final_df.to_parquet(output_path, index=False)

    logger.info(f"Saved ML-ready dataset: {output_path}")
    logger.info(f"Final shape: {final_df.shape}")


if __name__ == "__main__":
    main()
