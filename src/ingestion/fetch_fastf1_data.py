import os
import yaml
import pandas as pd
import fastf1
from tqdm import tqdm
from pathlib import Path
import time

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    
    from pathlib import Path

def session_already_fetched(base_dir: str) -> bool:
    required_files = [
        "laps.parquet",
        "results.parquet"
    ]
    return all((Path(base_dir) / f).exists() for f in required_files)



def main():
    config = load_config()

    seasons = config["fastf1"]["seasons"]
    sessions = config["fastf1"]["sessions"]
    cache_dir = config["fastf1"]["cache_dir"]
    raw_path = config["paths"]["raw_data"]

    # Enable FastF1 caching
    fastf1.Cache.enable_cache(cache_dir)

    for season in seasons:
        logger.info(f"Starting ingestion for season {season}")

        schedule = fastf1.get_event_schedule(season)

        for _, event in tqdm(schedule.iterrows(), total=len(schedule)):
            event_name = event["EventName"]

            for session_code in sessions:
                    base_dir = f"{raw_path}/{season}/{event_name}/{session_code}"

                    if session_already_fetched(base_dir):
                        logger.info(
                            f"Skipping (already fetched): "
                            f"{season} | {event_name} | {session_code}"
                        )
                        continue

                    try:
                        logger.info(f"Fetching: {season} | {event_name} | {session_code}")

                        session = fastf1.get_session(season, event_name, session_code)
                        session.load(telemetry=False, weather=False)

                        laps = session.laps
                        results = session.results

                        save_parquet(laps, f"{base_dir}/laps.parquet")
                        save_parquet(results, f"{base_dir}/results.parquet")

                        time.sleep(5)

                    except Exception as e:
                        logger.error(
                            f"FAILED | {season} | {event_name} | "
                            f"{session_code} | {e}"
                        )


        logger.info(f"Completed ingestion for season {season}")


if __name__ == "__main__":
    main()
