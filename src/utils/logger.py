import logging
import os

def get_logger(name: str):
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("logs/ingestion.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)
