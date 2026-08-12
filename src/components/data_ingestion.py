import os
import sys

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


# ==========================================================
# Configuration Class
# ==========================================================

@dataclass
class DataIngestionConfig:
    """
    Configuration class for Data Ingestion.

    Stores all file paths used during the ingestion process.
    """

    # Original dataset
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

    # Training dataset
    train_data_path: str = os.path.join("artifacts", "train.csv")

    # Testing dataset
    test_data_path: str = os.path.join("artifacts", "test.csv")

    # Source dataset
    source_data_path: str = os.path.join("notebooks", "study.csv")


# ==========================================================
# Data Ingestion Class
# ==========================================================

class DataIngestion:
    """
    Handles reading the dataset,
    splitting it into train and test sets,
    and saving them into the artifacts folder.
    """

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # ======================================================
    # Data Ingestion Method
    # ======================================================

    def initiate_data_ingestion(self):
        """
        Executes the complete data ingestion workflow.

        Workflow
        --------
        1. Validate dataset path.
        2. Read dataset.
        3. Save raw dataset.
        4. Perform Train-Test Split.
        5. Save train.csv and test.csv.
        6. Return train and test file paths.
        """

        logger.info("=" * 60)
        logger.info("Data Ingestion Started")
        logger.info("=" * 60)

        try:

            # --------------------------------------------------
            # Check if dataset exists
            # --------------------------------------------------

            if not os.path.exists(self.ingestion_config.source_data_path):

                raise FileNotFoundError(
                    f"Dataset not found at: "
                    f"{self.ingestion_config.source_data_path}"
                )

            logger.info(
                f"Reading dataset from: "
                f"{self.ingestion_config.source_data_path}"
            )

            # Read dataset
            df = pd.read_csv(
                self.ingestion_config.source_data_path
            )

            logger.info(
                f"Dataset loaded successfully "
                f"with shape: {df.shape}"
            )

            # --------------------------------------------------
            # Create Artifacts Directory
            # --------------------------------------------------

            os.makedirs("artifacts", exist_ok=True)

            logger.info("Artifacts directory is ready.")

            # --------------------------------------------------
            # Save Raw Dataset
            # --------------------------------------------------

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logger.info(
                f"Raw dataset saved to: "
                f"{self.ingestion_config.raw_data_path}"
            )

            # --------------------------------------------------
            # Train-Test Split
            # --------------------------------------------------

            logger.info("Performing Train-Test Split...")

            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42
            )

            logger.info(
                f"Training Dataset Shape : {train_set.shape}"
            )

            logger.info(
                f"Testing Dataset Shape : {test_set.shape}"
            )

            # --------------------------------------------------
            # Save Training Dataset
            # --------------------------------------------------

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            logger.info(
                f"Training dataset saved to: "
                f"{self.ingestion_config.train_data_path}"
            )

            # --------------------------------------------------
            # Save Testing Dataset
            # --------------------------------------------------

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info(
                f"Testing dataset saved to: "
                f"{self.ingestion_config.test_data_path}"
            )

            logger.info("=" * 60)
            logger.info("Data Ingestion Completed Successfully")
            logger.info("=" * 60)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logger.exception("Error occurred during Data Ingestion.")
            raise CustomException(e, sys)


# ==========================================================
# Run the Component Independently
# ==========================================================

if __name__ == "__main__":

    ingestion = DataIngestion()

    train_path, test_path = ingestion.initiate_data_ingestion()

    print("\nTrain File :", train_path)
    print("Test File  :", test_path)