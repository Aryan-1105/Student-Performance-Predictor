import os
import sys

from dataclasses import dataclass

import numpy as np

# ==========================================================
# Regression Models
# ==========================================================

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)

from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

# ==========================================================
# Custom Modules
# ==========================================================

from src.exception import CustomException
from src.logger import logger

from src.utils import save_object, evaluate_models

# ==========================================================
# Configuration Class
# ==========================================================


@dataclass
class ModelTrainerConfig:
    """
    Stores the location where the best trained
    model will be saved.
    """

    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# ==========================================================
# Model Trainer Class
# ==========================================================


class ModelTrainer:
    """
    This class is responsible for:

    1. Training multiple regression models
    2. Hyperparameter tuning
    3. Selecting the best model using validation data
    4. Evaluating the selected model on the test set
    5. Saving the best model
    """

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    # ======================================================
    # Train Models
    # ======================================================

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logger.info("=" * 80)
            logger.info("Model Training Started")
            logger.info("=" * 80)

            # ==================================================
            # Split Features & Target
            # ==================================================

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logger.info(f"Training Samples : {X_train.shape[0]}")

            logger.info(f"Testing Samples  : {X_test.shape[0]}")

            logger.info(f"Number of Features : {X_train.shape[1]}")

            # ==================================================
            # Machine Learning Models
            # ==================================================

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(max_iter=10000),
                "ElasticNet": ElasticNet(max_iter=10000),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Extra Trees": ExtraTreesRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "AdaBoost": AdaBoostRegressor(random_state=42),
                "K Neighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(
                    objective="reg:squarederror", random_state=42, verbosity=0
                ),
                "CatBoost": CatBoostRegressor(
                    verbose=False, allow_writing_files=False, random_state=42
                ),
            }

            # ==================================================
            # Hyperparameter Grid
            # ==================================================

            params = {
                "Linear Regression": {},
                "Ridge Regression": {"alpha": [0.1, 1, 10]},
                "Lasso Regression": {"alpha": [0.001, 0.01, 0.1]},
                "ElasticNet": {
                    "alpha": [0.001, 0.01, 0.1],
                    "l1_ratio": [0.2, 0.5, 0.8],
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [None, 5, 10, 20],
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                },
                "Extra Trees": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [100, 200],
                },
                "AdaBoost": {
                    "learning_rate": [0.01, 0.1, 1],
                    "n_estimators": [50, 100, 200],
                },
                "K Neighbors": {"n_neighbors": [3, 5, 7, 9]},
                "XGBoost": {"learning_rate": [0.01, 0.1], "n_estimators": [100, 200]},
                "CatBoost": {"depth": [4, 6, 8], "learning_rate": [0.01, 0.1]},
            }

            # ==================================================
            # Evaluate All Models
            # ==================================================

            logger.info("Evaluating all machine learning models...")

            model_report = evaluate_models(
                X_train=X_train, y_train=y_train, models=models, params=params
            )

            logger.info("Model evaluation completed successfully.")

            # ==================================================
            # Display Model Performance
            # ==================================================

            print("\n")
            print("=" * 80)
            print("MODEL PERFORMANCE REPORT")
            print("=" * 80)

            logger.info("=" * 80)
            logger.info("MODEL PERFORMANCE REPORT")
            logger.info("=" * 80)

            best_model_name = None
            best_model_score = float("-inf")
            best_model = None

            # ==================================================
            # Select Best Model Using Validation Score
            # ==================================================

            for model_name, model_info in model_report.items():

                train_score = model_info["train_score"]

                validation_score = model_info["validation_score"]

                print(
                    f"{model_name:<25}"
                    f"Train R² : {train_score:.4f}"
                    f"    Validation R² : "
                    f"{validation_score:.4f}"
                )

                logger.info(
                    f"{model_name:<25}"
                    f"Train R² : {train_score:.4f}"
                    f" | Validation R² : "
                    f"{validation_score:.4f}"
                )

                # --------------------------------------------------
                # Select Best Model
                # --------------------------------------------------

                if validation_score > best_model_score:

                    best_model_score = validation_score

                    best_model_name = model_name

                    best_model = model_info["model"]

            print("=" * 80)

            logger.info("=" * 80)

            # ==================================================
            # Validate Best Model
            # ==================================================

            if best_model is None:

                raise Exception("No valid model was selected.")

            # ==================================================
            # Minimum Validation Performance Check
            # ==================================================

            minimum_expected_score = 0.70

            if best_model_score < minimum_expected_score:

                raise Exception(
                    f"No model achieved the minimum required "
                    f"Validation R² Score "
                    f"({minimum_expected_score})."
                )

            # ==================================================
            # Save Best Model
            # ==================================================

            logger.info(f"Selected Best Model : " f"{best_model_name}")

            logger.info(f"Best Validation R² : " f"{best_model_score:.4f}")

            save_object(
                file_path=(self.model_trainer_config.trained_model_file_path),
                obj=best_model,
            )

            logger.info("Best Model Saved Successfully.")

            # ==================================================
            # Final Test Evaluation
            # ==================================================
            #
            # IMPORTANT:
            #
            # The test set is used ONLY here.
            #
            # It was NOT used to select the best model.
            #
            # ==================================================

            predicted_values = best_model.predict(X_test)

            final_r2_score = r2_score(y_test, predicted_values)

            # ==================================================
            # Final Result
            # ==================================================

            print("\n")
            print("=" * 80)

            print(f"Best Model          : " f"{best_model_name}")

            print(f"Validation R²       : " f"{best_model_score:.4f}")

            print(f"Final Test R²       : " f"{final_r2_score:.4f}")

            print("=" * 80)

            logger.info("=" * 80)

            logger.info(f"Best Model : " f"{best_model_name}")

            logger.info(f"Validation R² : " f"{best_model_score:.4f}")

            logger.info(f"Final Test R² : " f"{final_r2_score:.4f}")

            logger.info("=" * 80)

            return (best_model_name, final_r2_score)

        except Exception as e:

            logger.exception("Error occurred during Model Training.")

            raise CustomException(e, sys)


# ==========================================================
# Testing the Component
# ==========================================================

if __name__ == "__main__":

    try:

        logger.info("=" * 80)
        logger.info("Starting End-to-End Model Training Pipeline")
        logger.info("=" * 80)

        # --------------------------------------------------
        # Import Project Components
        # --------------------------------------------------

        from src.components.data_ingestion import DataIngestion

        from src.components.data_transformation import DataTransformation

        # --------------------------------------------------
        # Step 1 : Data Ingestion
        # --------------------------------------------------

        ingestion = DataIngestion()

        train_path, test_path = ingestion.initiate_data_ingestion()

        logger.info("Data Ingestion Completed.")

        # --------------------------------------------------
        # Step 2 : Data Transformation
        # --------------------------------------------------

        transformation = DataTransformation()

        train_array, test_array, preprocessor_path = (
            transformation.initiate_data_transformation(train_path, test_path)
        )

        logger.info("Data Transformation Completed.")

        # --------------------------------------------------
        # Step 3 : Model Training
        # --------------------------------------------------

        trainer = ModelTrainer()

        best_model_name, final_r2_score = trainer.initiate_model_trainer(
            train_array, test_array
        )

        # --------------------------------------------------
        # Final Output
        # --------------------------------------------------

        print("\n")
        print("=" * 80)
        print("END-TO-END TRAINING PIPELINE COMPLETED")
        print("=" * 80)

        print(f"Best Model : " f"{best_model_name}")

        print(f"Final Test R² Score : " f"{final_r2_score:.4f}")

        print("=" * 80)

        logger.info("Pipeline Executed Successfully.")

    except Exception as e:

        logger.exception("Pipeline Execution Failed.")

        raise CustomException(e, sys)
