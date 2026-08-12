import os
import sys
import json
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.exception import CustomException
from src.logger import logger

# ==========================================================
# Function to Save Python Objects
# ==========================================================


def save_object(file_path, obj):
    """
    Saves any Python object as a pickle (.pkl) file.

    This function is mainly used to save:
    - Trained Machine Learning Models
    - Preprocessing Pipelines
    - Encoders
    - Scalers

    Parameters
    ----------
    file_path : str
        Destination path where the object will be stored.

    obj : object
        Any Python object to be saved.
    """

    try:

        # Get directory path
        dir_path = os.path.dirname(file_path)

        # Create directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)

        logger.info(f"Saving object at: {file_path}")

        # Open file in binary write mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info("Object saved successfully.")

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================================
# Function to Load Saved Objects
# ==========================================================


def load_object(file_path):
    """
    Loads a saved pickle (.pkl) object.

    Parameters
    ----------
    file_path : str
        Path of the pickle file.

    Returns
    -------
    object
        Loaded Python object.
    """

    try:

        logger.info(f"Loading object from: {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logger.info("Object loaded successfully.")

        return obj

    except Exception as e:
        raise CustomException(e, sys)


# ==========================================================
# Evaluate Multiple Machine Learning Models
# ==========================================================


def evaluate_models(X_train, y_train, models, params):
    """
    Train and evaluate multiple regression models.

    Model selection is performed using a validation set created
    from the training data. The test set is intentionally excluded
    from model selection to prevent test-set leakage.

    Returns
    -------
    dict
        Dictionary containing:

        {
            model_name:
            {
                "model": trained_model,
                "train_score": float,
                "validation_score": float
            }
        }
    """

    try:

        report = {}

        logger.info("=" * 80)
        logger.info("Model Evaluation Started")
        logger.info("=" * 80)

        # ==================================================
        # Create Validation Set
        # ==================================================

        X_train_split, X_validation, y_train_split, y_validation = train_test_split(
            X_train, y_train, test_size=0.20, random_state=42
        )

        logger.info(f"Training Samples   : {X_train_split.shape[0]}")

        logger.info(f"Validation Samples : {X_validation.shape[0]}")

        # ==================================================
        # Train and Evaluate Models
        # ==================================================

        for model_name, model in models.items():

            logger.info(f"Training {model_name}")

            param_grid = params.get(model_name, {})

            # --------------------------------------------------
            # Hyperparameter Tuning
            # --------------------------------------------------

            if param_grid:

                search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=5,
                    cv=3,
                    scoring="r2",
                    random_state=42,
                    n_jobs=-1,
                )

                search.fit(X_train_split, y_train_split)

                model = search.best_estimator_

                logger.info(f"Best Parameters : {search.best_params_}")

            else:

                model.fit(X_train_split, y_train_split)

            # --------------------------------------------------
            # Validation Prediction
            # --------------------------------------------------

            validation_prediction = model.predict(X_validation)

            validation_score = r2_score(y_validation, validation_prediction)

            # --------------------------------------------------
            # Retrain Model on Full Training Data
            # --------------------------------------------------

            model.fit(X_train, y_train)

            # --------------------------------------------------
            # Training Prediction
            # --------------------------------------------------

            train_prediction = model.predict(X_train)

            train_score = r2_score(y_train, train_prediction)

            logger.info(
                f"{model_name}"
                f" | Train R² : {train_score:.4f}"
                f" | Validation R² : {validation_score:.4f}"
            )

            # --------------------------------------------------
            # Store Model Results
            # --------------------------------------------------

            report[model_name] = {
                "model": model,
                "train_score": train_score,
                "validation_score": validation_score,
            }

        # ==================================================
        # Save Model Performance Report
        # ==================================================

        json_report = {}

        for model_name, values in report.items():

            json_report[model_name] = {
                "Train R2": round(values["train_score"], 4),
                "Validation R2": round(values["validation_score"], 4),
            }

        os.makedirs("artifacts", exist_ok=True)

        with open("artifacts/model_report.json", "w") as json_file:

            json.dump(json_report, json_file, indent=4)

        logger.info("Model Performance Report Saved.")

        logger.info("=" * 80)
        logger.info("Model Evaluation Completed")
        logger.info("=" * 80)

        return report

    except Exception as e:

        raise CustomException(e, sys)
