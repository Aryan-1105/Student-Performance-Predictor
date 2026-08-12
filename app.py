"""
==========================================================
Student Performance Prediction System

Flask Application

This application allows users to enter
student information and predicts
their Mathematics Score.
==========================================================
"""

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import (

    CustomData,

    PredictPipeline

)

# ==========================================================
# Create Flask App
# ==========================================================

app = Flask(__name__)

# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():

    """
    Displays the home page.
    """

    return render_template("index.html")

# ==========================================================
# Prediction Route
# ==========================================================

@app.route("/predictdata", methods=["GET", "POST"])

def predict_datapoint():

    """
    Handles user input and returns
    predicted Math Score.
    """

    if request.method == "GET":

        return render_template("home.html")

    else:

        try:

            # ---------------------------------------------
            # Collect User Input
            # ---------------------------------------------

            data = CustomData(

                gender=request.form.get("gender"),

                race_ethnicity=request.form.get(
                    "ethnicity"
                ),

                parental_level_of_education=request.form.get(
                    "parental_level_of_education"
                ),

                lunch=request.form.get(
                    "lunch"
                ),

                test_preparation_course=request.form.get(
                    "test_preparation_course"
                ),

                reading_score=float(
                    request.form.get("reading_score")
                ),

                writing_score=float(
                    request.form.get("writing_score")
                )

            )

            # ---------------------------------------------
            # Convert to DataFrame
            # ---------------------------------------------

            pred_df = data.get_data_as_dataframe()

            # ---------------------------------------------
            # Prediction Pipeline
            # ---------------------------------------------

            prediction_pipeline = PredictPipeline()

            prediction = prediction_pipeline.predict(
                pred_df
            )

            predicted_score = round(
                prediction[0],
                2
            )

            # ---------------------------------------------
            # Display Result
            # ---------------------------------------------

            return render_template(

                "home.html",

                results=predicted_score

            )

        except Exception as e:

            return str(e)
        
# ==========================================================
# Run Flask Application
# ==========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=False

    )