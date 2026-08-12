# 🎓 Student Performance Predictor

An end-to-end Machine Learning application that predicts a student's **Mathematics Score** from demographic and academic information.

This project demonstrates the complete Machine Learning lifecycle — from data ingestion and preprocessing to model training, hyperparameter tuning, model selection, serialization, and deployment through a Flask web application.

---

## 🚀 Live Demo

🌐 **Live Website:**
[Student Performance Predictor](https://student-performance-predictor-8et9.onrender.com/)

---

## 📸 Application Preview

### Landing Page

![Landing Page](screenshot/landing_page.png)

### Prediction Page

![Prediction Page](screenshot/prediction_page.png)

### Prediction Result

![Prediction Result](screenshot/prediction_result.png)

---

## 📌 Features

* End-to-end Machine Learning pipeline
* Data ingestion and train/test splitting
* Data preprocessing and transformation
* Feature engineering
* Multiple regression model comparison
* Hyperparameter tuning using `RandomizedSearchCV`
* Validation-based model selection
* Final evaluation on a held-out test set
* Model serialization using Pickle
* Separate preprocessing pipeline
* Flask-based prediction application
* Responsive Bootstrap frontend
* Logging and custom exception handling
* Modular and reusable project architecture
* Cloud deployment using Render

---

## 🛠️ Tech Stack

### Programming

* Python
* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost
* CatBoost

### Backend

* Flask

### Frontend

* HTML5
* CSS3
* Bootstrap 5
* JavaScript

### Model Serialization

* Pickle

### Deployment

* Render

---

## 📂 Project Structure

```text
Student-Performance-Predictor/
│
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── model_report.json
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Training.ipynb
│   └── study.csv
│
├── screenshot/
│   ├── landing_page.png
│   ├── prediction_page.png
│   └── prediction_result.png
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
│       ├── favicon.ico
│       ├── hero.png
│       └── logo.png
│
├── templates/
│   ├── index.html
│   └── home.html
│
├── app.py
├── Procfile
├── requirements.txt
├── setup.py
├── README.md
└── .gitignore
```

### Generated Files

The following files are generated during model training and are intentionally excluded from version control:

* `artifacts/raw.csv`
* `artifacts/train.csv`
* `artifacts/test.csv`
* `catboost_info/`

The trained model, preprocessing pipeline, and model performance report are retained because they are required for the deployed application and project reproducibility.

---

## 📊 Dataset

The project uses the **Student Performance Dataset**.

### Input Features

The model uses the following features:

* Gender
* Race / Ethnicity
* Parental Level of Education
* Lunch
* Test Preparation Course
* Reading Score
* Writing Score

### Target Variable

* Mathematics Score

The objective is to predict a student's **Mathematics Score** based on demographic information and performance in other academic areas.

---

## ⚙️ Machine Learning Workflow

```text
Student Performance Dataset
          ↓
     Data Ingestion
          ↓
    Train / Test Split
          ↓
  Data Transformation
          ↓
   Feature Engineering
          ↓
    Validation Split
          ↓
 Hyperparameter Tuning
          ↓
   Model Comparison
          ↓
 Best Model Selection
          ↓
 Final Test Evaluation
          ↓
  Model Serialization
          ↓
 Flask Web Application
          ↓
      Prediction
```

---

## 🔬 Model Selection Strategy

The project is designed to avoid using the test set for model selection.

The training process follows this approach:

```text
Training Dataset
       ↓
80% Training / 20% Validation
       ↓
Train Candidate Models
       ↓
Hyperparameter Tuning
       ↓
Compare Validation R²
       ↓
Select Best Model
       ↓
Retrain Selected Model
on Full Training Dataset
       ↓
Evaluate Once on Test Dataset
```

### Why this approach?

Using the test set during model selection can lead to **test-set leakage** and overly optimistic performance estimates.

Instead:

* The training data is used for model development.
* A validation split is used to compare models.
* Hyperparameter tuning uses cross-validation on the training split.
* The held-out test set is reserved for the final evaluation.

This provides a cleaner estimate of how the selected model performs on unseen data.

---

## 🤖 Models Trained

The pipeline evaluates **12 regression models**:

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet
5. Decision Tree Regressor
6. Random Forest Regressor
7. Extra Trees Regressor
8. Gradient Boosting Regressor
9. AdaBoost Regressor
10. K-Nearest Neighbors Regressor
11. XGBoost Regressor
12. CatBoost Regressor

Hyperparameter tuning is performed using `RandomizedSearchCV` with 3-fold cross-validation for models that have defined parameter distributions.

---

## 📈 Model Performance

The primary evaluation metric is:

### R² Score

The coefficient of determination (R²) measures how much of the variance in the target variable is explained by the model.

The current model comparison report generated by the training pipeline is stored in:

```text
artifacts/model_report.json
```

### Validation Performance

| Model                |   Train R² | Validation R² |
| -------------------- | ---------: | ------------: |
| **Ridge Regression** | **0.8739** |    **0.8597** |
| Linear Regression    |     0.8743 |        0.8587 |
| ElasticNet           |     0.8724 |        0.8586 |
| Lasso Regression     |     0.8736 |        0.8580 |
| CatBoost             |     0.8882 |        0.8513 |
| Gradient Boosting    |     0.9050 |        0.8369 |
| Extra Trees          |     0.9767 |        0.8300 |
| XGBoost              |     0.9681 |        0.8252 |
| Random Forest        |     0.9648 |        0.8248 |
| AdaBoost             |     0.8565 |        0.8114 |
| Decision Tree        |     0.8550 |        0.7837 |
| K-Neighbors          |     0.6500 |        0.5126 |

### Current Best Model

Based on the current validation results:

**Ridge Regression**

* Training R²: **0.8739**
* Validation R²: **0.8597**

The training pipeline selects the model with the highest validation R² score.

> Note: The final test R² is generated during the training pipeline and depends on the current dataset, preprocessing configuration, and installed library versions. The validation results above are taken from the current `artifacts/model_report.json`.

---

## 🧠 Model Training Architecture

The project follows a modular Machine Learning architecture.

### Data Ingestion

Responsible for:

* Loading the dataset
* Splitting the dataset into training and testing data
* Saving the generated dataset splits

### Data Transformation

Responsible for:

* Identifying numerical and categorical features
* Applying preprocessing
* Encoding categorical variables
* Scaling numerical features
* Creating the preprocessing pipeline
* Saving the fitted preprocessor

### Model Training

Responsible for:

* Training multiple regression algorithms
* Performing hyperparameter tuning
* Evaluating validation performance
* Selecting the best-performing model
* Evaluating the selected model on the test dataset
* Saving the trained model
* Generating the model performance report

### Prediction Pipeline

Responsible for:

* Loading the trained model
* Loading the preprocessing pipeline
* Transforming user input
* Generating the predicted Mathematics Score

---

## 🖥️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Aryan-1105/Student-Performance-Predictor.git
```

### 2. Navigate to the Project Directory

```bash
cd Student-Performance-Predictor
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

### Train the Machine Learning Pipeline

```bash
python src/pipeline/train_pipeline.py
```

This process:

1. Loads the dataset
2. Splits the data
3. Applies preprocessing
4. Trains multiple models
5. Performs hyperparameter tuning
6. Compares validation performance
7. Selects the best model
8. Evaluates the selected model on the test set
9. Saves the trained model
10. Saves the preprocessing pipeline
11. Generates `model_report.json`

---

### Run the Flask Application

```bash
python app.py
```

The application will run locally at:

```text
http://127.0.0.1:5000
```

Open the address in your browser to use the prediction application.

---

## 🔮 Prediction Workflow

```text
User
 ↓
Enter Student Information
 ↓
Flask Application
 ↓
Prediction Pipeline
 ↓
Preprocessing Pipeline
 ↓
Trained Ridge Regression Model
 ↓
Predicted Mathematics Score
 ↓
Display Result
```

---

## 🌐 Deployment

The application is deployed using **Render**.

### Production Application

https://student-performance-predictor-8et9.onrender.com/

The deployment uses the Flask application and serialized Machine Learning artifacts to serve predictions through a web interface.

---

## 🔐 Security & Configuration

Sensitive configuration files should not be committed to the repository.

The `.gitignore` includes:

```text
.env
.streamlit/secrets.toml
venv/
__pycache__/
```

Generated training files and CatBoost logs are also excluded from version control.

---

## 🚧 Future Improvements

Potential improvements include:

* Docker containerization
* CI/CD using GitHub Actions
* Model monitoring
* Data drift detection
* Explainable AI using SHAP
* REST API development
* Database integration
* User authentication
* Automated model retraining
* Unit and integration testing
* Improved model experiment tracking

---

## 👨‍💻 Author

**Aryan Kumar Sahoo**

Mechanical Engineering
National Institute of Technology Rourkela

### Profiles

* GitHub: https://github.com/Aryan-1105
* LinkedIn: https://linkedin.com/in/aryan-kumar-sahoo

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is licensed under the MIT License.
