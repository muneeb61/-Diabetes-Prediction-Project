
# Diabetes Prediction Project

## Project Overview

This project is a **machine learning-based diabetes prediction** tool that leverages health data to classify whether a patient is diabetic or not. Using a **Support Vector Classifier (SVC)** algorithm, the model predicts diabetes based on health metrics, including glucose levels, BMI, insulin, and more. The dataset is taken from the **Pima Indians Diabetes Database** and contains various health features that have been preprocessed and analyzed.

The project aims to demonstrate the complete process of building a machine learning model, from data preprocessing and analysis to training, evaluation, and model deployment.

## Key Features

- **Data Preprocessing**: Handle missing values and prepare data for analysis.
- **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations, and trends in the dataset.
- **Machine Learning Model**: Train a predictive model using SVC (Support Vector Classifier).
- **Model Evaluation**: Evaluate model performance using metrics such as accuracy, confusion matrix, and classification report.
- **Model Persistence**: Save and load trained models for future use.

## Technologies Used

- **Python**: The programming language used for building the project.
- **Pandas**: Data manipulation and analysis library.
- **Numpy**: Library used for handling numerical operations.
- **Scikit-learn**: Provides machine learning algorithms, tools for splitting data, and evaluation metrics.
- **Matplotlib & Seaborn**: Visualization libraries used to create informative plots.
- **Pickle**: A module used to save and load machine learning models.

## Methods and Approach

### 1. Data Preprocessing

- **Null Value Checking**: Ensures there are no missing values in the dataset. If missing data is found, it is handled appropriately (e.g., using mean imputation, if needed).
- **Data Splitting**: The dataset is divided into training and testing sets (80% for training, 20% for testing), ensuring the model generalizes well when exposed to unseen data.

### 2. Data Visualization

- **Univariate Analysis**: 
  - Plots histograms and boxplots to understand the distribution and outliers in individual features like **Glucose**, **BMI**, and **Insulin**.
  - **Count Plots** are used to visualize the distribution of the target variable (**Outcome**).
  - **KDE Plots** (Kernel Density Estimation) are used to visualize the smooth distribution of continuous features like **BMI**.

- **Bivariate Analysis**:
  - **Heatmaps**: Show correlations between different features and the target variable.
  - **Scatter Plots**: Visualize relationships between continuous features and the target.
  - **Violin & Bar Plots**: Show distributions of continuous features with respect to the target variable.

### 3. Machine Learning Model

- **Support Vector Classifier (SVC)**:
  - SVC is a powerful classification algorithm used to classify whether the target variable (**Outcome**) indicates diabetes or not.
  - The model is trained using the training data (features: **Glucose**, **BMI**, **Insulin**, etc.) and tested using the test data.

### 4. Model Evaluation

- **Accuracy Score**: Provides the overall percentage of correct predictions made by the model.
- **Confusion Matrix**: Shows the counts of actual vs predicted classifications.
- **Classification Report**: Provides additional metrics like precision, recall, and F1-score.

### 5. Model Persistence

- **Pickle**: Once the model is trained, it is serialized and saved into a `.pkl` file using the `Pickle` library, which can be later loaded without retraining.

## Database

The project is based on the **Pima Indians Diabetes Database**, which contains medical information for 768 patients. This dataset includes features such as **Pregnancies**, **Glucose**, **Blood Pressure**, **Skin Thickness**, **Insulin**, **BMI**, **Diabetes Pedigree Function**, **Age**, and **Outcome**. 

- **Outcome** is the target variable, where 1 represents a diabetic patient, and 0 represents a non-diabetic patient.
- The dataset is publicly available from the **UCI Machine Learning Repository**.

### Data Columns:
- **Pregnancies**: Number of pregnancies the patient has had.
- **Glucose**: Plasma glucose concentration after a two-hour oral glucose tolerance test.
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body Mass Index (weight in kg / height in m^2).
- **DiabetesPedigreeFunction**: A function that calculates the likelihood of diabetes based on family history.
- **Age**: Age in years.
- **Outcome**: Whether the patient is diabetic (1) or not (0).

## Project Highlights

- **Exploratory Data Analysis (EDA)**: Thorough EDA using univariate and bivariate visualizations helped identify important patterns in the dataset, such as correlations and feature distributions.
- **Supervised Machine Learning**: SVC was chosen due to its effectiveness in high-dimensional spaces, making it well-suited for the diabetes prediction problem.
- **Model Evaluation**: The model achieved a respectable **accuracy of 76.62%**, which indicates its capability in predicting diabetes.
- **Model Persistence**: The trained model is saved into a `.pkl` file using **Pickle**, ensuring that the model can be easily reused without retraining.

## How to Run the Project

1. **Install Dependencies**:
   To install the necessary dependencies, run:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn pickle
   ```

2. **Load Data**:
   Use the `load_data()` function to load the diabetes dataset:
   ```python
   df = load_data("diabetes.csv")
   ```

3. **Visualize the Data**:
   Run the `UnivariateAnalysisRunner` and `BivariateAnalysisRunner` classes to generate insightful plots:
   ```python
   uni_runner = UnivariateAnalysisRunner(df)
   uni_runner.run_all()

   bi_runner = BivariateAnalysisRunner(df)
   bi_runner.run_all()
   ```

4. **Train the Model**:
   Split the data and train the SVC model using the `ModelTrainer` class:
   ```python
   trainer = ModelTrainer(X_train, X_test, y_train, y_test)
   trainer.train_model()
   trainer.evaluate_model()
   ```

5. **Save the Model**:
   Once the model is trained, save it for future use:
   ```python
   saver = ModelSaver(trainer.model)
   saver.save_model()
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
