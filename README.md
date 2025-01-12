# Gaming Behavior Analysis

This project analyzes player gaming behavior using SQL, Python, and Machine Learning. It explores factors like player demographics, engagement levels, and gaming habits, with visualizations and predictive models to derive actionable insights.

## Features
- **Data Analysis**: 
  - SQL and pandas used to query, clean, and analyze data.
  - Explored player demographics, gaming engagement, and retention.
- **Visualizations**:
  - Bar plots, stacked charts, and line graphs to present insights visually.
- **Machine Learning**:
  - **Random Forest Classifier**: Predicts engagement levels using Scikit-learn.
  - **Custom Logistic Regression**: Implemented manually for classification.

## Tools and Technologies
- **Python**: For data analysis, visualizations, and machine learning.
- **SQLite**: For database management and querying.
- **Scikit-learn**: For Random Forest and evaluation metrics.
- **Matplotlib** and **Seaborn**: For visualizing data insights.
- **NumPy**: For mathematical computations in logistic regression.

## Dataset
The dataset includes player demographics, engagement levels, in-game purchases, and session statistics. It was loaded into an SQLite database for querying and analysis.

## Key Analyses
1. **Demographics Likely to Make In-Game Purchases**:
   - Age, Gender, and Location-based trends.
   - Visualized with bar plots.
2. **Factors Contributing to High Engagement**:
   - Metrics like PlayTimeHours, SessionsPerWeek, and AchievementsUnlocked.
3. **Game Genre Insights**:
   - Retention rates, session durations, and popularity.
4. **Machine Learning Models**:
   - **Random Forest**: Used Scikit-learn to predict engagement levels.
   - **Custom Logistic Regression**: Built a gradient descent-based classifier.

## Folder Structure

The project is organized as follows:

plots/: Contains visualizations of the data analysis.
visualizations/: Subfolder for storing specific visualization files.
src/: Contains the Python scripts for data analysis and machine learning.
script.py: Main Python script for analysis and modeling.
README.md: Project documentation.
GamingProj.tex: LaTeX file for the report.

## Dataset

The dataset includes player demographics, engagement levels, in-game purchases, and session statistics. It was loaded into an SQLite database for querying and analysis.

- **Source**: [Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset) on Kaggle.

