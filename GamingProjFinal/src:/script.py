#!/usr/bin/env python
# coding: utf-8

# In[252]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt


# In[253]:


import os
os.chdir("/Users/ahmadkhan/Downloads/GamingProject")


# In[254]:


# Connect to SQLite database
conn = sqlite3.connect("gaming_data.db")

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Export DataFrame to SQLite table
df.to_sql("player_data", conn, if_exists="replace", index=False)

print("Dataset successfully loaded into SQLite!")


# In[255]:


query = "SELECT * FROM player_data LIMIT 5;"
result = pd.read_sql(query, conn)
print(result)


# In[256]:


query = "SELECT COUNT(*) FROM player_data;"
result = pd.read_sql(query, conn)
print(result)


# In[257]:


query = "PRAGMA table_info(player_data);"
result = pd.read_sql(query, conn)
print(result)


# In[258]:


# Analyzing player demographics to determine who is most likely to make in-game purchases (By Age) 
query = """
SELECT Age,
       COUNT(*) AS TotalPlayers,
       SUM(InGamePurchases) AS Purchasers,
       ROUND(SUM(InGamePurchases) * 100.0 / COUNT(*), 2) AS PurchasePercentage
FROM player_data
GROUP BY Age
ORDER BY PurchasePercentage DESC;
"""
result_age = pd.read_sql(query, conn)
print(result_age)


# In[259]:


# Analyzing player demographics to determine who is most likely to make in-game purchases (By Gender) 

query = """
SELECT Gender,
       COUNT(*) AS TotalPlayers,
       SUM(InGamePurchases) AS Purchasers,
       ROUND(SUM(InGamePurchases) * 100.0 / COUNT(*), 2) AS PurchasePercentage
FROM player_data
GROUP BY Gender
ORDER BY PurchasePercentage DESC;
"""
result_gender = pd.read_sql(query, conn)
print(result_gender)


# In[260]:


# Analyzing player demographics to determine who is most likely to make in-game purchases (By Location) 

query = """
SELECT Location,
       COUNT(*) AS TotalPlayers,
       SUM(InGamePurchases) AS Purchasers,
       ROUND(SUM(InGamePurchases) * 100.0 / COUNT(*), 2) AS PurchasePercentage
FROM player_data
GROUP BY Location
ORDER BY PurchasePercentage DESC
LIMIT 10;  -- Show top 10 locations
"""
result_location = pd.read_sql(query, conn)
print(result_location)


# In[261]:


# Plot purchase percentage by Age
result_age.plot(kind='bar', x='Age', y='PurchasePercentage', legend=False)
plt.title("Purchase Percentage by Age")
plt.xlabel("Age")
plt.ylabel("Purchase Percentage (%)") 
plt.show()


# In[262]:


# Plot purchase percentage by Gender
result_gender.plot(kind='bar', x='Gender', y='PurchasePercentage', legend=False, figsize=(8, 6))
plt.title("Purchase Percentage by Gender")
plt.xlabel("Gender")
plt.ylabel("Purchase Percentage (%)")
plt.xticks(rotation=0)  # Keep x-axis labels horizontal
plt.tight_layout()
plt.show()



# In[263]:


# Plot purchase percentage for Locations
result_location.plot(kind='bar', x='Location', y='PurchasePercentage', legend=False)
plt.title("Purchase Percentage by Location")
plt.xlabel("Location")
plt.ylabel("Purchase Percentage (%)")
plt.show()


# In[264]:


# What Factors Contribute Most to a High Engagement Level
query = """
SELECT EngagementLevel,
       AVG(PlayTimeHours) AS AvgPlayTime,
       AVG(SessionsPerWeek) AS AvgSessions,
       AVG(AchievementsUnlocked) AS AvgAchievements
FROM player_data
GROUP BY EngagementLevel
ORDER BY CASE
            WHEN EngagementLevel = 'High' THEN 1
            WHEN EngagementLevel = 'Medium' THEN 2
            WHEN EngagementLevel = 'Low' THEN 3
         END;  -- Order levels High > Medium > Low
"""
result_factors = pd.read_sql(query, conn)
print(result_factors)


# In[265]:


# Plot of factors contributing to engagement levels
result_factors.plot(kind='bar', x='EngagementLevel', figsize=(10, 6))
plt.title("Factors Contributing to Engagement Levels")
plt.xlabel("Engagement Level")
plt.ylabel("Average Values")
plt.xticks(rotation=0)
plt.legend(["Play Time (Hours)", "Sessions Per Week", "Achievements Unlocked"])
plt.show()


# In[266]:


# Which Game Genres Have the Highest Retention (High Engagement Level)?
query = """
SELECT GameGenre,
       COUNT(*) AS TotalPlayers,
       SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) AS HighEngagedPlayers,
       ROUND(SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS HighEngagementPercentage
FROM player_data
GROUP BY GameGenre
ORDER BY HighEngagementPercentage DESC
LIMIT 10;  -- Show top 10 genres
"""
result_genres = pd.read_sql(query, conn)
print(result_genres)


# In[267]:


# Plot for Which Game Genres Have the Highest Retention (High Engagement Level)?
result_genres.plot(kind='bar', x='GameGenre', y='HighEngagementPercentage', legend=False, figsize=(10, 6))
plt.title("Game Genres by High Engagement Percentage")
plt.xlabel("Game Genre")
plt.ylabel("High Engagement Percentage (%)")
plt.xticks(rotation=45)
plt.show()


# In[268]:


# Which Game Genres Have the Longest Average Session Durations?
query = """
SELECT GameGenre,
       AVG(AvgSessionDurationMinutes) AS AvgSessionDuration
FROM player_data
GROUP BY GameGenre
ORDER BY AvgSessionDuration DESC
LIMIT 10;  -- Show the top 10 genres
"""
result_session_duration = pd.read_sql(query, conn)
print(result_session_duration)


# In[269]:


# Plot for Which Game Genres Have the Longest Average Session Durations?
result_session_duration.plot(kind='bar', x='GameGenre', y='AvgSessionDuration', legend=False, figsize=(10, 6))
plt.title("Game Genres by Average Session Duration")
plt.xlabel("Game Genre")
plt.ylabel("Average Session Duration (Minutes)")
plt.xticks(rotation=45)
plt.show()


# In[270]:


# Which Genres are Most Popular Among High Engagement Players?
query = """
SELECT GameGenre,
       COUNT(*) AS TotalPlayers,
       SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) AS HighEngagedPlayers,
       ROUND(SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS HighEngagementPercentage
FROM player_data
GROUP BY GameGenre
ORDER BY HighEngagementPercentage DESC
LIMIT 10;  -- Show the top 10 genres
"""
result_high_engagement = pd.read_sql(query, conn)
print(result_high_engagement)


# In[271]:


# Plot for Which Genres are Most Popular Among High Engagement Players?
result_high_engagement.plot(kind='bar', x='GameGenre', y='HighEngagementPercentage', legend=False, figsize=(10, 6))
plt.title("Game Genres by High Engagement Percentage")
plt.xlabel("Game Genre")
plt.ylabel("High Engagement Percentage (%)")
plt.xticks(rotation=45)
plt.show()


# In[272]:


# How Does Engagement Level Vary by Age?
query = """
SELECT Age,
       EngagementLevel,
       COUNT(*) AS PlayerCount,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY Age), 2) AS EngagementPercentage
FROM player_data
GROUP BY Age, EngagementLevel
ORDER BY Age, EngagementLevel;
"""
result_age = pd.read_sql(query, conn)
print(result_age)


# In[273]:


# Barplot for Age and Engagement Level
sns.barplot(data=result_age, x='Age', y='EngagementPercentage', hue='EngagementLevel')
plt.title("Engagement Level by Age")
plt.xlabel("Age")
plt.ylabel("Engagement Percentage (%)")
plt.xticks(rotation=45)
plt.legend(title="Engagement Level")
plt.show()


# In[274]:


# Grouped Age and Engagement Level
query = """
SELECT
    CASE
        WHEN Age BETWEEN 15 AND 20 THEN '15-20'
        WHEN Age BETWEEN 21 AND 25 THEN '21-25'
        WHEN Age BETWEEN 26 AND 30 THEN '26-30'
        WHEN Age BETWEEN 31 AND 40 THEN '31-40'
        ELSE 'Other' END AS AgeGroup,
    EngagementLevel,
    COUNT(*) AS PlayerCount,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (
        PARTITION BY 
            CASE
                WHEN Age BETWEEN 15 AND 20 THEN '15-20'
                WHEN Age BETWEEN 21 AND 25 THEN '21-25'
                WHEN Age BETWEEN 26 AND 30 THEN '26-30'
                WHEN Age BETWEEN 31 AND 40 THEN '31-40'
                ELSE 'Other' END
    ), 2) AS EngagementPercentage
FROM player_data
GROUP BY 
    CASE
        WHEN Age BETWEEN 15 AND 20 THEN '15-20'
        WHEN Age BETWEEN 21 AND 25 THEN '21-25'
        WHEN Age BETWEEN 26 AND 30 THEN '26-30'
        WHEN Age BETWEEN 31 AND 40 THEN '31-40'
        ELSE 'Other' END,
    EngagementLevel
ORDER BY 
    CASE
        WHEN Age BETWEEN 15 AND 20 THEN '15-20'
        WHEN Age BETWEEN 21 AND 25 THEN '21-25'
        WHEN Age BETWEEN 26 AND 30 THEN '26-30'
        WHEN Age BETWEEN 31 AND 40 THEN '31-40'
        ELSE 'Other' END,
    EngagementLevel;
"""
result_age_group = pd.read_sql(query, conn)
print(result_age_group)


# In[275]:


# Plot for Grouped Age and Engagement Level
plt.figure(figsize=(10, 6))
sns.barplot(
    data=result_age_group,
    x="AgeGroup",
    y="EngagementPercentage",
    hue="EngagementLevel",
    palette="viridis"  # Change color palette if desired
)

# Add chart details
plt.title("Engagement Level by Age Group", fontsize=14)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Engagement Percentage (%)", fontsize=12)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title="Engagement Level", fontsize=10)
plt.tight_layout()

# Show the chart
plt.show()


# In[276]:


# How Does Engagement Level Vary by Gender?
query = """
SELECT Gender,
       EngagementLevel,
       COUNT(*) AS PlayerCount,
       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY Gender), 2) AS EngagementPercentage
FROM player_data
GROUP BY Gender, EngagementLevel
ORDER BY Gender, EngagementPercentage DESC;
"""
result_gender = pd.read_sql(query, conn)
print(result_gender)


# In[277]:


# Plot for Gender and Engagement Level?

result_gender.pivot(index='Gender', columns='EngagementLevel', values='EngagementPercentage').plot(
    kind='bar', figsize=(8, 6)
)
plt.title("Engagement Level by Gender")
plt.xlabel("Gender")
plt.ylabel("Engagement Percentage (%)")
plt.legend(title="Engagement Level")
plt.show()


# In[278]:


# Which Locations Have the Highest Proportion of High Engagement Players?
query = """
SELECT Location,
       COUNT(*) AS TotalPlayers,
       SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) AS HighEngagedPlayers,
       ROUND(SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS HighEngagementPercentage
FROM player_data
GROUP BY Location
ORDER BY HighEngagementPercentage DESC
LIMIT 10;  -- Show top 10 locations
"""
result_location = pd.read_sql(query, conn)
print(result_location)


# In[279]:


# Plot for Which Locations Have the Highest Proportion of High Engagement Players?
result_location.plot(kind='bar', x='Location', y='HighEngagementPercentage', legend=False, figsize=(10, 6))
plt.title("Locations by High Engagement Percentage")
plt.xlabel("Location")
plt.ylabel("High Engagement Percentage (%)")
plt.xticks(rotation=45)
plt.show()


# In[280]:


# Is There a Significant Relationship Between PlayerLevel and AchievementsUnlocked?

query = """
SELECT PlayerLevel,
       AVG(AchievementsUnlocked) AS AvgAchievements
FROM player_data
GROUP BY PlayerLevel
ORDER BY PlayerLevel;
"""
result_level_achievements = pd.read_sql(query, conn)
print(result_level_achievements)

import matplotlib.pyplot as plt

# Line plot for PlayerLevel vs AvgAchievements
plt.figure(figsize=(10, 6))
plt.plot(result_level_achievements['PlayerLevel'], result_level_achievements['AvgAchievements'], marker='o')
plt.title("Relationship Between Player Level and Achievements Unlocked", fontsize=14)
plt.xlabel("Player Level", fontsize=12)
plt.ylabel("Average Achievements Unlocked", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[281]:


# Do Higher Levels Correlate with Longer Average Session Durations?

query = """
SELECT PlayerLevel,
       AVG(AvgSessionDurationMinutes) AS AvgSessionDuration
FROM player_data
GROUP BY PlayerLevel
ORDER BY PlayerLevel;
"""
result_level_duration = pd.read_sql(query, conn)
print(result_level_duration)

# Line plot for PlayerLevel vs AvgSessionDuration
plt.figure(figsize=(10, 6))
plt.plot(result_level_duration['PlayerLevel'], result_level_duration['AvgSessionDuration'], marker='o', color='green')
plt.title("Relationship Between Player Level and Average Session Duration", fontsize=14)
plt.xlabel("Player Level", fontsize=12)
plt.ylabel("Average Session Duration (Minutes)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()



# In[282]:


# How Does GameDifficulty Affect Session Duration, Engagement, and Retention?

query = """
SELECT 
    GameDifficulty,
    AVG(AvgSessionDurationMinutes) AS AvgSessionDuration,
    COUNT(*) AS TotalPlayers,
    SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) AS HighEngagedPlayers,
    ROUND(SUM(CASE WHEN EngagementLevel = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS HighEngagementPercentage
FROM player_data
GROUP BY GameDifficulty
ORDER BY AvgSessionDuration DESC;
"""
result_difficulty = pd.read_sql(query, conn)
print(result_difficulty)

import matplotlib.pyplot as plt
import numpy as np

# Data preparation
categories = result_difficulty['GameDifficulty']
session_durations = result_difficulty['AvgSessionDuration']
engagement_percentages = result_difficulty['HighEngagementPercentage']

# Set up the bar width and positions
bar_width = 0.35
x = np.arange(len(categories))  # Positions for the bars

# Create the figure
plt.figure(figsize=(10, 6))

# Add bars for Average Session Duration
plt.bar(x - bar_width/2, session_durations, bar_width, label='Avg Session Duration (min)', color='blue', alpha=0.7)

# Add bars for High Engagement Percentage
plt.bar(x + bar_width/2, engagement_percentages, bar_width, label='High Engagement (%)', color='green', alpha=0.7)

# Add labels and titles
plt.title("Effect of Game Difficulty on Session Duration and Engagement", fontsize=14)
plt.xlabel("Game Difficulty", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.xticks(x, categories, rotation=45)  # Set category names on x-axis
plt.legend(loc='upper right')

# Final layout adjustments
plt.tight_layout()
plt.show()



# In[283]:


#  Which Difficulties Are Associated with Higher Engagement Levels?
query = """
SELECT 
    GameDifficulty,
    EngagementLevel,
    COUNT(*) AS PlayerCount,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY GameDifficulty), 2) AS EngagementPercentage
FROM player_data
GROUP BY GameDifficulty, EngagementLevel
ORDER BY GameDifficulty, EngagementLevel;
"""
result_engagement = pd.read_sql(query, conn)
print(result_engagement)
import seaborn as sns

# Pivot the data for visualization
pivot_result = result_engagement.pivot(index='GameDifficulty', columns='EngagementLevel', values='EngagementPercentage')

# Stacked bar chart
pivot_result.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Engagement Levels by Game Difficulty", fontsize=14)
plt.xlabel("Game Difficulty", fontsize=12)
plt.ylabel("Engagement Percentage (%)", fontsize=12)
plt.legend(title="Engagement Level", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[284]:


#Building a machine learning model to predict EngagementLevel based on other features.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data from SQLite
query = "SELECT * FROM player_data;"
df = pd.read_sql(query, conn)

# Encode categorical variables
label_encoder = LabelEncoder()
df['EngagementLevel'] = label_encoder.fit_transform(df['EngagementLevel'])  # High=2, Medium=1, Low=0
df = pd.get_dummies(df, columns=['GameDifficulty'], drop_first=True)  # One-hot encoding for GameDifficulty

# Select features and target
features = ['PlayTimeHours', 'SessionsPerWeek', 'AchievementsUnlocked', 
            'AvgSessionDurationMinutes', 'PlayerLevel'] + [col for col in df.columns if 'GameDifficulty_' in col]
X = df[features]
y = df['EngagementLevel']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[285]:


# Create a DataFrame for classification report
classification_data = {
    "Class": ["Low (0)", "Medium (1)", "High (2)"],
    "Precision": [0.92, 0.91, 0.91],
    "Recall": [0.88, 0.88, 0.95],
    "F1-Score": [0.90, 0.90, 0.93]
}

df_classification = pd.DataFrame(classification_data)

# Plot bar chart
x = np.arange(len(df_classification["Class"]))  # Label locations
width = 0.25  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

# Bars for Precision, Recall, and F1-Score
bars_precision = ax.bar(x - width, df_classification["Precision"], width, label="Precision", color="skyblue")
bars_recall = ax.bar(x, df_classification["Recall"], width, label="Recall", color="lightgreen")
bars_f1 = ax.bar(x + width, df_classification["F1-Score"], width, label="F1-Score", color="salmon")

# Add labels and title
ax.set_xlabel("Engagement Class", fontsize=12)
ax.set_ylabel("Scores", fontsize=12)
ax.set_title("Classification Report Metrics by Engagement Class", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(df_classification["Class"], fontsize=10)
ax.legend(fontsize=10)

# Add values on top of the bars
def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset for text
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

add_values(bars_precision)
add_values(bars_recall)
add_values(bars_f1)

plt.tight_layout()
plt.show()


# In[286]:


# Extract feature importances
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title("Feature Importance for Predicting Engagement Level", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance at the top
plt.tight_layout()
plt.show()


# In[287]:


# Load the data from SQLite
query = "SELECT * FROM player_data;"
df = pd.read_sql(query, conn)

# One-hot encode the GameGenre column
df = pd.get_dummies(df, columns=['GameGenre'], drop_first=True)  # One-hot encoding for GameGenre

# Select relevant features and target
features = [col for col in df.columns if 'GameGenre_' in col]  # All game genres
features += ['InGamePurchases', 'AchievementsUnlocked']  # Add the numeric features
X = df[features]
y = df['EngagementLevel']

# Encode the target (EngagementLevel)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)  # High=2, Medium=1, Low=0

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_model.predict(X_test)

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# In[288]:


# Generate classification report as a dictionary
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

# Convert the report into a DataFrame
report_df = pd.DataFrame(report).transpose()

# Exclude the "accuracy" row
report_df = report_df.drop(index=['accuracy'])

# Create a grouped bar chart for Precision, Recall, and F1-Score
x = np.arange(len(report_df.index))  # Label locations
width = 0.25  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

# Bars for Precision, Recall, and F1-Score
precision_bars = ax.bar(x - width, report_df['precision'], width, label='Precision', color='skyblue')
recall_bars = ax.bar(x, report_df['recall'], width, label='Recall', color='lightgreen')
f1_bars = ax.bar(x + width, report_df['f1-score'], width, label='F1-Score', color='salmon')

# Add labels, title, and legend
ax.set_xlabel("Classes", fontsize=12)
ax.set_ylabel("Scores", fontsize=12)
ax.set_title("Classification Metrics by Class", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(report_df.index, fontsize=10, rotation=45)
ax.legend(fontsize=10)

# Add values on top of the bars
def add_bar_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # Offset for text
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_bar_values(precision_bars)
add_bar_values(recall_bars)
add_bar_values(f1_bars)

#


# In[289]:


# Extract feature importances
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title("Feature Importance for Predicting Engagement Level", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.gca().invert_yaxis()  # Most important feature at the top
plt.tight_layout()
plt.show()


# In[290]:


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect("gaming_data.db")

# Load the data from SQLite
query = "SELECT * FROM player_data;"
df = pd.read_sql(query, conn)

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['EngagementLevel'] = label_encoder.fit_transform(df['EngagementLevel'])  # High=2, Medium=1, Low=0
df = pd.get_dummies(df, columns=['GameDifficulty'], drop_first=True)  # One-hot encoding for GameDifficulty

# Select features and target
features = ['PlayTimeHours', 'SessionsPerWeek', 'AchievementsUnlocked', 
            'AvgSessionDurationMinutes', 'PlayerLevel'] + [col for col in df.columns if 'GameDifficulty_' in col]
X = df[features].copy()  # Ensure a clean copy
y = df['EngagementLevel'].values

# 1. Check for NaN or infinite values
X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
X = X.fillna(0)  # Replace NaN with 0 (or use another strategy, e.g., mean imputation)

# 2. Normalize features (ensure they're numeric)
X = X.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # Standardization

# Define sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression implementation
def logistic_regression(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Initialize weights
    costs = []

    for _ in range(iterations):
        # Compute predictions
        z = np.dot(X, theta)
        predictions = sigmoid(z)

        # Compute cost
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
        costs.append(cost)

        # Gradient descent update
        gradient = (1/m) * np.dot(X.T, (predictions - y))
        theta -= learning_rate * gradient

    return theta, costs

# One-vs-All Classification: Train separate logistic regression models for each class
num_classes = len(np.unique(y))
models = []

for class_label in range(num_classes):
    # Create binary target for current class
    y_binary = (y == class_label).astype(int)

    # Train logistic regression using gradient descent
    theta, costs = logistic_regression(X.values, y_binary, learning_rate=0.01, iterations=1000)
    models.append(theta)

print("Logistic regression models trained successfully!")

# Evaluate the models
from sklearn.metrics import classification_report, accuracy_score

# Predictions
predictions = []
for theta in models:
    predictions.append(sigmoid(np.dot(X.values, theta)))

predictions = np.array(predictions).T
y_pred = np.argmax(predictions, axis=1)

# Print evaluation results
print("Accuracy:", accuracy_score(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred, target_names=label_encoder.classes_))

# Create classification metrics bar chart
report = classification_report(y, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Plot precision, recall, and F1-score
x = np.arange(len(report_df.index[:-3]))  # Exclude support and avg rows
width = 0.25  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))
precision_bars = ax.bar(x - width, report_df['precision'][:-3], width, label='Precision', color='skyblue')
recall_bars = ax.bar(x, report_df['recall'][:-3], width, label='Recall', color='lightgreen')
f1_bars = ax.bar(x + width, report_df['f1-score'][:-3], width, label='F1-Score', color='salmon')

ax.set_xlabel("Classes", fontsize=12)
ax.set_ylabel("Scores", fontsize=12)
ax.set_title("Classification Metrics by Class", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(label_encoder.classes_, rotation=45, fontsize=10)
ax.legend(fontsize=10)

def add_values(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

add_values(precision_bars)
add_values(recall_bars)
add_values(f1_bars)

plt.tight_layout()
plt.show()

# Plot feature importances
importances = np.mean(np.abs(models), axis=0)  # Average absolute weights across models
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title("Feature Importance for Predicting Engagement Level", fontsize=14)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance at the top
plt.tight_layout()
plt.show()


# In[ ]:




