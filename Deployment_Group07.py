import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


data=pd.read_csv("C:\Users\lam\Documents\y2s2\Data Mining\Dm Project\Dataset_Group07.csv")

# Display dataset
st.header("Dataset")
st.write(data)

# Display missing values
st.subheader("Missing Values")
missing_values = data.isnull().sum()
st.write(missing_values)

# Display duplicated data
st.subheader("Duplicated Data")
duplicated_data = sum(data.duplicated())
st.write(f"Number of Duplicated Data: {duplicated_data}")

st.header("Exploratory Data Analysis (EDA)")
# Display data description
st.subheader("Data Description")
st.write(data.describe())

# Display data types
st.subheader("Data Types")
st.write(data.dtypes)

# Plot number of deaths for each cause per year
st.subheader("1. What are the number of deaths for each cause per every year?")
cause_cols = data.columns[3:]

deaths_by_year = data.groupby('Year')[cause_cols].sum()

for cause in cause_cols:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(deaths_by_year.index, deaths_by_year[cause])
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Deaths')
    ax.set_title(f'Number of Deaths by {cause} per Year')
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.subheader("2.Top 5 countries with highest average number of cardiovascular diseases")
top_countries_cardiovascular = data.groupby('Country/Territory')['Cardiovascular Diseases'].mean().sort_values(ascending=False).head(5)

fig, ax = plt.subplots(figsize=(10, 6))
top_countries_cardiovascular.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Top 5 Countries with Highest Average Number of Cardiovascular Diseases')
ax.set_xlabel('Country')
ax.set_ylabel('Average Number of Deaths')
st.pyplot(fig)

st.write("3.Number of road injuries over the years in Afghanistan")
afghanistan_road_injuries = data[data['Country/Territory'] == 'Afghanistan'][['Year', 'Road Injuries']]

# Generate and display the plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(afghanistan_road_injuries['Year'], afghanistan_road_injuries['Road Injuries'], marker='o', linestyle='-', color='orange')
ax.set_title('Road Injuries in Afghanistan Over the Years')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Road Injuries')
ax.grid(True)
st.pyplot(fig)

st.write("4.Is there a correlation between self harm and the prevalence of digestive disease?")
# Select relevant columns for scatterplot and regression plot
data_sHarm_Diges = data[["Self-harm", "Digestive Diseases"]]

# Calculate correlation coefficient
correlation = data_sHarm_Diges.corr(method='spearman')  # Using Spearman's rank correlation

# Create scatterplot with correlation coefficient displayed
st.subheader("Scatterplot with Correlation Coefficient")
fig, ax = plt.subplots()
sns.scatterplot(x="Self-harm", y="Digestive Diseases", data=data_sHarm_Diges, ax=ax)

# Add correlation coefficient text annotation
ax.text(0.05, 0.95, f"Correlation: {correlation.iloc[0, 1]:.2f}",
        ha='left', va='top', transform=ax.transAxes, fontsize=12)

st.pyplot(fig)

# Create regression plot
st.subheader("Regression Plot")
fig, ax = plt.subplots()
sns.regplot(x="Self-harm", y="Digestive Diseases", data=data_sHarm_Diges, ax=ax)
st.pyplot(fig)

# Select numerical data for correlation matrix
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate correlation matrix
cor = numerical_data.corr()

# Create heatmap
st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(cor, annot=False, cmap='coolwarm', ax=ax)
plt.title('Correlation Matrix')
st.pyplot(fig)

# Splitting the data into features and target
# Target variable: Total deaths (sum of all deaths due to various causes)
data['Total_Deaths'] = data.iloc[:, 3:].sum(axis=1)

# Feature selection
features = ['Year'] + list(data.columns[3:-1])
X = data[features]
y = data['Total_Deaths']

#Data Normalization
st.header("Data Normalization")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

st.header("Data Mining(Random Forest)")
# Model training
# Training the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_score = model.score(X_test, y_test)
st.header("Model Training and Evaluation")
st.subheader("Model Evaluation Results")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"R^2 Score: {r2_score}")

# Actual vs Predicted values
st.subheader("Actual vs Predicted Values")
comparison_df = pd.DataFrame({'Actual Values': y_test.values, 'Predicted Values': y_pred})
st.write(comparison_df.head(10))

# Scatter plot of actual vs predicted values
st.subheader("Scatter Plot of Actual vs Predicted Values")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(comparison_df['Actual Values'], comparison_df['Predicted Values'], alpha=0.6)
ax.plot([comparison_df['Actual Values'].min(), comparison_df['Actual Values'].max()],
         [comparison_df['Actual Values'].min(), comparison_df['Actual Values'].max()],
         color='red', linestyle='--', linewidth=2)
ax.set_title('Actual vs Predicted Values')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
st.pyplot(fig)

# Residual plot
st.subheader("Residual Plot")
comparison_df['Residuals'] = comparison_df['Actual Values'] - comparison_df['Predicted Values']
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(comparison_df['Predicted Values'], comparison_df['Residuals'], alpha=0.6)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residual Plot')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
importance = model.feature_importances_

# Creating a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Sorting the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest')
plt.gca().invert_yaxis()
plt.show()
st.pyplot(plt)

st.header("Data Mining(Linear Regression)")
# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Making predictions
y_pred = model.predict(X_test)

# Making predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_score = model.score(X_test, y_test)
st.header("Model Training and Evaluation")
st.subheader("Model Evaluation Results")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"R^2 Score: {r2_score}")

# Actual vs Predicted values
st.subheader("Actual vs Predicted Values")
comparison_df = pd.DataFrame({'Actual Values': y_test.values, 'Predicted Values': y_pred})
st.write(comparison_df.head(10))

# Scatter plot of actual vs predicted values
st.subheader("Scatter Plot of Actual vs Predicted Values")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(comparison_df['Actual Values'], comparison_df['Predicted Values'], alpha=0.6)
ax.plot([comparison_df['Actual Values'].min(), comparison_df['Actual Values'].max()],
         [comparison_df['Actual Values'].min(), comparison_df['Actual Values'].max()],
         color='red', linestyle='--', linewidth=2)
ax.set_title('Actual vs Predicted Values')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
st.pyplot(fig)

# Residual plot
st.subheader("Residual Plot")
comparison_df['Residuals'] = comparison_df['Actual Values'] - comparison_df['Predicted Values']
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(comparison_df['Predicted Values'], comparison_df['Residuals'], alpha=0.6)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residual Plot')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
# Get the coefficients and corresponding feature names
importance = model.coef_

# Creating a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Sorting the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest')
plt.gca().invert_yaxis()
plt.show()
st.pyplot(plt)

st.header("Data Mining(XG Boost)")
# Training and fitting the XGBoost model
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2_score = model.score(X_test, y_test)
st.header("Model Training and Evaluation")
st.subheader("Model Evaluation Results")
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
st.write(f"R^2 Score: {r2_score}")

# Actual vs Predicted values
st.subheader("Actual vs Predicted Values")
comparison_df = pd.DataFrame({'Actual Values': y_test.values, 'Predicted Values': y_pred})
st.write(comparison_df.head(10))

# Scatter plot of actual vs predicted values
st.subheader("Scatter Plot of Actual vs Predicted Values")
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(comparison_df['Actual Values'], comparison_df['Predicted Values'], alpha=0.6)
ax.plot([comparison_df['Actual Values'].min(), comparison_df['Actual Values'].max()],
         [comparison_df['Actual Values'].min(), comparison_df['Actual Values'].max()],
         color='red', linestyle='--', linewidth=2)
ax.set_title('Actual vs Predicted Values')
ax.set_xlabel('Actual Values')
ax.set_ylabel('Predicted Values')
st.pyplot(fig)

# Residual plot
st.subheader("Residual Plot")
comparison_df['Residuals'] = comparison_df['Actual Values'] - comparison_df['Predicted Values']
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(comparison_df['Predicted Values'], comparison_df['Residuals'], alpha=0.6)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residual Plot')
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
st.pyplot(fig)

# Feature Importance
st.subheader("Feature Importance")
importance = model.feature_importances_

# Creating a DataFrame for visualization
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Sorting the features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Random Forest')
plt.gca().invert_yaxis()
plt.show()
st.pyplot(plt)