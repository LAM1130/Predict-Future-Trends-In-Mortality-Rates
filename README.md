# Data Mining Project - Group 07

**A complete data mining and machine learning project with Streamlit deployment.**

---

## 📋 Project Overview

This project performs data mining, exploratory data analysis (EDA), and machine learning modeling on the dataset `Dataset_Group07.csv`. It includes:

- Data preprocessing and visualization (using pandas, seaborn, matplotlib)
- Machine learning models (scikit-learn + XGBoost)
- Interactive web deployment using **Streamlit**

---

## 🛠️ Required Libraries

Install all dependencies before running the project:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost streamlit
```

## 🚀 How to Run the Project
You can run this project in Google Colab, Jupyter Notebook, or via Streamlit (recommended for the final deployment).
1. Using Google Colab (Recommended for quick testing)

Upload the project files to your Google Drive:
Place the .ipynb file and the dataset in this exact folder:
/content/drive/MyDrive/Data_Mining_G7/

Open the notebook in Google Colab.
Mount your Google Drive and run the cells.

Dataset file must be named: Dataset_Group07.csv

2. Using Jupyter Notebook

Place both the project notebook (.ipynb) and the dataset in the same folder.
Dataset must be named: Dataset_Group07.csv
Open Jupyter Notebook and run the project.

3. Running the Streamlit Deployment (Interactive Web App)
Bash# 1. Make sure you are in the project folder
cd path/to/your/project

# 2. Run the deployment script
streamlit run Deployment_Group07.py
Note: If you haven't installed Streamlit yet:Bashpip install streamlit
The web app will open automatically in your browser.
