overview: This project implements a movie recommendation system using a hybrid approach that combines content-based filtering (via TF-IDF + cosine similarity) and collaborative filtering (via KNN). Additionally, supervised learning models such as Logistic Regression, Naive Bayes, KNN Classifier, and XGBoost are applied for classification and regression tasks.

A user-friendly Streamlit GUI allows users to interactively explore recommendations based on genre, rating filters, and selected movies.

Multiple models were trained and evaluated, and their performance was compared using appropriate metrics such as RMSE, MAE, Accuracy, Precision, Recall, and F1-Score to determine their effectiveness in recommendation quality.

project structure :

 ┣  GUI_app.py                 # Streamlit interface
 ┣  ML_project.ipynb   # Model training and evaluation
 ┣  movielens_cleaned.csv      # Preprocessed dataset (sampled from MovieLens 20M)
 dataset folder have ratings and movies.csv
 ┣  requirements.txt           # Python dependencies
 ┗  README.md 


Models Used :
Content-Based Filtering: TF-IDF on genres + cosine similarity

Collaborative Filtering: KNN with cosine distance (Surprise & NearestNeighbors)

Classification Models: Logistic Regression, Naive Bayes, KNN

Regression Model: XGBoost

Dataset :
Based on MovieLens 20M, sampled to 100,000 rows for efficiency.

Used: ratings.csv, movies.csv

Preprocessed and merged into movielens_cleaned.csv


Evaluation Metrics :
Regression: RMSE, MAE

Classification: Accuracy, Precision, Recall, F1-Score
visualization for comparison of evaluation metrics


GUI (Streamlit App) :
Choose movie, genre, and rating threshold

Get 10 recommendations using:

--> Content-Based filtering

---> User-Based collaborative filtering

Clean interface with filter options



How to Run the Project :
Model Training & Evaluation
Open the file ML_project.ipynb in Google Colab or visual studio or jupyter notebook

Run all cells to:

Load and preprocess the MovieLens data

Train all models (Logistic Regression, Naive Bayes, KNN, XGBoost)

Evaluate results using RMSE, MAE, Precision, Recall, and F1-Score



Running the GUI (Streamlit App) :
Open GUI_app.py in Visual Studio Code

Make sure all required packages are installed (see requirements.txt)
streamlit run GUI_app.py
The app opens in your browser. Use it to:

Select a genre and rating filter

Choose a movie

View recommendations from both content-based and user-based models