
import pandas as pd
import mlflow
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sys
from src.decorators import log_call, timing

@log_call
@timing
def train_regression(task):
    df = pd.read_csv(f"../datasets/{task}.csv")
    X, y = df["text"], df["label"]
    
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X)
    
    model = LogisticRegression()
    model.fit(X_train, y)
    
    with open(f"../models/regression_{task}.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"../models/vectorizer_{task}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    mlflow.start_run()
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", model.score(X_train, y))
    mlflow.end_run()
    print(f"Entraînement terminé pour {task} avec régression.")
'''
if __name__ == "__main__":
    task = sys.argv[1]  # Exemple : darija, sentiment, spam, toxic
    train_regression(task)'''
