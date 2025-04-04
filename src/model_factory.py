
from abc import ABC, abstractmethod
from transformers import pipeline
import pickle


# --- Factory Pattern pour les modèles de classification ---
class ClassificationModel(ABC):
    @abstractmethod
    def predict(self, text: str):
        pass

class BertModel(ClassificationModel):
    def __init__(self):
        self.pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    def predict(self, text: str):
        return self.pipeline(text)

class RegressionModel(ClassificationModel):
    def __init__(self, task):
        with open(f"../models/regression_{task}.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(f"../models/vectorizer_{task}.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
    def predict(self, text: str):
        return self.model.predict(self.vectorizer.transform([text]))[0]

class LSTMModel(ClassificationModel):
    def __init__(self, task):
        import torch
        from src.train_lstm import LSTMClassifier  # Assure-toi que train_lstm.py contient la classe LSTMClassifier
        self.model = LSTMClassifier(50, 128, 2)
        self.model.load_state_dict(torch.load(f"../models/lstm_{task}.pth"))
        self.model.eval()
    def predict(self, text: str):
        # Ici, il faut pré-traiter le texte pour obtenir une représentation (exemple simplifié)
        import torch
        # Simuler une entrée de taille (1, 100, 50)
        dummy_input = torch.randn(1, 100, 50)
        with torch.no_grad():
            output = self.model(dummy_input)
        prediction = torch.argmax(output).item()
        return prediction

class ModelFactory:
    @staticmethod
    def get_model(model_type, task) -> ClassificationModel:
        if model_type == "bert":
            return BertModel()
        elif model_type == "regression":
            return RegressionModel(task)
        elif model_type == "lstm":
            return LSTMModel(task)
        else:
            raise ValueError("Modèle inconnu. Choisissez 'bert', 'regression' ou 'lstm'.")
