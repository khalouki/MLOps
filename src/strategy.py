
from abc import ABC, abstractmethod
from src.model_factory import BertModel, RegressionModel, LSTMModel

class ClassificationStrategy(ABC):
    @abstractmethod
    def classify(self, text: str):
        pass

class BertStrategy(ClassificationStrategy):
    def __init__(self):
        self.model = BertModel()
    def classify(self, text: str):
        return self.model.predict(text)

class RegressionStrategy(ClassificationStrategy):
    def __init__(self, task):
        self.model = RegressionModel(task)
    def classify(self, text: str):
        return self.model.predict(text)

class LSTMStrategy(ClassificationStrategy):
    def __init__(self, task):
        self.model = LSTMModel(task)
    def classify(self, text: str):
        return self.model.predict(text)

