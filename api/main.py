
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from src.jwt_security import verify_token
from src.strategy import BertStrategy, RegressionStrategy, LSTMStrategy

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
app = FastAPI()

def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalide")
    return payload

class PredictRequest(BaseModel):
    task: str         # Ex: darija, sentiment, spam, toxic
    model_type: str   # "bert", "regression" ou "lstm"
    text: str

@app.post("/predict/")
def predict(request: PredictRequest, user: dict = Depends(get_current_user)):
    if request.model_type == "bert":
        strategy = BertStrategy()
    elif request.model_type == "regression":
        strategy = RegressionStrategy(request.task)
    elif request.model_type == "lstm":
        strategy = LSTMStrategy(request.task)
    else:
        raise HTTPException(status_code=400, detail="Modèle inconnu")
    prediction = strategy.classify(request.text)
    return {"task": request.task, "model": request.model_type, "prediction": prediction}
