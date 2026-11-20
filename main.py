from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("logistic.pkl")


class Result(BaseModel):
    StudyHours: float


@app.get("/")
def home():
    return {"msg": "uvicorn is running"}


@app.post("/predresult")
def datainsert(ds: Result):

    data = [[ds.StudyHours]]

    probability = model.predict_proba(data)

    pass_probability = probability[0][1]

    threshold = 0.5

    if pass_probability >= threshold:
        result = "Pass"
    else:
        result = "Fail"

    return {
        "StudyHours": ds.StudyHours,
        "Probability": round(float(pass_probability), 2),
        "Threshold": threshold,
        "Result": result
    }