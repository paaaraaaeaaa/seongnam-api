from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

app = FastAPI(title="성남시 교통사고 위험 예측 API")
model = joblib.load('seongnam_model.pkl')

# 프론트엔드에서 받을 데이터 양식
class PredictRequest(BaseModel):
    target_name: str
    speeding: float
    center_line: float
    signal: float
    safe_dist: float
    duty: float
    pedestrian: float
    etc: float

@app.get("/")
def read_root():
    return {"message": "성남시 교통사고 위험 예측 24시간 API 서버 정상 작동 중!"}

@app.post("/predict")
def predict_risk(data: PredictRequest):
    try:
        input_df = pd.DataFrame([{
            '대상사고 구분명': data.target_name,
            '과속': data.speeding,
            '중앙선 침범': data.center_line,
            '신호위반': data.signal,
            '안전거리 미확보': data.safe_dist,
            '안전운전 의무 불이행': data.duty,
            '보행자 보호의무 위반': data.pedestrian,
            '기타': data.etc
        }])
        
        risk_score = model.predict(input_df)[0]
        
        return {
            "위험지수_결과": round(float(risk_score), 2),
            "상태": "위험" if risk_score >= 50 else "안전"
        }
    except Exception as e:
        return {"에러_원인": str(e), "상세_설명": traceback.format_exc()}
