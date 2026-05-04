from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

app = FastAPI(title="성남시 교통사고 예측 API")

# 파일 불러오기
model = joblib.load('accident_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

class PredictRequest(BaseModel):
    year: int
    accident_type: str
    dong_code: int
    signal_violation: float
    speeding: float
    safe_distance: float

@app.get("/")
def read_root():
    return {"message": "서버 정상 작동 중!"}

@app.post("/predict")
def predict_accident(data: PredictRequest):
    try:
        input_df = pd.DataFrame([{
            '연도': data.year,
            '대상사고 구분명': data.accident_type,
            '법정동코드': data.dong_code,
            '신호위반': data.signal_violation,
            '과속': data.speeding,
            '안전거리 미확보': data.safe_distance
        }])
        
        input_df['대상사고 구분명'] = label_encoder.transform(input_df['대상사고 구분명'].astype(str))
        predicted_accidents = model.predict(input_df)[0]
        
        return {
            "예상_사고건수": round(predicted_accidents, 2),
            "위험도": "위험" if predicted_accidents >= 30 else "안전"
        }
    except Exception as e:
        # 🌟 에러가 나면 500으로 뻗는 대신, 화면에 에러 원인을 그대로 보여줍니다!
        return {
            "에러_원인": str(e),
            "상세_설명": traceback.format_exc()
        }
