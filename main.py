from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import shap
import traceback

app = FastAPI(title="성남시 예측 및 XAI(SHAP) API")
model = joblib.load('seongnam_model.pkl')

class PredictRequest(BaseModel):
    year: int
    dong: str
    target_name: str
    speeding: float
    center_line: float
    signal: float
    safe_dist: float
    duty: float
    pedestrian: float
    etc: float

@app.post("/predict")
def predict_risk(data: PredictRequest):
    try:
        # 1. 입력 데이터
        input_df = pd.DataFrame([{
            '연도': data.year, '법정동코드': data.dong, '대상사고 구분명': data.target_name,
            '과속': data.speeding, '중앙선 침범': data.center_line, '신호위반': data.signal,
            '안전거리 미확보': data.safe_dist, '안전운전 의무 불이행': data.duty,
            '보행자 보호의무 위반': data.pedestrian, '기타': data.etc
        }])
        
        # 2. 위험 지수 예측
        risk_score = model.predict(input_df)[0]
        
        # 3. 🌟 SHAP (XAI) 연산: 왜 이 점수가 나왔는지 분석
        preprocessor = model.named_steps['preprocessor']
        regressor = model.named_steps['regressor']
        
        # 입력값을 AI가 이해하는 숫자로 변환 후 SHAP 분석
        input_transformed = preprocessor.transform(input_df)
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(input_transformed)
        
        # 숫자형 데이터(위반 항목들)가 점수에 미친 영향(SHAP 값) 추출
        shap_impact = {
            "과속": float(shap_values[0][1]),
            "중앙선침범": float(shap_values[0][2]),
            "신호위반": float(shap_values[0][3]),
            "안전거리": float(shap_values[0][4]),
            "안전운전의무": float(shap_values[0][5]),
            "보행자보호": float(shap_values[0][6])
        }

        return {
            "위험지수_결과": round(float(risk_score), 2),
            "상태": "위험" if risk_score >= 50 else "안전",
            "SHAP_분석": shap_impact  # 프론트로 XAI 데이터 전송!
        }
    except Exception as e:
        return {"에러_원인": str(e), "상세_설명": traceback.format_exc()}
