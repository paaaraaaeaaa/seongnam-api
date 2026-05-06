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
        # 🌟 핵심 해결책: 프론트가 보낸 '영어' 변수를 모델이 아는 '한글' 컬럼명으로 완벽하게 1:1 매핑
        mapped_data = {
            "연도": int(data.year),
            "법정동코드": str(data.dong),
            "대상사고 구분명": str(data.target_name),
            "과속": float(data.speeding),
            "중앙선 침범": float(data.center_line),
            "신호위반": float(data.signal),
            "안전거리 미확보": float(data.safe_dist),
            "안전운전 의무 불이행": float(data.duty),
            "보행자 보호의무 위반": float(data.pedestrian),
            "기타": float(data.etc)
        }
        
        # 매핑된 진짜 사용자 입력 데이터를 데이터프레임으로 변환
        input_df = pd.DataFrame([mapped_data])
        
        # 모델 예측 실행
        prediction = model.predict(input_df)[0]
        risk_status = "위험" if prediction >= 50 else "안전"
        
        # SHAP 연산 수행
        explainer = shap.TreeExplainer(model.named_steps['regressor'])
        shap_values = explainer.shap_values(input_df)
        
        # 한글 컬럼명과 SHAP 수치를 묶어서 반환
        shap_dict = dict(zip(input_df.columns, shap_values[0]))
        
        # JSON 형태로 최종 반환
        return {
            "위험지수_결과": round(float(prediction), 2),
            "상태": risk_status,
            "SHAP_분석": shap_dict
        }
        
    except Exception as e:
        return {"error": str(e)}
