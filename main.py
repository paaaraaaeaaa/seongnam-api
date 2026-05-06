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
        # 🌟 1. AI 해킹: 모델이 학습할 때 기억해둔 '법정동코드'의 진짜 타입을 훔쳐옵니다!
        dong_input = data.dong
        try:
            ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            # cat_features = ['대상사고 구분명', '법정동코드'] 이므로 인덱스 1 추출
            expected_type = type(ohe.categories_[1][0])
            dong_input = expected_type(data.dong) # int든 float이든 str이든 강제로 맞춰버림!
        except Exception as type_e:
            dong_input = int(data.dong) # 만약 실패하면 정수형으로 시도

        # 2. 완벽하게 타입이 맞춰진 데이터를 입력
        mapped_data = {
            "연도": int(data.year),
            "법정동코드": str(data.dong)
            "대상사고 구분명": str(data.target_name),
            "과속": float(data.speeding),
            "중앙선 침범": float(data.center_line),
            "신호위반": float(data.signal),
            "안전거리 미확보": float(data.safe_dist),
            "안전운전 의무 불이행": float(data.duty),
            "보행자 보호의무 위반": float(data.pedestrian),
            "기타": float(data.etc)
        }
        
        input_df = pd.DataFrame([mapped_data])
        prediction = model.predict(input_df)[0]
        risk_status = "위험" if prediction >= 50 else "안전"
        
        shap_dict = {}
        try:
            processed_data = model.named_steps['preprocessor'].transform(input_df)
            explainer = shap.TreeExplainer(model.named_steps['regressor'])
            shap_values = explainer.shap_values(processed_data)
            
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            target_features = ["과속", "중앙선 침범", "신호위반", "안전거리 미확보", "안전운전 의무 불이행", "보행자 보호의무 위반", "기타"]
            
            for name, val in zip(feature_names, shap_values[0]):
                for target in target_features:
                    if target in name:
                        shap_dict[target] = round(float(val), 2)
        except Exception as shap_e:
            print("SHAP 에러:", shap_e)

        return {
            "위험지수_결과": round(float(prediction), 2),
            "상태": risk_status,
            "SHAP_분석": shap_dict
        }
        
    except Exception as e:
        return {"error": str(e), "위험지수_결과": -1}
