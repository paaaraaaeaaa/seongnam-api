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
        # 🌟 진실의 방: 팀원 1의 전처리 파이프라인과 완벽하게 일치하는 데이터 매핑
        mapped_data = {
            "연도": int(data.year),
            # 핵심!! 문자열(str)이 아닌 반드시 정수(int)로 변환해서 모델에 줘야 합니다!
            "법정동코드": int(data.dong), 
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
        
        # 모델 예측 (이제 지역 코드를 완벽하게 인식합니다)
        prediction = model.predict(input_df)[0]
        risk_status = "위험" if prediction >= 50 else "안전"
        
        # SHAP 연산 (폭포수 그래프 항목 무제한 추출기)
        shap_dict = {}
        try:
            processed_data = model.named_steps['preprocessor'].transform(input_df)
            explainer = shap.TreeExplainer(model.named_steps['regressor'])
            shap_values = explainer.shap_values(processed_data)
            
            # ColumnTransformer가 뱉어낸 'num__과속' 같은 이름을 추적
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            target_features = ["과속", "중앙선 침범", "신호위반", "안전거리 미확보", "안전운전 의무 불이행", "보행자 보호의무 위반", "기타"]
            
            for name, val in zip(feature_names, shap_values[0]):
                for target in target_features:
                    if target in name:
                        shap_dict[target] = round(float(val), 2)
        except Exception as shap_e:
            print("SHAP 매핑 에러:", shap_e)

        return {
            "위험지수_결과": round(float(prediction), 2),
            "상태": risk_status,
            "SHAP_분석": shap_dict
        }
        
    except Exception as e:
        return {"error": str(e), "위험지수_결과": -1}
