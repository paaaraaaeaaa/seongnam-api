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
        # 🌟 함정 1 해결: 프론트에서 온 코드를 팀원 1이 학습했을 법한 '한글 동이름'으로 역변환!
        # (만약 프론트에서 이미 한글로 온다면 그대로 씁니다)
        dong_mapping = {
            "4113110100": "수진1동", "4113110200": "수진2동", "4113110300": "신흥1동",
            "4113110400": "신흥2동", "4113110500": "신흥3동", "4113110600": "단대동",
            "4113110700": "은행동", "4113110800": "양지동", "4113110900": "태평1동",
            "4113111000": "태평2동", "4113111100": "태평3동", "4113111200": "태평4동"
        }
        real_dong_name = dong_mapping.get(str(data.dong), str(data.dong))

        mapped_data = {
            "연도": int(data.year),
            "법정동코드": real_dong_name,
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
        
        # 모델 예측
        prediction = model.predict(input_df)[0]
        risk_status = "위험" if prediction >= 50 else "안전"
        
        # 🌟 함정 2 해결: 고정 인덱스(3, 5) 폐기! 변환된 컬럼 이름을 추적해서 진짜 SHAP 값을 매핑
        shap_dict = {}
        try:
            processed_data = model.named_steps['preprocessor'].transform(input_df)
            explainer = shap.TreeExplainer(model.named_steps['regressor'])
            shap_values = explainer.shap_values(processed_data)
            
            # 전처리기가 만든 새로운 컬럼 이름들을 가져옴
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            
            # 우리가 프론트엔드에 보여줄 항목만 쏙쏙 뽑아냄 (개수 제한 없음!)
            target_features = ["과속", "중앙선 침범", "신호위반", "안전거리 미확보", "안전운전 의무 불이행", "보행자 보호의무 위반", "기타"]
            
            for name, val in zip(feature_names, shap_values[0]):
                for target in target_features:
                    # '과속' 이라는 단어가 포함된 컬럼의 SHAP 값을 찾아서 저장
                    if target in name:
                        shap_dict[target] = round(float(val), 2)
                        
        except Exception as shap_e:
            print("SHAP 자동 매핑 에러:", shap_e)

        return {
            "위험지수_결과": round(float(prediction), 2),
            "상태": risk_status,
            "SHAP_분석": shap_dict
        }
        
    except Exception as e:
        return {"error": str(e), "위험지수_결과": -1}
