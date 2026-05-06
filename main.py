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
        # 1. 프론트엔드 데이터를 모델이 아는 한글로 완벽 매핑
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
        
        input_df = pd.DataFrame([mapped_data])
        
        # 2. 모델 예측 실행 (이건 완벽하게 작동합니다)
        prediction = model.predict(input_df)[0]
        risk_status = "위험" if prediction >= 50 else "안전"
        
        # 3. SHAP 연산 (서버 터짐 방지 처리 완료!)
        shap_dict = {}
        try:
            # 🌟 [핵심 수정] 날것의 데이터가 아닌, 전처리(변환)가 끝난 데이터를 SHAP에 넣어야 합니다!
            processed_data = model.named_steps['preprocessor'].transform(input_df)
            explainer = shap.TreeExplainer(model.named_steps['regressor'])
            shap_values = explainer.shap_values(processed_data)
            
            # 총괄님이 예전에 보고서에 작성하셨던 추출 방식을 그대로 부활시켰습니다.
            # (만약 그래프에 항목을 더 추가하고 싶으시면 숫자를 바꿔서 추가하시면 됩니다!)
            shap_dict = {
                "과속": float(shap_values[0][3]),
                "신호위반": float(shap_values[0][5]),
                "안전거리": float(shap_values[0][6])
            }
        except Exception as shap_e:
            # 만약 그래프 계산에서 에러가 나더라도, 서버가 뻗지 않고 빈 그래프만 보냅니다.
            print("SHAP 계산 에러:", shap_e)

        # 4. JSON 형태로 최종 반환
        return {
            "위험지수_결과": round(float(prediction), 2),
            "상태": risk_status,
            "SHAP_분석": shap_dict
        }
        
    except Exception as e:
        # 최악의 경우 에러가 나도, 프론트가 0점 대신 에러 이유를 화면에 띄우도록 합니다.
        return {"error": str(e), "위험지수_결과": -1}
