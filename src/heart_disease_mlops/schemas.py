from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PatientRecord(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[57])
    sex: int = Field(..., ge=0, le=1, description="0=female, 1=male", examples=[1])
    cp: int = Field(..., ge=0, le=4, description="Chest pain type", examples=[2])
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure", examples=[140])
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol", examples=[241])
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl", examples=[0])
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result", examples=[1])
    thalach: float = Field(..., ge=50, le=250, description="Maximum heart rate", examples=[123])
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina", examples=[1])
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression", examples=[0.2])
    slope: int = Field(..., ge=0, le=3, description="Peak exercise ST slope", examples=[2])
    ca: float = Field(..., ge=0, le=4, description="Major vessels colored", examples=[0])
    thal: float = Field(..., ge=0, le=7, description="Thalassemia code", examples=[3])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "age": 57,
                "sex": 1,
                "cp": 2,
                "trestbps": 140,
                "chol": 241,
                "fbs": 0,
                "restecg": 1,
                "thalach": 123,
                "exang": 1,
                "oldpeak": 0.2,
                "slope": 2,
                "ca": 0,
                "thal": 3,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    risk_probability: float
    confidence: float
    model_name: str


class BatchPredictionRequest(BaseModel):
    records: list[PatientRecord]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]

