from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from model_utils import (
    preprocess_gdm_input,
    preprocess_child_input,
    get_gdm_precautions,
    get_child_precautions,
)

app = FastAPI()

origins = [
    "http://127.0.0.1:8080",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gdm_model = joblib.load("../models/gdm_classifier_small.joblib")
child_model = joblib.load("../models/child_outcome_model_full.joblib")


class GDMInput(BaseModel):
    AgeAtStartOfSpell: float
    WeightMeasured: float
    Height: float
    BodyMassIndexAtBooking: float
    Obese: str
    Ethnicity: str
    Glucoselevelblood: float


class ChildInput(BaseModel):
    Index_of_Multiple_Deprivation_Rank: float
    IMD_Decile: int
    AgeAtStartOfSpell: float
    WeightMeasured: float
    Height: float
    Body_Mass_Index_at_Booking: float
    Obese: int
    Ethnicity: str
    Risk_Factors: str
    AntenatalMedicalFactors: str
    PreviousObstetricHistory: str
    Parity: int
    Gravida: int
    Glucoselevelblood: float
    GlucoseToleranceTest: str
    Glucoselevel0minblood: float
    Glucoselevel120minblood: float
    FolicAcidDose: str
    SystolicBloodPressureCuff: float
    Diastolic_Blood_Pressure: float
    VitaminDlevelblood: float
    O_Thyroidfunctionblood: float
    Delivery_Outcome: str
    OnsetofLabourMethod: str
    Contraction_frequency_prior_to_delivery: float
    PrimaryIndicationforCaesarean: str
    Category_Caesarean_Section: str
    Perineal_care: str
    EstimatedTotalBloodLoss: float
    Gestation: int
    Severely_Premature: str
    Gestation_Days: int
    Gestation_at_booking_Weeks: float
    No_Of_previous_Csections: int
    BabyBirthWeight: float
    Presence_of_meconium: str
    BW_Centile: float
    Shoulder_Dystocia: str
    LOS_mother_after_delivery: float
    Sex: str
    Still_Birth: str
    TotalApgarScoreat1minutes: int
    APGAR_Score_5: int
    TotalApgarScoreat10minutes: int
    Maternity_Month: str


FEATURE_NAMES = [
    "Index_of_Multiple_Deprivation_Rank",
    "IMD_Decile",
    "AgeAtStartOfSpell",
    "WeightMeasured",
    "Height",
    "Body_Mass_Index_at_Booking",
    "Obese",
    "Ethnicity",
    "Risk_Factors",
    "AntenatalMedicalFactors",
    "PreviousObstetricHistory",
    "Parity",
    "Gravida",
    "Glucoselevelblood",
    "GlucoseToleranceTest",
    "Glucoselevel0minblood",
    "Glucoselevel120minblood",
    "FolicAcidDose",
    "SystolicBloodPressureCuff",
    "Diastolic_Blood_Pressure",
    "VitaminDlevelblood",
    "O_Thyroidfunctionblood",
    "Delivery_Outcome",
    "OnsetofLabourMethod",
    "Contraction_frequency_prior_to_delivery",
    "PrimaryIndicationforCaesarean",
    "Category_Caesarean_Section",
    "Perineal_care",
    "EstimatedTotalBloodLoss",
    "Gestation",
    "Severely_Premature",
    "Gestation_Days",
    "Gestation_at_booking_Weeks",
    "No_Of_previous_Csections",
    "BabyBirthWeight",
    "Presence_of_meconium",
    "BW_Centile",
    "Shoulder_Dystocia",
    "LOS_mother_after_delivery",
    "Sex",
    "Still_Birth",
    "TotalApgarScoreat1minutes",
    "APGAR_Score_5",
    "TotalApgarScoreat10minutes",
    "Maternity_Month"
]


@app.post("/predict_gdm")
def predict_gdm(data: GDMInput):
    try:
        input_vector = preprocess_gdm_input(data)
        pred_proba = gdm_model.predict(input_vector)
        pred_label = int(pred_proba[0] > 0.5)
        precautions = get_gdm_precautions(pred_label)
        return {"GDM_Prediction": pred_label, "Probability": float(pred_proba[0]), "Precautions": precautions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict_child")
def predict_child(data: ChildInput):
    try:
        input_vector = preprocess_child_input(data)
        pred_proba = child_model.predict(input_vector)[0]
        pred_label = int(pred_proba > 0.5)
        precautions = get_child_precautions(pred_label)

        importances = child_model.feature_importance(importance_type='gain')
        top_indices = np.argsort(importances)[::-1][:5]

        top_features = {FEATURE_NAMES[idx]: float(importances[idx]) for idx in top_indices}

        return {
            "Child_OutcomePrediction": pred_label,
            "Probability": float(pred_proba),
            "Precautions": precautions,
            "TopFeatureImportances": top_features
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))