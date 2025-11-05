import requests

url = "http://127.0.0.1:8000/predict_child"

sample_data = {
    "Index_of_Multiple_Deprivation_Rank": 1,  # most deprived
    "IMD_Decile": 1,
    "AgeAtStartOfSpell": 45,  # advanced maternal age
    "WeightMeasured": 110,
    "Height": 150,
    "Body_Mass_Index_at_Booking": 48.9,  # very high obesity
    "Obese": 1,
    "Ethnicity": "Unknown",
    "Risk_Factors": "Obesity-BMI>=35, Domestic abuse, Smoking",
    "AntenatalMedicalFactors": "Mental health, Previous uterine surgery",
    "PreviousObstetricHistory": "Early pre-term birth <34 weeks, Fetal loss",
    "Parity": 4,
    "Gravida": 5,
    "Glucoselevelblood": 13.0,
    "GlucoseToleranceTest": "Abnormal",
    "Glucoselevel0minblood": 12.0,
    "Glucoselevel120minblood": 15.0,
    "FolicAcidDose": "No",
    "SystolicBloodPressureCuff": 160,
    "Diastolic_Blood_Pressure": 100,
    "VitaminDlevelblood": 15,
    "O_Thyroidfunctionblood": 3.5,
    "Delivery_Outcome": "Still Birth",
    "OnsetofLabourMethod": "Medical Induction",
    "Contraction_frequency_prior_to_delivery": 15,
    "PrimaryIndicationforCaesarean": "Fetal anomalies",
    "Category_Caesarean_Section": "Cat 1",
    "Perineal_care": "Third",
    "EstimatedTotalBloodLoss": 1200,
    "Gestation": 28,
    "Severely_Premature": "Yes",
    "Gestation_Days": 196,
    "Gestation_at_booking_Weeks": 10,
    "No_Of_previous_Csections": 3,
    "BabyBirthWeight": 900,
    "Presence_of_meconium": "Yes",
    "BW_Centile": 10,
    "Shoulder_Dystocia": "Yes",
    "LOS_mother_after_delivery": 20,
    "Sex": "Female",
    "Still_Birth": "Yes",   
    "TotalApgarScoreat1minutes": 2,
    "APGAR_Score_5": 3,
    "TotalApgarScoreat10minutes": 4,
    "Maternity_Month": "December",
}

response = requests.post(url, json=sample_data)
print("Child API Response:", response.json())