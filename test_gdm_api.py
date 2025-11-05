import requests

url = "http://127.0.0.1:8000/predict_gdm"

sample_data = {
    "AgeAtStartOfSpell": 35,
    "WeightMeasured": 75,
    "Height": 160,
    "BodyMassIndexAtBooking": 28.0,
    "Obese": "No",
    "Ethnicity": "GBR",
    "Glucoselevelblood": 6.0
}

response = requests.post(url, json=sample_data)
print("GDM API Response:", response.json())