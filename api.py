from fastapi import FastAPI
import uvicorn
import joblib
from pydantic import BaseModel


app = FastAPI()
rf_model = joblib.load("rf_car_quality.pkl")


class CarFeatures(BaseModel):
    buying_price: int
    maintenance_cost: int
    number_of_doors: int
    number_of_persons: int
    lug_boot: int
    safety: int


@app.post("/araba_degerlendir")
def tahmin_et(car: CarFeatures):
    
   
    data = [[
        car.buying_price,
        car.maintenance_cost,
        car.number_of_doors,
        car.number_of_persons,
        car.lug_boot,
        car.safety
    ]]
    
    
    predict = rf_model.predict(data)
    
    
    quality_dict = {0: "Kabul Edilemez", 1: "İdare Eder", 2: "İyi", 3: "Çok İyi"}
    result_text = quality_dict[int(predict[0])]
    
    return {
        "model_mesaji": "Araba başarıyla analiz edildi.", 
        "kalite_sinifi": result_text
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

