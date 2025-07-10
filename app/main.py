from fastapi import FastAPI
from pydantic import BaseModel
from app.predict import predecir_P_tot, predecir_N_tot


app = FastAPI(title="API para predicción del Nitrogeno y Fosforo total")

# Ajusta los campos de entrada según tus variables reales
class EntradaDatos(BaseModel):
    N_NH3:float
    N_NO2:float
    N_NO3:float
    SDT:float
    pH_CAMPO:float
    TURBIEDAD:float
    TEMP_AMB:float

@app.post("/predecir/P_tot")
def pred_P_tot(entrada: EntradaDatos):
    resultado = predecir_P_tot(entrada.dict())
    return {"modelo": "P_tot", "Valor estimado": resultado}

@app.post("/predecir/N_tot")
def pred_N_tot(entrada: EntradaDatos):
    resultado = predecir_N_tot(entrada.dict())
    return {"modelo": "N_tot", "Valor estimado": resultado}
