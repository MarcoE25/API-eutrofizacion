from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.predict import predecir_P_tot, predecir_N_tot

app = FastAPI(title="API para predicción del Nitrogeno y Fosforo total")

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EntradaDatos(BaseModel):
    N_NH3: float
    N_NO2: float
    N_NO3: float
    SDT: float
    pH_CAMPO: float
    TURBIEDAD: float
    TEMP_AMB: float

@app.post("/predict")
def predict_both(entrada: EntradaDatos):
    datos = entrada.dict()
    resultado_P = predecir_P_tot(datos)
    resultado_N = predecir_N_tot(datos)
    return {
        "P_TOT": resultado_P,
        "N_TOT": resultado_N
    }

