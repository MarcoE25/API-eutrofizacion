import os
import joblib
import numpy as np
import zipfile
import gdown

# Ruta base (un nivel arriba de este archivo)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ruta del ZIP y carpeta donde se extraen los modelos
ZIP_PATH = os.path.join(BASE_DIR, "random_forest1.zip")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Google Drive ID para el zip
ID_DRIVE = "1NzwJ0lsUDNmTk_ik4RpG2s-Dj40NoMK8"

def descargar_y_extraer_zip():
    if not os.path.exists(ZIP_PATH):
        print(f"Descargando {ZIP_PATH} desde Google Drive...")
        url = f"https://drive.google.com/uc?id={ID_DRIVE}"
        gdown.download(url, ZIP_PATH, quiet=False)
    else:
        print(f"Archivo ZIP ya existe en {ZIP_PATH}")

    # Extraer solo si la carpeta model está vacía o no existe
    if not os.path.exists(MODEL_DIR) or len(os.listdir(MODEL_DIR)) == 0:
        print(f"Extrayendo {ZIP_PATH} a {MODEL_DIR} ...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)
        print("Extracción completada.")
    else:
        print(f"La carpeta {MODEL_DIR} ya contiene archivos, no se extrae.")

# Inicializar modelos y escaladores globales
P_tot_model = None
P_tot_scaler_input = None
P_tot_scaler_output = None
N_tot_model = None
N_tot_scaler_input = None
N_tot_scaler_output = None

def inicializar_modelos():
    global P_tot_model, P_tot_scaler_input, P_tot_scaler_output
    global N_tot_model, N_tot_scaler_input, N_tot_scaler_output

    # Descargar y extraer si es necesario
    descargar_y_extraer_zip()

    # Cargar los archivos .pkl desde model/
    P_tot_model = joblib.load(os.path.join(MODEL_DIR, "random_forest1.pkl"))
    P_tot_scaler_input = joblib.load(os.path.join(MODEL_DIR, "min_max_scaler1.pkl"))
    P_tot_scaler_output = joblib.load(os.path.join(MODEL_DIR, "y_scaler1.pkl"))

    N_tot_model = joblib.load(os.path.join(MODEL_DIR, "random_forest2.pkl"))
    N_tot_scaler_input = joblib.load(os.path.join(MODEL_DIR, "min_max_scaler2.pkl"))
    N_tot_scaler_output = joblib.load(os.path.join(MODEL_DIR, "y_scaler2.pkl"))

# Ejecuta al importar el módulo para cargar modelos
inicializar_modelos()

def predecir_P_tot(data_dict):
    entrada = np.array([list(data_dict.values())])
    entrada_scaled = P_tot_scaler_input.transform(entrada)
    pred_scaled = P_tot_model.predict(entrada_scaled)
    pred_final = P_tot_scaler_output.inverse_transform(pred_scaled.reshape(-1, 1))
    return float(pred_final[0][0])

def predecir_N_tot(data_dict):
    entrada = np.array([list(data_dict.values())])
    entrada_scaled = N_tot_scaler_input.transform(entrada)
    pred_scaled = N_tot_model.predict(entrada_scaled)
    pred_final = N_tot_scaler_output.inverse_transform(pred_scaled.reshape(-1, 1))
    return float(pred_final[0][0])

