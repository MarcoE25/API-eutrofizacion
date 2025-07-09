import joblib
import numpy as np

# Carga de modelos y scalers
P_tot_model = joblib.load("model/random_forest1.pkl")
P_tot_scaler_input = joblib.load("model/min_max_scaler1.pkl")
P_tot_scaler_output = joblib.load("model/y_scaler1.pkl")

N_tot_model = joblib.load("model/random_forest2.pkl")
N_tot_scaler_input = joblib.load("model/min_max_scaler2.pkl")
N_tot_scaler_output = joblib.load("model/y_scaler2.pkl")

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
