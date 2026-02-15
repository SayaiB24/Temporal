# preprocessing.py
import pandas as pd
import torch
from torch_geometric.data import Data

def build_patient_graph():
    # Load Kaggle data
    patients = pd.read_csv('data/PATIENTS.csv')
    diagnoses = pd.read_csv('data/DIAGNOSES_ICD.csv')
    
    # 1. Map ICD codes to unique IDs (Node Features)
    unique_codes = diagnoses['ICD9_CODE'].unique()
    code_map = {code: i for i, code in enumerate(unique_codes)}
    
    # 2. Create Edges (Patient-to-Diagnosis)
    edge_index = []
    for _, row in diagnoses.iterrows():
        p_id = row['SUBJECT_ID']
        code_id = code_map[row['ICD9_CODE']]
        edge_index.append([p_id, code_id])
        
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 3. Create Targets (Survival Time & Event)
    # Event=1 if deathtime exists, 0 if censored
    y_time = (patients['DEATHTIME'] - patients['DOB']).dt.days # Simplified
    y_event = patients['EXPIRE_FLAG']
    
    return Data(edge_index=edge_index, y_time=y_time, y_event=y_event)