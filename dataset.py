# dataset.py (Nội dung mới)

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def load_and_engineer(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)
    data = df.copy()
    data['dt'] = pd.to_datetime(data['date'] + ' ' + data['time'], errors='coerce', dayfirst=True)
    data = data.dropna(subset=['dt']).sort_values('dt').reset_index(drop=True)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = phi2 - phi1
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2.0)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2.0)**2
        return 2*R*np.arcsin(np.sqrt(a))

    data['lat_prev'] = data['latitude'].shift(1)
    data['lon_prev'] = data['longitude'].shift(1)
    data['dt_prev']  = data['dt'].shift(1)
    
    dist = haversine(data['latitude'].ffill(), data['longitude'].ffill(), data['lat_prev'].bfill(), data['lon_prev'].bfill())
    td = (data['dt'] - data['dt_prev']).dt.total_seconds().fillna(1).replace(0, 1)
    data['speed_mps'] = (dist / td).replace([np.inf, -np.inf], 0).fillna(0)
    data['hour'] = data['dt'].dt.hour
    data['weekday'] = data['dt'].dt.weekday

    feature_cols = ['latitude','longitude','speed_mps','hour','weekday']
    X = data[feature_cols].values
    y, classes = pd.factorize(data['type'])
    
    return X, y, classes.to_list(), feature_cols # Trả về feature_cols

def prepare_data(csv_path="IoT_GPS_Tracker.csv", num_clients=10, beta=0.5):
    # FIX: Nhận đủ 4 giá trị trả về từ hàm load_and_engineer
    X, y, classes, feature_cols = load_and_engineer(csv_path) 
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    
    sc = StandardScaler().fit(X_tr)
    X_tr_scaled = sc.transform(X_tr)
    X_te_scaled = sc.transform(X_te)

    n_classes = len(classes)
    label_distribution = np.random.dirichlet([beta] * num_clients, n_classes)
    class_indices = [np.where(y_tr == i)[0] for i in range(n_classes)]
    
    client_indices = [[] for _ in range(num_clients)]
    for c_idx in range(n_classes):
        indices_for_class_c = class_indices[c_idx]
        np.random.shuffle(indices_for_class_c)
        proportions = label_distribution[c_idx]
        splits = np.split(indices_for_class_c, (np.cumsum(proportions) * len(indices_for_class_c)).astype(int)[:-1])
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    trainloaders = []
    for indices in client_indices:
        if len(indices) == 0: continue
        dataset = TensorDataset(torch.tensor(X_tr_scaled[indices], dtype=torch.float32), torch.tensor(y_tr[indices], dtype=torch.long))
        trainloaders.append(DataLoader(dataset, batch_size=32, shuffle=True))

    valloader = DataLoader(TensorDataset(torch.tensor(X_te_scaled, dtype=torch.float32), torch.tensor(y_te, dtype=torch.long)), batch_size=64)
    trainloader_centralized = DataLoader(TensorDataset(torch.tensor(X_tr_scaled, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long)), batch_size=32, shuffle=True)
    
    # FIX: Trả về đúng 6 giá trị
    return trainloaders, valloader, trainloader_centralized, len(feature_cols), n_classes, classes