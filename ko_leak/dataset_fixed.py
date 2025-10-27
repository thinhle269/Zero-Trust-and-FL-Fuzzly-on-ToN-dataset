# dataset_fixed.py
# Three-way split: D_train (70%), D_val (15%), D_test (15%)
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

    # Haversine distance to estimate speed
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

    dist = haversine(data['latitude'].ffill(), data['longitude'].ffill(),
                     data['lat_prev'].bfill(), data['lon_prev'].bfill())
    td = (data['dt'] - data['dt_prev']).dt.total_seconds().fillna(1).replace(0, 1)
    data['speed_mps'] = (dist / td).replace([np.inf, -np.inf], 0).fillna(0)
    data['hour'] = data['dt'].dt.hour
    data['weekday'] = data['dt'].dt.weekday

    feature_cols = ['latitude','longitude','speed_mps','hour','weekday']
    X = data[feature_cols].values
    y, classes = pd.factorize(data['type'])

    return X, y, classes.to_list(), feature_cols

def dirichlet_partition_indices(y_train, num_clients=10, beta=0.5, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    label_distribution = rng.dirichlet([beta] * num_clients, size=len(classes))
    class_indices = [np.where(y_train == c)[0] for c in classes]
    client_indices = [[] for _ in range(num_clients)]
    for ci, idxs in enumerate(class_indices):
        rng.shuffle(idxs)
        props = label_distribution[ci]
        cuts = (props * len(idxs)).astype(int)
        # adjust rounding
        while cuts.sum() < len(idxs):
            cuts[rng.randint(0, num_clients)] += 1
        s = 0
        for k in range(num_clients):
            client_indices[k].extend(idxs[s:s+cuts[k]]); s += cuts[k]
    return client_indices

def prepare_data(csv_path="IoT_GPS_Tracker.csv", num_clients=10, beta=0.5, seed=42):
    X, y, classes, feature_cols = load_and_engineer(csv_path)

    # 70/15/15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

    # scale on train only
    sc = StandardScaler().fit(X_train)
    X_train = sc.transform(X_train)
    X_val   = sc.transform(X_val)
    X_test  = sc.transform(X_test)

    # Non-IID partition on train
    idx_clients = dirichlet_partition_indices(y_train, num_clients=num_clients, beta=beta, seed=seed)
    # filter out empty clients
    non_empty = [idx for idx in idx_clients if len(idx) > 0]
    trainloaders = []
    for idx in non_empty:
        ds = TensorDataset(torch.tensor(X_train[idx], dtype=torch.float32),
                           torch.tensor(y_train[idx], dtype=torch.long))
        trainloaders.append(DataLoader(ds, batch_size=32, shuffle=True))

    # loaders
    valloader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                         torch.tensor(y_val, dtype=torch.long)), batch_size=64)
    testloader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                          torch.tensor(y_test, dtype=torch.long)), batch_size=64)
    trainloader_centralized = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                       torch.tensor(y_train, dtype=torch.long)), batch_size=32, shuffle=True)

    input_size = len(feature_cols)
    n_classes = len(classes)
    return trainloaders, valloader, testloader, trainloader_centralized, input_size, n_classes, classes
