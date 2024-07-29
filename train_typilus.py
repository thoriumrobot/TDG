import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint
import logging
from collections import defaultdict

def extract_features(attr):
    type_mapping = {'class': 0, 'method': 1, 'field': 2, 'parameter': 3, 'variable': 4}
    name_mapping = defaultdict(lambda: len(name_mapping))

    node_type = attr.get('type', '')
    node_name = attr.get('name', '')

    type_id = type_mapping.get(node_type, len(type_mapping))
    if node_name not in name_mapping:
        name_mapping[node_name] = len(name_mapping)
    name_id = name_mapping[node_name]

    return [float(type_id), float(name_id)]

def preprocess_tdg(tdg):
    features = []
    labels = []
    for node in tdg['nodes']:
        attr = node['attr']
        feature_vector = extract_features(attr)
        label = float(attr.get('nullable', 0))
        features.append(feature_vector)
        labels.append(label)
    return np.array(features), np.array(labels)

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

def load_tdg_data(json_dir):
    data = []
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(json_dir, file_name), 'r') as f:
                data.append(json.load(f))
    return data

def main(json_output_dir, model_output_path):
    tdg_data = load_tdg_data(json_output_dir)
    features, labels = [], []
    for tdg in tdg_data:
        f, l = preprocess_tdg(tdg)
        features.append(f)
        labels.append(l)

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    input_dim = len(features[0])
    model = build_model(input_dim)
    train_model(model, X_train, y_train, X_val, y_val)
    best_model = load_model('best_model.keras')
    best_model.save(model_output_path)
    logging.info(f"Model training complete and saved as {model_output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_typilus.py <JsonOutputDir> <ModelOutputPath>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO)
    json_output_dir = sys.argv[1]
    model_output_path = sys.argv[2]

    main(json_output_dir, model_output_path)
