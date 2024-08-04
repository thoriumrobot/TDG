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
import random
import tensorflow as tf

def extract_features(attr):
    type_mapping = {'class': 0, 'method': 1, 'field': 2, 'parameter': 3, 'variable': 4, 'literal': 5}
    name_mapping = defaultdict(lambda: len(name_mapping))
    type_name_mapping = defaultdict(lambda: len(type_name_mapping))

    node_type = attr.get('type', '')
    node_name = attr.get('name', '')
    actual_type = attr.get('actual_type', '')
    nullable = float(attr.get('nullable', 0))

    type_id = type_mapping.get(node_type, len(type_mapping))
    name_id = name_mapping[node_name]
    type_name_id = type_name_mapping[actual_type]

    return [float(type_id), float(name_id), float(type_name_id), nullable]

def preprocess_tdg(tdg):
    features = []
    labels = []
    for node in tdg['nodes']:
        attr = node.get('attr', {})
        feature_vector = extract_features(attr)
        label = float(attr.get('nullable', 0))
        features.append(feature_vector)
        labels.append(label)
    logging.info(f"Extracted {len(features)} features and {len(labels)} labels from TDG")
    return np.array(features), np.array(labels)

def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_dataset, val_dataset, batch_size):
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')
    history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint], batch_size=batch_size)
    return history

def load_tdg_data(json_dir):
    data = []
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(json_dir, file_name), 'r') as f:
                data.append(json.load(f))
    logging.info(f"Loaded {len(data)} TDG files")
    return data

def balance_dataset(features, labels):
    pos_indices = [i for i, label in enumerate(labels) if label == 1]
    neg_indices = [i for i, label in enumerate(labels) if label == 0]
    
    random.shuffle(neg_indices)
    selected_neg_indices = neg_indices[:len(pos_indices)]
    
    selected_indices = pos_indices + selected_neg_indices
    random.shuffle(selected_indices)
    
    balanced_features = np.array([features[i] for i in selected_indices])
    balanced_labels = np.array([labels[i] for i in selected_indices])
    
    logging.info(f"Balanced dataset to {len(balanced_features)} features and {len(balanced_labels)} labels")
    return balanced_features, balanced_labels

def create_tf_dataset(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=len(features)).batch(batch_size)
    return dataset

def main(json_output_dir, model_output_path):
    tdg_data = load_tdg_data(json_output_dir)
    features, labels = [], []
    for tdg in tdg_data:
        f, l = preprocess_tdg(tdg)
        if len(f) > 0:  # Only include if features were extracted
            features.append(f)
            labels.append(l)

    if len(features) == 0 or len(labels) == 0:
        logging.error("No valid features or labels found. Exiting.")
        return

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    features, labels = balance_dataset(features, labels)

    batch_size = 32

    train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_dataset = create_tf_dataset(train_features, train_labels, batch_size)
    val_dataset = create_tf_dataset(val_features, val_labels, batch_size)

    input_dim = len(features[0])
    model = build_model(input_dim)
    train_model(model, train_dataset, val_dataset, batch_size)
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
