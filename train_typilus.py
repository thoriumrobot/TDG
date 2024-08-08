import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
from tdg_utils import load_tdg_data, f1_score, create_tf_dataset, NodeIDMapper, node_id_mapper

#node_id_mapper = NodeIDMapper()  # Initialize the node ID mapper

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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_score])
    return model

def main(json_output_dir, model_output_path, batch_size):
    file_list = [os.path.join(json_output_dir, file) for file in os.listdir(json_output_dir) if file.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_dataset = create_tf_dataset(train_files, batch_size, balance=True, is_tdg=True)
    val_dataset = create_tf_dataset(val_files, batch_size, balance=False, is_tdg=True)

    # Check the first batch to get the input dimension
    sample_feature, _, _ = next(iter(train_dataset))
    input_dim = sample_feature.shape[-1]

    model = build_model(input_dim)
    checkpoint = ModelCheckpoint(model_output_path, monitor='val_f1_score', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_f1_score', patience=10, mode='max')

    history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint, early_stopping])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python train_typilus.py <JsonOutputDir> <ModelOutputPath> <BatchSize>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    json_output_dir = sys.argv[1]
    model_output_path = sys.argv[2]
    batch_size = int(sys.argv[3])

    main(json_output_dir, model_output_path, batch_size)
