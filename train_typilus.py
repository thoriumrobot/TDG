import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from spektral.layers import GCNConv, GlobalSumPool
import tensorflow as tf
import logging
from tdg_utils import load_tdg_data, f1_score, create_tf_dataset

def build_gnn_model(input_dim):
    # Define the input layers
    node_features_input = Input(shape=(None, input_dim))  # Shape: (batch_size, num_nodes, num_features)
    adj_input = Input(shape=(None, None))  # Shape: (batch_size, num_nodes, num_nodes)

    # Apply GCN layers
    x = GCNConv(128, activation='relu')([node_features_input, adj_input])
    x = Dropout(0.5)(x)
    x = GCNConv(64, activation='relu')([x, adj_input])
    x = Dropout(0.5)(x)

    # Pooling to get a single vector for the entire graph
    x = GlobalSumPool()(x)

    # Dense layers for final prediction
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Define the model
    model = Model(inputs=[node_features_input, adj_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1_score])

    return model

def main(json_output_dir, model_output_path, batch_size):
    file_list = [os.path.join(json_output_dir, file) for file in os.listdir(json_output_dir) if file.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_dataset = create_tf_dataset(train_files, batch_size, balance=True, is_tdg=True)
    val_dataset = create_tf_dataset(val_files, batch_size, balance=False, is_tdg=True)

    # Check the first batch to get the input dimension
    sample_feature, _, _, _ = next(iter(train_dataset))
    input_dim = sample_feature.shape[-1]

    model = build_gnn_model(input_dim)
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
