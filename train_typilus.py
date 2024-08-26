import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from spektral.layers import GCNConv, GlobalSumPool
import tensorflow as tf
import logging
from tdg_utils import load_tdg_data, f1_score, create_tf_dataset
from tensorflow.keras import Model

# Instantiate metrics once
precision_metric = tf.keras.metrics.Precision()
recall_metric = tf.keras.metrics.Recall()

def f1_score(y_true, y_pred, precision_metric, recall_metric):
    # Update the metrics with the current batch
    precision_metric.update_state(y_true, y_pred)
    recall_metric.update_state(y_true, y_pred)

    precision_value = precision_metric.result()
    recall_value = recall_metric.result()
    
    # Compute the F1 score
    return 2 * (precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon())

class PrintLayer(Layer):
    def call(self, inputs):
        tf.print("Shape of tensor:", tf.shape(inputs))
        return inputs

def build_gnn_model(input_dim, max_nodes):
    node_features_input = Input(shape=(max_nodes, input_dim), name="node_features")
    adj_input = Input(shape=(max_nodes, max_nodes), name="adjacency_matrix")

    logging.info(f"Node features input shape: {node_features_input.shape}")
    logging.info(f"Adjacency matrix input shape: {adj_input.shape}")

    x = GCNConv(128, activation='relu')([node_features_input, adj_input])
    x = Dropout(0.5)(x)
    x = GCNConv(64, activation='relu')([x, adj_input])
    x = Dropout(0.5)(x)
    x = GlobalSumPool()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[node_features_input, adj_input], outputs=output)
    
    # Pass metrics to the model compilation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[lambda y_true, y_pred: f1_score(y_true, y_pred, precision_metric, recall_metric)])

    return model

def main(json_output_dir, model_output_path, batch_size):
    file_list = [os.path.join(json_output_dir, file) for file in os.listdir(json_output_dir) if file.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_dataset = create_tf_dataset(train_files, batch_size, balance=True, is_tdg=True)
    val_dataset = create_tf_dataset(val_files, batch_size, balance=True, is_tdg=True)

    # Unpack the dataset correctly
    sample_feature, sample_labels, sample_node_ids, sample_adj = next(iter(train_dataset))
    input_dim = sample_feature.shape[-1] if sample_feature.shape[0] > 0 else 4  # Default to 4 if shape is invalid
    max_nodes = sample_feature.shape[1] if sample_feature.shape[0] > 0 else 1  # Default to 1 node if shape is invalid

    # Pass both input_dim and max_nodes to the model
    model = build_gnn_model(input_dim, max_nodes)
    
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
