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

class PrintLayer(Layer):
    def call(self, inputs):
        tf.print("Shape of tensor:", tf.shape(inputs))
        return inputs

class BooleanMaskLayer(Layer):
    def call(self, inputs):
        output, mask = inputs
        return tf.boolean_mask(output, mask)

    def compute_output_shape(self, input_shape):
        output_shape, mask_shape = input_shape
        # Since we are masking, the output shape is unknown until runtime
        return (None, output_shape[-1])  # Returns the correct shape after masking

def build_gnn_model(input_dim, max_nodes):
    node_features_input = Input(shape=(max_nodes, input_dim), name="node_features")
    adj_input = Input(shape=(max_nodes, max_nodes), name="adjacency_matrix")
    prediction_mask = Input(shape=(max_nodes,), dtype=tf.bool, name="prediction_mask")

    logging.info(f"Node features input shape: {node_features_input.shape}")
    logging.info(f"Adjacency matrix input shape: {adj_input.shape}")

    x = GCNConv(256, activation='relu')([node_features_input, adj_input])
    x = Dropout(0.2)(x)
    x = GCNConv(128, activation='relu')([x, adj_input])
    x = Dropout(0.2)(x)
    x = GCNConv(64, activation='relu')([x, adj_input])
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    # Apply the custom boolean mask layer, now assuming the output is per node
    masked_output = BooleanMaskLayer()([output, prediction_mask])

    model = Model(inputs=[node_features_input, adj_input, prediction_mask], outputs=masked_output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Try different rates like 0.01, 0.0001
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', f1_score])

    return model

def main(json_output_dir, model_output_path, batch_size):
    file_list = [os.path.join(json_output_dir, file) for file in os.listdir(json_output_dir) if file.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_dataset = create_tf_dataset(train_files, batch_size, balance=True, is_tdg=True)
    val_dataset = create_tf_dataset(val_files, batch_size, balance=True, is_tdg=True)

    # Unpack the dataset correctly
    (sample_feature, sample_adj, prediction_mask), sample_labels = next(iter(train_dataset))
    input_dim = sample_feature.shape[-1] if sample_feature.shape[0] > 0 else 4  # Default to 4 if shape is invalid
    max_nodes = sample_feature.shape[1] if sample_feature.shape[0] > 0 else 1  # Default to 1 node if shape is invalid

    # Pass both input_dim and max_nodes to the model
    model = build_gnn_model(input_dim, max_nodes)
    
    # Define the ModelCheckpoint callback
    checkpoint = ModelCheckpoint(
        filepath=model_output_path,  # Path where the model will be saved
        monitor='val_loss',          # Metric to monitor
        save_best_only=True,         # Save only the best model
        mode='min',                  # Minimize the monitored metric (for loss)
        verbose=1                    # Print a message when the model is saved
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='min')

    history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint, early_stopping])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_typilus.py <JsonOutputDir> <ModelOutputPath>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    json_output_dir = sys.argv[1]
    model_output_path = sys.argv[2]  # Ensure this is a full path, e.g., 'models/best_model.keras'
    batch_size = 1 #int(sys.argv[3])

    main(json_output_dir, model_output_path, batch_size)
