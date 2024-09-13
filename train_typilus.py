import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from spektral.layers import GCNConv
import tensorflow as tf
import logging
from tdg_utils import create_tf_dataset, f1_score
from tensorflow.keras import Model

class BooleanMaskLayer(Layer):
    def call(self, inputs):
        output, mask = inputs
        return tf.boolean_mask(output, mask)

    def compute_output_shape(self, input_shape):
        output_shape, mask_shape = input_shape
        return (None, output_shape[-1])

def build_gnn_model(input_dim):
    node_features_input = Input(shape=(None, input_dim), name="node_features")
    adj_input = Input(shape=(None, None), name="adjacency_matrix")
    prediction_mask = Input(shape=(None,), dtype=tf.bool, name="prediction_mask")

    x = GCNConv(256, activation='relu')([node_features_input, adj_input])
    x = Dropout(0.2)(x)
    x = GCNConv(128, activation='relu')([x, adj_input])
    x = Dropout(0.2)(x)
    x = GCNConv(64, activation='relu')([x, adj_input])
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    masked_output = BooleanMaskLayer()([output, prediction_mask])

    model = Model(inputs=[node_features_input, adj_input, prediction_mask], outputs=masked_output)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', f1_score])

    return model

def main(json_output_dir, model_output_path, batch_size):
    file_list = [os.path.join(json_output_dir, file) for file in os.listdir(json_output_dir) if file.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    train_dataset = create_tf_dataset(train_files, batch_size, balance=True, is_tdg=True)
    val_dataset = create_tf_dataset(val_files, batch_size, balance=True, is_tdg=True)

    # Get input dimension from a sample batch
    for (sample_feature, sample_adj, prediction_mask), sample_labels in train_dataset.take(1):
        input_dim = sample_feature.shape[-1]

    model = build_gnn_model(input_dim)
    
    checkpoint = ModelCheckpoint(
        filepath=model_output_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=25, mode='min')

    model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint, early_stopping])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train_typilus.py <JsonOutputDir> <ModelOutputPath>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    json_output_dir = sys.argv[1]
    model_output_path = sys.argv[2]
    batch_size = 1  # Adjust as needed

    main(json_output_dir, model_output_path, batch_size)
