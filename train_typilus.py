import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import logging
import tensorflow as tf
from tdg_utils import f1_score, create_tf_dataset

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

def main(json_output_dir, model_output_path):
    file_list = [os.path.join(json_output_dir, f) for f in os.listdir(json_output_dir) if f.endswith('.json')]
    train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

    batch_size = 32

    train_dataset = create_tf_dataset(train_files, batch_size)
    val_dataset = create_tf_dataset(val_files, batch_size)

    sample_feature, _ = next(train_dataset.as_numpy_iterator())
    input_dim = len(sample_feature[0])
    model = build_model(input_dim)

    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_f1_score', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_f1_score', patience=5, mode='max', restore_best_weights=True)

    history = model.fit(train_dataset, epochs=50, validation_data=val_dataset, callbacks=[checkpoint, early_stopping])

    best_model = load_model('best_model.keras', custom_objects={'f1_score': f1_score})
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
