import json
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

def load_tdg_data(json_dir):
    data = []
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            with open(os.path.join(json_dir, file_name), 'r') as f:
                data.append(json.load(f))
    return data

def preprocess_data(tdg_data):
    features = []
    labels = []
    for tdg in tdg_data:
        for node in tdg['nodes']:
            feature_vector = extract_features(node)
            label = get_label(node)
            features.append(feature_vector)
            labels.append(label)
    return features, labels

def extract_features(node):
    return [node['attr'].get('type', 0), node['attr'].get('name', 0)]

def get_label(node):
    return node['attr'].get('nullable', 0)

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint])
    return history

def main(json_dir):
    tdg_data = load_tdg_data(json_dir)
    features, labels = preprocess_data(tdg_data)
    X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    input_dim = len(features[0])
    model = build_model(input_dim)
    train_model(model, X_train, y_train, X_val, y_val)
    best_model = load_model('best_model.h5')
    best_model.save('final_model.h5')
    print("Model training complete and saved as final_model.h5")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_typilus.py <JsonDirectory>")
        sys.exit(1)
    json_dir = sys.argv[1]
    main(json_dir)

