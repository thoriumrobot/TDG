import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TDGDataset(Dataset):
    def __init__(self, json_files):
        self.graphs = []
        self.labels = []
        for json_file in json_files:
            with open(json_file, 'r') as file:
                data = json.load(file)
                graph = nx.readwrite.json_graph.node_link_graph(data)
                self.graphs.append(graph)
                # Extract labels based on node attributes (e.g., presence of @Nullable)
                labels = {node: 1 if '@Nullable' in data['nodes'][node].get('attributes', '') else 0 for node in graph.nodes()}
                self.labels.append(labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

class TypilusModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TypilusModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_model(model, train_loader, criterion, optimizer, num_epochs=25):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.float()
            labels = labels.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.round(outputs)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    model.load_state_dict(best_model_wts)
    return model

def main(data_directory, output_model_file):
    json_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.json')]
    dataset = TDGDataset(json_files)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

    model = TypilusModel(input_dim=10, hidden_dim=20, output_dim=1)  # Example dimensions
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model = train_model(model, train_loader, criterion, optimizer, num_epochs=25)

    torch.save(best_model.state_dict(), output_model_file)
    print(f'Model saved as {output_model_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Typilus Model on TDG JSON Files")
    parser.add_argument('data_directory', type=str, help='Path to the directory containing TDG JSON files')
    parser.add_argument('output_model_file', type=str, help='Path to save the trained model')
    args = parser.parse_args()
    main(args.data_directory, args.output_model_file)

