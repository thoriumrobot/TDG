import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import accuracy_score
import json
import os

class TypilusModel(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(TypilusModel, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

def load_graph_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    node_features = []
    edge_index = [[], []]
    labels = []
    node_mapping = {}

    # Create node features and labels
    for idx, (node, attrs) in enumerate(data.items()):
        node_mapping[node] = idx
        node_features.append(attrs['type'])  # Dummy feature; should be one-hot encoded or similar
        labels.append(attrs['type'])  # Dummy label; should be properly encoded

    # Create edge index
    for node, attrs in data.items():
        for neighbor in attrs['depends_on']:
            edge_index[0].append(node_mapping[node])
            edge_index[1].append(node_mapping[neighbor])

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=20):
    best_acc = 0.0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

        val_acc = evaluate_model(val_loader, model)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model.state_dict()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Acc: {val_acc}')

    return best_model

def evaluate_model(loader, model):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(data.y.cpu().numpy())

    acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    return acc

def main():
    train_data_dir = 'path/to/train/jsons'
    val_data_dir = 'path/to/val/jsons'

    train_files = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir) if f.endswith('.json')]
    val_files = [os.path.join(val_data_dir, f) for f in os.listdir(val_data_dir) if f.endswith('.json')]

    train_data = [load_graph_data(f) for f in train_files]
    val_data = [load_graph_data(f) for f in val_files]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    model = TypilusModel(num_node_features=1, num_classes=10)  # Adjust `num_node_features` and `num_classes` as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_model = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=20)

    torch.save(best_model, 'best_typilus_model.pth')
    print('Model saved as best_typilus_model.pth')

if __name__ == '__main__':
    main()

