import json
import os
import dgl
import torch
from torch.utils.data import DataLoader, Dataset
from dgl.data.utils import save_graphs, load_graphs
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class JavaTDGDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.graphs = self.json_to_graphs(data)
    
    def json_to_graphs(self, data):
        graphs = []
        for graph_data in data:
            g = dgl.graph((graph_data['edges'], graph_data['nodes']))
            graphs.append(g)
        return graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

def load_dataset(directory):
    dataset = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('type_dependency_graph.json'):
                dataset.append(JavaTDGDataset(os.path.join(root, file)))
    return dataset

class TypilusModel(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(TypilusModel, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.classify = nn.Linear(h_feats, num_classes)
    
    def forward(self, g):
        h = g.ndata['feat']
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

def train(model, data_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch in data_loader:
            graphs, labels = batch
            outputs = model(graphs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            graphs, labels = batch
            outputs = model(graphs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

def main():
    directory = "path_to_java_projects"  # Replace with your path
    dataset = load_dataset(directory)
    
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    model = TypilusModel(in_feats=10, h_feats=20, num_classes=2)  # Modify features as needed
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train(model, train_loader, criterion, optimizer, num_epochs=10)
    accuracy = evaluate(model, test_loader)
    print(f'Accuracy: {accuracy}')
    
    torch.save(model.state_dict(), 'typilus_model.pth')
    print('Model saved as typilus_model.pth')

if __name__ == '__main__':
    main()
