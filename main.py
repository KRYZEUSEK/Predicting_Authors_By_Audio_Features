import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, ConfusionMatrix

from Preprocessing import Train, test


class ModelDataset(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.data = dataframe.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        labels = self.data[idx, -1]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


train_data = ModelDataset(Train)
test_data = ModelDataset(test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(10, 64)
        self.activ1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 128)
        self.activ2 = nn.ReLU()
        self.layer3 = nn.Linear(128, 64)
        self.activ3 = nn.ReLU()
        self.output_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activ1(x)
        x = self.layer2(x)
        x = self.activ2(x)
        x = self.layer3(x)
        x = self.activ3(x)
        x = self.output_layer(x)
        return x


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True
)

acc = Accuracy("multiclass", num_classes=10)
confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=10)

for epoch in range(30):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    running_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features.float())
            loss = criterion(outputs, labels.long())
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            acc.update(predicted, labels.long())
    
    accuracy = acc.compute()
    val_loss = val_loss / len(test_loader.dataset)
    scheduler.step(val_loss)
    early_stopping = EarlyStopping(patience=5)
    early_stopping.step(val_loss)
    if early_stopping.should_stop:
        print(f"Early stopping at epoch {epoch+1}")
        break

    print(
        f"Epoch [{epoch+1}/30], Training Loss: {running_loss:.4f}, Validation Loss: {val_loss:.4f}"
    )
    print(f"Test Accuracy: {accuracy:.4f}")
