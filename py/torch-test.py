import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# 1. Data Loading & Preprocessing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.FashionMNIST('data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. Model Definition
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),           # Use ReLU activation (better than sigmoid for deep networks)
            nn.Dropout(0.2),      # Add dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 3. Training and Evaluation
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx == 900:
            print(f"[EPOCH {epoch}] [LOSS {loss.item():.5f}] [ACCURACY {(output.argmax(1) == target).type(torch.float).sum().item()} out of {len(data)}]")


def evaluate():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            correct += (output.argmax(1) == target).type(torch.float).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"\n[EVALUATION] [LOSS {test_loss:.5f}] [ACCURACY {correct} out of {len(test_loader.dataset)}]\n")


# 4. Print Model Summary
print("Neural Network Summary:\t\t[f := Sigmoid]")
for i, layer in enumerate(model.layers):
    if isinstance(layer, nn.Linear):
        print(f"Layer {i+1}\t{layer.in_features} neurons")

start_time = time.time()

# 5. Training Loop
for epoch in range(1, 11):
    train(epoch)

# 6. Evaluation
evaluate()

end_time = time.time()
print(f"Time taken: {end_time - start_time:.0f} seconds")
