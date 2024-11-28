import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision import models
from torch.utils.data import DataLoader

batch_size = 64

train_dataset_path = './dataset/train'
test_dataset_path = './dataset/test'

transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize(size=(224, 224))
])

train_dataset = ImageFolder(root=train_dataset_path, transform=transform)
test_dataset = ImageFolder(root=test_dataset_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = models.efficientnet_v2_s(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
model.load_state_dict(torch.load("model_effnet_weightedv3.pth"))
model.to(device)

weigths = Tensor([12,4,6,6])
weigths.to(device)

loss_fn = nn.CrossEntropyLoss(weigths)
loss_fn.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ##if batch % 10 == 0:
            ##loss, current = loss.item(), (batch + 1) * len(X)
            ##print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct, bad, bad_predicted_as_good = 0, 0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            bad += (y == 0).type(torch.float).sum().item()
            bad_predicted_as_good += torch.logical_and(pred.argmax(1) == 1, y == 0).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    bad_predicted_as_good /= bad
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n, Bad predicted as good: {100*bad_predicted_as_good:>0.2f}% of bad images")
    return bad_predicted_as_good, correct

if(device == "cuda"):
    epochs = 100
    best_metric = 999999999999999
    best_model_state = None
    best_epoch = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        metric, accuracy = test(test_loader, model, loss_fn)
        torch.save(model.state_dict(), f"model_effnet_acc{accuracy:>4f}_badgood{metric:>4f}.pth")

        """if metric < best_metric and accuracy > 0.69:
            print("new Best model!")
            best_metric = metric
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = t
        if t % 5 == 0:
            torch.save(best_model_state, "model_effnet_weightedv3.pth")"""

    print("Done!")
    print("Best epoch: " + str(t))
