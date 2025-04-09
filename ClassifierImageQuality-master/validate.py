import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision import models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd

valid_dataset_path = "./dataset/valid"

transform = v2.Compose([v2.ToTensor(), v2.Resize(size=(224, 224))])

valid_dataset = ImageFolder(root=valid_dataset_path, transform=transform)

batch_size = len(valid_dataset)

valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

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
model.load_state_dict(
    torch.load(
        "ClassifierImageQuality-master/model_effnet_acc0.770383_badgood0.146552.pth"
    )
)
model.to(device)

loss_fn = nn.CrossEntropyLoss()

classes = ["bad", "good", "too close", "too far"]


def valid(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            y = y.cpu()
            pred = pred.cpu()
            cf_matrix = confusion_matrix(y, pred.argmax(1))
            df_cm = pd.DataFrame(
                cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
                index=[i for i in classes],
                columns=[i for i in classes],
            )
            plt.figure(figsize=(12, 7))
            sn.heatmap(df_cm, annot=True)
            plt.xlabel("Predicted class")
            plt.ylabel("True class")
            plt.savefig("cmEffnetv4.png")

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if device == "cuda":
    valid(valid_loader, model, loss_fn)
