import optuna_search
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torchvision import models
from torch.utils.data import DataLoader

import optuna
from optuna.trial import TrialState


def define_model(model_type):
    model = None
    if model_type == "EfficientNet":
        model = models.efficientnet_v2_s(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    elif model_type == "ResNet50":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 4)
    return model


# 1. Define an objective function to be maximized.
def objective(trial):
    batch_size = 64

    train_dataset_path = "./dataset/train"
    test_dataset_path = "./dataset/test"

    transform = v2.Compose([v2.ToTensor(), v2.Resize(size=(224, 224))])

    train_dataset = ImageFolder(root=train_dataset_path, transform=transform)
    test_dataset = ImageFolder(root=test_dataset_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    weigths = Tensor(
        [
            trial.suggest_float("bad_weight", 0.0, 10.0),
            trial.suggest_float("good_weight", 0.0, 10.0),
            7,
            7,
        ]
    )
    # weigths = Tensor([12, 6, 8, 8])
    weigths.to(device)

    loss_fn = nn.CrossEntropyLoss(weigths)
    loss_fn.to(device)

    model = define_model(
        trial.suggest_categorical("model_type", ["ResNet50", "EfficientNet"])
    ).to(device)
    model.to(device)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    epochs = 10
    best_metric = 999999999999999
    best_model_state = None
    best_epoch = 0
    accuracy = 0.0

    for t in range(epochs):
        print(f"Epoch {t}\n-------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        metric, accuracy = test(test_loader, model, loss_fn)

        # Not supported with multi objective
        # trial.report(accuracy, t)

        # if trial.should_prune():
        #    raise optuna.exceptions.TrialPruned()

    return accuracy, metric


# 3. Create a study object and optimize the objective function.


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
            bad_predicted_as_good += (
                torch.logical_and(pred.argmax(1) == 1, y == 0)
                .type(torch.float)
                .sum()
                .item()
            )
    test_loss /= num_batches
    correct /= size
    bad_predicted_as_good /= bad
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n, Bad predicted as good: {100 * bad_predicted_as_good:>0.2f}% of bad images"
    )
    return bad_predicted_as_good, correct


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

if device == "cuda":
    study_name = "optuna_search"
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage="sqlite:///{}.db".format(study_name),
    )
    print("Starting optimization...")

    study.optimize(objective, n_trials=99)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Done!")
