import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop

def load_class_names(filepath):
    class_names = []
    with open(filepath, 'r') as file:
        for line in file:
            name, _ = line.split(':')
            class_names.append(name.strip())
    return class_names


class_names = load_class_names('lowAltitude_classification/label_to_id.txt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'facebook/dinov2-large-imagenet1k-1-layer'
test_dataset_path = "/home/kamyar/Documents/iNaturalist_split/test"
test_dataset = ImageFolder(root=test_dataset_path)

processor = AutoImageProcessor.from_pretrained(model_name)
mean = processor.image_mean
std = processor.image_std
interpolation = processor.resample

test_transform = Compose([
    CenterCrop(size=(256, 256)),
    ToTensor(),
    Normalize(mean=mean, std=std),
])

test_dataset.transform = test_transform
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=16)

model = AutoModelForImageClassification.from_pretrained(model_name, ignore_mismatched_sizes=True)
model = model.to(device)
num_classes = 30
model.classifier = nn.Linear(2048, num_classes).to(device)
model.load_state_dict(torch.load('lowAltitude_classification/best_classification_weights.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    correct_predictions_test = 0
    total_samples_test = 0

    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        logits = output.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        correct_predictions_test += (preds == y_batch).sum().item()
        total_samples_test += y_batch.size(0)

    test_accuracy = correct_predictions_test / total_samples_test
    print(f'Test accuracy: {test_accuracy:.4f}')

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay.from_predictions(all_preds, all_labels, display_labels=class_names, normalize='true', cmap="Blues", xticks_rotation='vertical')
disp.ax_.set_xticks(rotation="vertical")
plt.show()

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
# plt.xticks(rotation='vertical')
# plt.xticks(range(len(class_names)), [label[:10] + '...' if len(label) > 10 else label for label in class_names])
# disp.plot(cmap=plt.cm.Blues)
# disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
# plt.show()
