import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import urllib
import mmcv
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor, train_segmentor
import os
import dinov2.dinov2.eval.segmentation_m2f.models.segmentors


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")
        image = np.array(image)
        mask = np.array(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask


transform_train = A.Compose([
    # A.Resize(256, 256),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # ToTensorV2()
])

transform_val = A.Compose([
    # ToTensorV2()
])

train_images_dir = "/home/kamyar/Documents/segmentation_augmentation/train/images"
train_masks_dir = "/home/kamyar/Documents/segmentation_augmentation/train/masks"
val_images_dir = "/home/kamyar/Documents/segmentation_augmentation/val/images"
val_masks_dir = "/home/kamyar/Documents/segmentation_augmentation/val/masks"

train_image_paths = [os.path.join(train_images_dir, img) for img in os.listdir(train_images_dir)]
train_mask_paths = [os.path.join(train_masks_dir, mask) for mask in os.listdir(train_masks_dir)]
val_image_paths = [os.path.join(val_images_dir, img) for img in os.listdir(val_images_dir)]
val_mask_paths = [os.path.join(val_masks_dir, mask) for mask in os.listdir(val_masks_dir)]

train_dataset = SegmentationDataset(train_image_paths, train_mask_paths, transform=transform_train)
val_dataset = SegmentationDataset(val_image_paths, val_mask_paths, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


# Load your config and checkpoint
DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()

# with open('/home/kamyar/PycharmProjects/droneSegmentation/lowAltitude_segmentation/Dinov2_mask2former/config.yaml', 'r') as file:
cfg_str = load_config_from_url(CONFIG_URL)
cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

cfg.seed = 42
cfg.device = 'cuda'
cfg.gpu_ids = [0]
cfg.work_dir = './dinov2_mask2former_workspace'
cfg.pretrained = CHECKPOINT_URL
cfg.num_classes = 25
cfg.panoptic_on = False
cfg.semantic_on = True
cfg.instance_on = False
cfg.load_from = CHECKPOINT_URL

model = init_segmentor(cfg)
load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
model.to(device)

train_segmentor(model, train_dataset, cfg)

exit()


def calculate_iou(pred, target):
    intersection = (pred & target).float().sum((1, 2))
    union = (pred | target).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        print(masks.shape)
        optimizer.zero_grad()
        outputs = model(images, img_metas=[], gt_semantic_seg=masks, gt_labels=None, gt_masks=None)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        print("done")

    model.eval()
    val_loss = 0.0
    val_miou = 0.0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_outputs = model(val_images, img_metas=[])
            val_loss += criterion(val_outputs, val_masks).item()
            pred_masks = torch.argmax(val_outputs, dim=1)
            val_miou += calculate_iou(pred_masks, val_masks)

    val_loss /= len(val_loader)
    val_miou /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}, Val Loss: {val_loss}, Val mIoU: {val_miou}")

# Save your fine-tuned model if needed
torch.save(model.state_dict(), "fine_tuned_model.pth")
