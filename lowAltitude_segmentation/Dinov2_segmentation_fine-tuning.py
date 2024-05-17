"""
Creating the datasetDict and turning to huggingface dataset,
Then, fine-tuning that data on Dino-v2 model
"""

from transformers import (
    Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
)
import glob
import albumentations as A
import cv2
import numpy as np
import utils_mask2former
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch
import torch.nn.functional as F
from tqdm import tqdm

def load_model(num_classes=1):
    image_processor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=True
    )
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        'facebook/mask2former-swin-tiny-ade-semantic',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )
    return model, image_processor



ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
def get_images(root_path):
    train_images = glob.glob('/home/kamyar/Documents/segmentation_augmentation/train/images/*')
    train_images.sort()
    train_masks = glob.glob( '/home/kamyar/Documents/segmentation_augmentation/train/masks/*')
    train_masks.sort()
    valid_images = glob.glob('/home/kamyar/Documents/segmentation_augmentation/val/images/*')
    valid_images.sort()
    valid_masks = glob.glob('/home/kamyar/Documents/segmentation_augmentation/val/masks/*')
    valid_masks.sort()
    return train_images, train_masks, valid_images, valid_masks
def train_transforms():
    train_image_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.25),
        A.Rotate(limit=25),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ], is_check_shapes=False)
    return train_image_transform
def valid_transforms():
    valid_image_transform = A.Compose([
        A.Normalize(mean=ADE_MEAN, std=ADE_STD)
    ], is_check_shapes=False)
    return valid_image_transform


def collate_fn(batch, image_processor):
    inputs = list(zip(*batch))
    images = inputs[0]
    segmentation_maps = inputs[1]
    batch = image_processor(
        images,
        segmentation_maps=segmentation_maps,
        return_tensors='pt',
        do_resize=False,
        do_rescale=False,
        do_normalize=False
    )
    batch['orig_image'] = inputs[2]
    batch['orig_mask'] = inputs[3]
    return batch


class SegmentationDataset(Dataset):
    def __init__(
            self,
            image_paths,
            mask_paths,
            tfms,
            classes_to_train,
            all_classes,
            feature_extractor
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = utils_mask2former.set_class_values(
            self.all_classes, self.classes_to_train
        )
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8')
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')
        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        orig_image = image.copy()
        image = image.transpose(2, 0, 1)
        mask = transformed['mask']

        # Get 2D label mask.
        mask = utils_mask2former.get_label_mask(mask, self.class_values, self.label_colors_list)
        orig_mask = mask.copy()

        return image, mask, orig_image, orig_mask




def get_dataset(
    train_image_paths,
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    img_size,
    feature_extractor
):
    train_tfms = train_transforms()
    valid_tfms = valid_transforms()
    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    return train_dataset, valid_dataset
def get_data_loaders(train_dataset, valid_dataset, batch_size, processor):
    collate_func = partial(collate_fn, image_processor=processor)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_func
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_func
    )
    return train_data_loader, valid_data_loader


def train(
        model,
        train_dataloader,
        device,
        optimizer,
        classes_to_train,
        processor,
        metric
):
    print('Training')
    model.train()
    train_running_loss = 0.0
    prog_bar = tqdm(
        train_dataloader,
        total=len(train_dataloader),
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    counter = 0  # to keep track of batch counter
    num_classes = len(classes_to_train)
    for i, data in enumerate(prog_bar):
        counter += 1
        pixel_values = data['pixel_values'].to(device)
        mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
        class_labels = [class_label.to(device) for class_label in data['class_labels']]
        pixel_mask = data['pixel_mask'].to(device)
        optimizer.zero_grad()
        outputs = model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
            pixel_mask=pixel_mask
        )
        ##### BATCH-WISE LOSS #####
        loss = outputs.loss
        train_running_loss += loss.item()
        ###########################

        ##### BACKPROPAGATION AND PARAMETER UPDATION #####
        loss.backward()
        optimizer.step()
        ##################################################
        target_sizes = [(image.shape[0], image.shape[1]) for image in data['orig_image']]
        pred_maps = processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )
        metric.add_batch(references=data['orig_mask'], predictions=pred_maps)

    ##### PER EPOCH LOSS #####
    train_loss = train_running_loss / counter
    ##########################
    iou = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=True)['mean_iou']
    return train_loss, iou


def validate(
        model,
        valid_dataloader,
        device,
        classes_to_train,
        label_colors_list,
        epoch,
        save_dir,
        processor,
        metric
):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    num_classes = len(classes_to_train)
    with torch.no_grad():
        prog_bar = tqdm(
            valid_dataloader,
            total=(len(valid_dataloader)),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        counter = 0  # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            pixel_values = data['pixel_values'].to(device)
            mask_labels = [mask_label.to(device) for mask_label in data['mask_labels']]
            class_labels = [class_label.to(device) for class_label in data['class_labels']]
            pixel_mask = data['pixel_mask'].to(device)
            outputs = model(
                pixel_values=pixel_values,
                mask_labels=mask_labels,
                class_labels=class_labels,
                pixel_mask=pixel_mask
            )
            target_sizes = [(image.shape[0], image.shape[1]) for image in data['orig_image']]
            pred_maps = processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes
            )
            ##### BATCH-WISE LOSS #####
            loss = outputs.loss
            valid_running_loss += loss.item()
            ###########################
            metric.add_batch(references=data['orig_mask'], predictions=pred_maps)

    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################
    iou = metric.compute(num_labels=num_classes, ignore_index=255, reduce_labels=True)['mean_iou']
    return valid_loss, iou


import torch
import os
import argparse
import evaluate
# from custom_datasets import get_images, get_dataset, get_data_loaders
# from model import load_model
# from config import ALL_CLASSES, LABEL_COLORS_LIST
# from engine import train, validate
# from utils import save_model, SaveBestModel, save_plots, SaveBestModelIOU
from torch.optim.lr_scheduler import MultiStepLR
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
parser = argparse.ArgumentParser()

if __name__ == '__main__':
    # Create a directory with the model name for outputs.
    out_dir = os.path.join('outputs')
    out_dir_valid_preds = os.path.join(out_dir, 'valid_preds')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_valid_preds, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    file_path = 'lowAltitude_classification/label_to_id.txt'
    id2label = {}
    label2id = {}
    with open(file_path, 'r') as file:
        for line in file:
            label, _id = line.strip().split(':')
            id2label[int(_id)] = label
            label2id[label] = int(_id)
    ALL_CLASSES = label2id.keys()
    model, processor = load_model(num_classes=len(ALL_CLASSES))
    model = model.to(device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")


    lr = 0.0001
    image_size = 256

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path=''
    )
    train_dataset, valid_dataset = get_dataset(
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        ALL_CLASSES,
        img_size=image_size,
        feature_extractor=processor
    )
    # train_dataloader, valid_dataloader = get_data_loaders(
    #     train_dataset,
    #     valid_dataset,
    #     args.batch,
    #     processor
    # )
    # # Initialize `SaveBestModel` class.
    # save_best_model = SaveBestModel()
    # save_best_iou = SaveBestModelIOU()
    # # LR Scheduler.
    # scheduler = MultiStepLR(
    #     optimizer, milestones=args.scheduler_epochs, gamma=0.1, verbose=True
    # )
    # train_loss, train_miou = [], []
    # valid_loss, valid_miou = [], []
    #
    # metric = evaluate.load("mean_iou")
    # for epoch in range(args.epochs):
    #     print(f"EPOCH: {epoch + 1}")
    #     train_epoch_loss, train_epoch_miou = train(
    #         model,
    #         train_dataloader,
    #         device,
    #         optimizer,
    #         ALL_CLASSES,
    #         processor,
    #         metric
    #     )
    #     valid_epoch_loss, valid_epoch_miou = validate(
    #         model,
    #         valid_dataloader,
    #         device,
    #         ALL_CLASSES,
    #         LABEL_COLORS_LIST,
    #         epoch,
    #         save_dir=out_dir_valid_preds,
    #         processor=processor,
    #         metric=metric
    #     )
    #     train_loss.append(train_epoch_loss)
    #     # train_pix_acc.append(train_epoch_pixacc)
    #     train_miou.append(train_epoch_miou)
    #     valid_loss.append(valid_epoch_loss)
    #     # valid_pix_acc.append(valid_epoch_pixacc)
    #     valid_miou.append(valid_epoch_miou)
    #     save_best_model(
    #         valid_epoch_loss, epoch, model, out_dir, name='model_loss'
    #     )
    #     save_best_iou(
    #         valid_epoch_miou, epoch, model, out_dir, name='model_iou'
    #     )
    #     print(
    #         f"Train Epoch Loss: {train_epoch_loss:.4f},",
    #         f"Train Epoch mIOU: {train_epoch_miou:4f}"
    #     )
    #     print(
    #         f"Valid Epoch Loss: {valid_epoch_loss:.4f},",
    #         f"Valid Epoch mIOU: {valid_epoch_miou:4f}"
    #     )
    #     if args.scheduler:
    #         scheduler.step()
    #     print('-' * 50)
    # # Save the loss and accuracy plots.
    # save_plots(
    #     train_loss, valid_loss,
    #     train_miou, valid_miou,
    #     out_dir
    # )
    # # Save final model.
    # save_model(model, out_dir, name='final_model')
    # print('TRAINING COMPLETE')













# from datasets import Dataset, DatasetDict, Image
# import torch
# import os
# from torch.utils.data import DataLoader
# import numpy as np
# from transformers import Dinov2Model, Dinov2PreTrainedModel
# from transformers.modeling_outputs import SemanticSegmenterOutput
# from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ToTensor, Normalize, CenterCrop
# from transformers import AutoImageProcessor, AutoModelForImageClassification
# import evaluate
# from transformers import AutoImageProcessor, TFSegformerForSemanticSegmentation
# from transformers import AutoFeatureExtractor
# from transformers import SegformerForSemanticSegmentation
# from transformers import TrainingArguments
# from transformers import Trainer
# from transformers import SegformerFeatureExtractor
#
# """ Append all the images of train/val directories in lists"""
# folder_img_train = '/home/kamyar/Documents/segmentation_augmentation/train/images'
# folder_masks_train = '/home/kamyar/Documents/segmentation_augmentation/train/masks'
# folder_img_val = '/home/kamyar/Documents/segmentation_augmentation/val/images'
# folder_masks_val = '/home/kamyar/Documents/segmentation_augmentation/val/masks'
#
# image_extensions = ['.jpg', '.JPG']
# image_paths_train = []
# mask_paths_train = []
# image_paths_val = []
# mask_paths_val = []
#
# for file in os.listdir(folder_img_train):
#     if os.path.isfile(os.path.join(folder_img_train, file)):
#         _, extension = os.path.splitext(file)
#         if extension.lower() in image_extensions:
#             image_paths_train.append(os.path.join(folder_img_train, file))
#
# for file in os.listdir(folder_masks_train):
#     if os.path.isfile(os.path.join(folder_masks_train, file)):
#         _, extension = os.path.splitext(file)
#         if extension.lower() in image_extensions:
#             mask_paths_train.append(os.path.join(folder_masks_train, file))
#
# for file in os.listdir(folder_img_val):
#     if os.path.isfile(os.path.join(folder_img_val, file)):
#         _, extension = os.path.splitext(file)
#         if extension.lower() in image_extensions:
#             image_paths_val.append(os.path.join(folder_img_val, file))
#
# for file in os.listdir(folder_masks_val):
#     if os.path.isfile(os.path.join(folder_masks_val, file)):
#         _, extension = os.path.splitext(file)
#         if extension.lower() in image_extensions:
#             mask_paths_val.append(os.path.join(folder_masks_val, file))
#
# def create_dataset(image_paths, label_paths):
#     dataset = Dataset.from_dict({"pixel_values": sorted(image_paths),
#                                 "label": sorted(label_paths)})
#     dataset = dataset.cast_column("pixel_values", Image())
#     dataset = dataset.cast_column("label", Image())
#
#     return dataset
#
# # step 1: create Dataset objects
# train_dataset = create_dataset(image_paths_train, mask_paths_train)
# validation_dataset = create_dataset(image_paths_val, mask_paths_val)
#
# # step 2: create DatasetDict
# dataset = DatasetDict({
#     "train": train_dataset,
#     "validation": validation_dataset,
#   }
# )
#
#
# file_path = 'lowAltitude_classification/label_to_id.txt'
# id2label = {}
# label2id = {}
# with open(file_path, 'r') as file:
#     for line in file:
#         label, _id = line.strip().split(':')
#         id2label[int(_id)] = label
#         label2id[label] = int(_id)
#
# processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
# model = TFSegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
#
#
# jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
# def train_transforms(example_batch):
#     images = [jitter(ToTensor()(x)) for x in example_batch['pixel_values']]
#     labels = [(np.expand_dims(x, axis=-1)) for x in example_batch['label']]
#     print(labels[0].size)
#     inputs = processor(images, labels)
#     return inputs
#
# def val_transforms(example_batch):
#     images = [ToTensor()(x) for x in example_batch['pixel_values']]
#     labels = [ToTensor()(x) for x in example_batch['label']]
#     inputs = processor(images, labels)
#     return inputs
#
# # Set transforms
# dataset['train'].set_transform(train_transforms)
# dataset['validation'].set_transform(val_transforms)
#
#
# # model = SegformerForSemanticSegmentation.from_pretrained(
# #     model_checkpoint,
# #     num_labels=len(id2label),
# #     id2label=id2label,
# #     label2id=label2id,
# #     ignore_mismatched_sizes=True,
# # )
# train_loader = DataLoader(dataset['train'], batch_size=2, shuffle=True, num_workers=2)
# valid_loader = DataLoader(dataset['validation'], batch_size=1, shuffle=False, num_workers=2)
#
# exit()
# epochs = 50
# lr = 0.00006
# batch_size = 2
#
# # training_args = TrainingArguments(
# #     "nvidia/segformer-b0-finetuned-drone",
# #     learning_rate=lr,
# #     num_train_epochs=epochs,
# #     per_device_train_batch_size=batch_size,
# #     per_device_eval_batch_size=batch_size,
# #     save_total_limit=3,
# #     evaluation_strategy="steps",
# #     save_strategy="steps",
# #     save_steps=20,
# #     eval_steps=20,
# #     logging_steps=1,
# #     eval_accumulation_steps=5,
# #     load_best_model_at_end=True,
# #     push_to_hub=False,
# #
# # )
#
#
#
# def compute_metrics(eval_pred):
#     with torch.no_grad():
#         logits, labels = eval_pred
#         logits_tensor = torch.from_numpy(logits)
#         # scale the logits to the size of the label
#         logits_tensor = nn.functional.interpolate(
#             logits_tensor,
#             size=labels.shape[-2:],
#             mode="bilinear",
#             align_corners=False,
#         ).argmax(dim=1)
#
#         pred_labels = logits_tensor.detach().cpu().numpy()
#         # currently using _compute instead of compute
#         # see this issue for more info: https://github.com/huggingface/evaluate/pull/328#issuecomment-1286866576
#         metrics = metric._compute(
#             predictions=pred_labels,
#             references=labels,
#             num_labels=len(id2label),
#             ignore_index=0,
#             do_reduce_labels=feature_extractor.do_reduce_labels,
#         )
#
#         # add per category metrics as individual key-value pairs
#         per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
#         per_category_iou = metrics.pop("per_category_iou").tolist()
#
#         metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
#         metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
#
#         return metrics
#
#
#
#
#
#
#
#
#
#
#
#
# # class SegmentationDataset(Dataset):
# #   def __init__(self, dataset, transform):
# #     self.dataset = dataset
# #     self.transform = transform
# #
# #   def __len__(self):
# #     return len(self.dataset)
# #
# #   def __getitem__(self, idx):
# #     item = self.dataset[idx]
# #     original_image = np.array(item["image"])
# #     original_segmentation_map = np.array(item["label"])
# #
# #     transformed = self.transform(image=original_image, mask=original_segmentation_map)
# #     image, target = torch.tensor(transformed['image']), torch.LongTensor(transformed['mask'])
# #
# #     return image, target, original_image, original_segmentation_map
# #
# #
# # model_name = 'facebook/dinov2-base'
# # processor = AutoImageProcessor.from_pretrained(model_name)
# # mean = processor.image_mean
# # std = processor.image_std
# # interpolation = processor.resample
# #
# #
# # import albumentations as A
# # transform = A.Compose([
# #     A.Normalize(mean=mean, std=std),
# # ])
# #
# # train_dataset = SegmentationDataset(dataset["train"], transform=transform)
# # val_dataset = SegmentationDataset(dataset["validation"], transform=transform)
# #
# #
# # def collate_fn(inputs):
# #     batch = dict()
# #     batch["pixel_values"] = torch.stack([i[0] for i in inputs], dim=0)
# #     batch["labels"] = torch.stack([i[1] for i in inputs], dim=0)
# #     batch["original_images"] = [i[2] for i in inputs]
# #     batch["original_segmentation_maps"] = [i[3] for i in inputs]
# #
# #     return batch
# #
# #
# # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
# # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
# #
# #
# # class LinearClassifier(torch.nn.Module):
# #     def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
# #         super(LinearClassifier, self).__init__()
# #
# #         self.in_channels = in_channels
# #         self.width = tokenW
# #         self.height = tokenH
# #         self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))
# #
# #     def forward(self, embeddings):
# #         embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)
# #         embeddings = embeddings.permute(0, 3, 1, 2)
# #
# #         return self.classifier(embeddings)
# #
# #
# # class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
# #     def __init__(self, config):
# #         super().__init__(config)
# #
# #         self.dinov2 = Dinov2Model(config)
# #         self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)
# #
# #     def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
# #         # use frozen features
# #         outputs = self.dinov2(pixel_values,
# #                               output_hidden_states=output_hidden_states,
# #                               output_attentions=output_attentions)
# #         # get the patch embeddings - so we exclude the CLS token
# #         patch_embeddings = outputs.last_hidden_state[:, 1:, :]
# #
# #         # convert to logits and upsample to the size of the pixel values
# #         logits = self.classifier(patch_embeddings)
# #         logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear",
# #                                                  align_corners=False)
# #
# #         loss = None
# #         if labels is not None:
# #             # important: we're going to use 0 here as ignore index instead of the default -100
# #             # as we don't want the model to learn to predict background
# #             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
# #             loss = loss_fct(logits.squeeze(), labels.squeeze())
# #
# #         return SemanticSegmenterOutput(
# #             loss=loss,
# #             logits=logits,
# #             hidden_states=outputs.hidden_states,
# #             attentions=outputs.attentions,
# #         )
# #
# #
# # model = Dinov2ForSemanticSegmentation.from_pretrained("facebook/dinov2-base", id2label=id2label,
# #                                                       num_labels=len(id2label))
# #
# # for name, param in model.named_parameters():
# #   if name.startswith("dinov2"):
# #     param.requires_grad = False
# #
# #
# #
# # metric = evaluate.load("mean_iou")
# #
# #
# # from torch.optim import AdamW
# # from tqdm.auto import tqdm
# #
# # # training hyperparameters
# # # NOTE: I've just put some random ones here, not optimized at all
# # # feel free to experiment, see also DINOv2 paper
# # learning_rate = 5e-5
# # epochs = 10
# #
# # optimizer = AdamW(model.parameters(), lr=learning_rate)
# #
# # # put model on GPU (set runtime to GPU in Google Colab)
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model.to(device)
# #
# # # put model in training mode
# # model.train()
# #
# # for epoch in range(epochs):
# #   print("Epoch:", epoch)
# #   for idx, batch in enumerate(tqdm(train_dataloader)):
# #       pixel_values = batch["pixel_values"].to(device)
# #       labels = batch["labels"].to(device)
# #
# #       # forward pass
# #       outputs = model(pixel_values, labels=labels)
# #       loss = outputs.loss
# #
# #       loss.backward()
# #       optimizer.step()
# #
# #       # zero the parameter gradients
# #       optimizer.zero_grad()
# #
# #       # evaluate
# #       with torch.no_grad():
# #         predicted = outputs.logits.argmax(dim=1)
# #
# #         # note that the metric expects predictions + labels as numpy arrays
# #         metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
# #
# #       # let's print loss and metrics every 100 batches
# #       if idx % 100 == 0:
# #         metrics = metric.compute(num_labels=len(id2label),
# #                                 ignore_index=0,
# #                                 reduce_labels=False,
# #         )
# #
# #         print("Loss:", loss.item())
# #         print("Mean_iou:", metrics["mean_iou"])
# #         print("Mean accuracy:", metrics["mean_accuracy"])
# #
# #
