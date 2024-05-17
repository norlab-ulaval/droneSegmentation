from transformers import (
    Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
)

def load_model(num_classes=1):
    image_processor = Mask2FormerImageProcessor(
        ignore_index=255, reduce_labels=False
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        'facebook/mask2former-swin-large-ade-semantic',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    return model, image_processor