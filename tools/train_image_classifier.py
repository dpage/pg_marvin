import argparse
import ast
import math
from pathlib import Path
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from transformers import ViTImageProcessor, ViTForImageClassification


class ImageClassificationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
 
    def __call__(self, batch):
        encodings = self.feature_extractor([x[0] for x in batch], return_tensors='pt')
        encodings['labels'] = torch.tensor([x[1] for x in batch], dtype=torch.long)
        return encodings 
    

class Classifier(pl.LightningModule):

    def __init__(self, model, lr: float, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy(
            task='multiclass' if model.config.num_labels > 2 else 'binary',
            num_classes=model.config.num_labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    

def read_command_line():
    parser = argparse.ArgumentParser(
        description='Train an image classifier model.')
    parser.add_argument("--model-dir", "-m", required=True,
                        help="the directory path in which to store the model")
    parser.add_argument("--training-dir", "-t", required=True,
                        help="the directory containing training images")
    parser.add_argument("--base-model",
                        default="google/vit-base-patch16-224-in21k",
                    help="the model on which to base the new model. This can "
                         "be either a local path, or the name of a model from "
                         "the Hugging Face collection at "
                         "https://huggingface.co/models (default: "
                         "'google/vit-base-patch16-224-in21k')")
    parser.add_argument("--ignore-mismatched-sizes", action='store_true',
                        help="ignore mismatched model sizes")
    parser.add_argument("--learning-rate", default=2e-5, type=float,
                        help="the learning rate to use when training "
                             "(default: 2e-5)")
    parser.add_argument("--batch-size", default=8, type=int,
                        help="the batch size for the data loader "
                             "(default: 8)")
    parser.add_argument("--num-workers", default=0, type=int,
                        help="the number of worker processes to use "
                             "(default: 0)")
    parser.add_argument("--accelerator", default="auto",
                        help="one of 'cpu', 'gpu', 'tpu', 'ipu', or 'auto' "
                             "(default: 'auto')")
    parser.add_argument("--devices", default="auto",
                        help="the accelerator devices to use. Use an integer "
                             "to specify the number of devices, a Python "
                             "style list to specify a set of devices "
                             "(e.g. '[1, 3, 7]'), or 'auto' (default: 'auto')")
    parser.add_argument("--precision", default="16-mixed",
                        help="an integer or string value to specify the "
                             "precision to use for the trainer. See "
                             "https://lightning.ai/docs/pytorch/latest/common/precision.html "
                             "(default: '16-mixed')")
    parser.add_argument("--max-epochs", default=4, type=int,
                        help="the maximum number of training epochs to run "
                             "(default: 4)")
    
    args = parser.parse_args()

    return args


def train_model():
    args = read_command_line()

    # Mangle inputs to the state we need them in
    data_dir = Path(args.training_dir)

    try:
        devices = ast.literal_eval(args.devices)
    except (ValueError, SyntaxError):
        try:
            devices = int(args.devices)
        except ValueError:
            devices = args.devices

    try:
        precision = int(args.precision)
    except ValueError:
        precision = args.precision

    # Sort out the training data
    ds = ImageFolder(data_dir)
    indices = torch.randperm(len(ds)).tolist()
    n_val = math.floor(len(indices) * .15)
    train_ds = torch.utils.data.Subset(ds, indices[:-n_val])
    val_ds = torch.utils.data.Subset(ds, indices[-n_val:])

    # Label name mappings
    label2id = {}
    id2label = {}

    for i, class_name in enumerate(ds.classes):
        label2id[class_name] = str(i)
        id2label[str(i)] = class_name

    # Create the extractor and collator
    feature_extractor = ViTImageProcessor.from_pretrained(args.base_model)
    model = ViTForImageClassification.from_pretrained(
        args.base_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes
    )
    collator = ImageClassificationCollator(feature_extractor)

    # Create the data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collator, num_workers=args.num_workers)

    # Train!
    pl.seed_everything(42)
    classifier = Classifier(model, lr=args.learning_rate)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=devices, precision=precision, max_epochs=args.max_epochs)
    trainer.fit(classifier, train_loader, val_loader)

    val_batch = next(iter(val_loader))
    outputs = model(**val_batch)

    # Save the model
    feature_extractor.save_pretrained(args.model_dir)
    model.save_pretrained(args.model_dir)

    # Dump out some interesting info
    print('Preds: ', outputs.logits.softmax(1).argmax(1))
    print('Labels:', val_batch['labels'])


if __name__ == '__main__':
    train_model()