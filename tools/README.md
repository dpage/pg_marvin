# pg_marvin Tools

This directory contains various tools for working with pg_marvin - primarily for
training models and other tasks that don't make sense to run from within the
database server.

## Setup

Create a Python virtual environment in which to run these tools, and install 
the packages in [requirements.txt](requirements.txt) to be able to run them:

```sh
$ python3 -m venv marvin
$ source marvin/bin/activate
$ pip install -r requirements.txt
Collecting accelerate==0.22.0 (from -r requirements.txt (line 1))
...
...
```

## Usage

Activate the virtual environment before trying to run any of the tools:

```sh
$ source marvin/bin/activate
```

Once activated, you can run any of the tools using the Python interpreter
that is found in the PATH.

## Tools

### train_image_classifier.py

This script will train an image classifier model, based on images stored in
a training data directory, organised in sub-directories named for the label to
assign the contents. For example:

```
training_data/
  cat/
    cat1.jpg
    cat2.jpg
    cat3.jpg
  dog/
    dog1.jpg
    dog2.jpg
    dog3.jpg
  fish/
    fish1.jpg
    fish2.jpg
    fish3.jpg
```

The file names themselves can be anything; only the name of the directory they
are in matters.

The available options are as follows:

```bash
$ python3 train_image_classifier.py --help
usage: train_image_classifier.py [-h] --model-dir MODEL_DIR --training-dir TRAINING_DIR [--base-model BASE_MODEL] [--ignore-mismatched-sizes] [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--accelerator ACCELERATOR] [--devices DEVICES]
                                 [--precision PRECISION] [--max-epochs MAX_EPOCHS]

Train an image classifier model.

options:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR, -m MODEL_DIR
                        the directory path in which to store the model
  --training-dir TRAINING_DIR, -t TRAINING_DIR
                        the directory containing training images
  --base-model BASE_MODEL
                        the model on which to base the new model. This can be either a local path, or the name of a model from the Hugging Face collection at https://huggingface.co/models (default: 'google/vit-base-patch16-224-in21k')
  --ignore-mismatched-sizes
                        ignore mismatched model sizes
  --learning-rate LEARNING_RATE
                        the learning rate to use when training (default: 2e-5)
  --batch-size BATCH_SIZE
                        the batch size for the data loader (default: 8)
  --num-workers NUM_WORKERS
                        the number of worker processes to use (default: 0)
  --accelerator ACCELERATOR
                        one of 'cpu', 'gpu', 'tpu', 'ipu', or 'auto' (default: 'auto')
  --devices DEVICES     the accelerator devices to use. Use an integer to specify the number of devices, a Python style list to specify a set of devices (e.g. '[1, 3, 7]'), or 'auto' (default: 'auto')
  --precision PRECISION
                        an integer or string value to specify the precision to use for the trainer. See https://lightning.ai/docs/pytorch/latest/common/precision.html (default: '16-mixed')
  --max-epochs MAX_EPOCHS
                        the maximum number of training epochs to run (default: 4)
```

A typical training task might look as follows; in this case training a model
to b saved at *~/Public/uk-coarse-fish* using the images found at
*~/Downloads/uk-coarse-fish* for 50 epochs, based on the default model
*google/vit-base-patch16-224-in21k*, and using otherwise-default options:

```bash
$ python3 train_image_classifier.py -m ~/Public/uk-coarse-fish -t ~/Downloads/uk-coarse-fish --max-epochs 50
Some weights of ViTForImageClassification were not initialized from the model checkpoint at google/vit-base-patch16-224-in21k and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Global seed set to 42
Using 16bit Automatic Mixed Precision (AMP)
GPU available: True (mps), used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
HPU available: False, using: 0 HPUs

  | Name    | Type                      | Params
------------------------------------------------------
0 | model   | ViTForImageClassification | 85.8 M
1 | val_acc | MulticlassAccuracy        | 0     
------------------------------------------------------
85.8 M    Trainable params
0         Non-trainable params
85.8 M    Total params
343.222   Total estimated model params size (MB)
Epoch 49: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 271/271 [01:19<00:00,  3.42it/s, v_num=0, val_acc=0.887]`Trainer.fit` stopped: `max_epochs=50` reached.                                                                                                                                                                                     
Epoch 49: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 271/271 [01:20<00:00,  3.39it/s, v_num=0, val_acc=0.887]
Preds:  tensor([0, 2, 0, 4, 1, 8, 1, 4])
Labels: tensor([0, 3, 0, 4, 1, 8, 1, 4])
```