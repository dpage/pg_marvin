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

### load_qa_vectors.py

This script will connect to a PostgreSQL database, and load embedding vectors
from the specified document into a table. Vectors are generated using the
_flax-sentence-embeddings/all_datasets_v3_mpnet-base_ model from HuggingFace,
and are intended to be used for text generation when answering questions.

The pgvector PostgreSQL extension is used to provide the vector datatype and
operators and indexing methods (although it does not currently use an index).

This script is used in conjunction with _query_qa_vectors.py_ which will use
the resulting embeddings to answer questions.

The input document can be either a text file or a PDF. The script will 
do it's best to break down the text into paragraphs (in a quite naive way)
and will generate and store embeddings for all paragraphs that are over
20 tokens long.

The available options are follows:

```shell
python3 load_qa_vectors.py --help
usage: load_qa_vectors.py [-h] [-c] [-o HOST] [-p PORT] [-d DB] [-u USER] store file

Preprocess and load vectors of text into PostgreSQL.

positional arguments:
  store                 the name of the vector store to use. Equates to a table in PostgreSQL
  file                  the file to load text from

options:
  -h, --help            show this help message and exit
  -c, --clear           clear existing data from the store before loading
  -o HOST, --host HOST  the hostname or IP address of the PostgreSQL server (default: 127.0.0.1)
  -p PORT, --port PORT  the port number for the PostgreSQL server (default: 5432)
  -d DB, --db DB        the name of the database to use (default: postgres)
  -u USER, --user USER  the name of the database to use (default: postgres)
```

An example of the usage of the script is shown below:

```shell
python3 load_qa_vectors.py -d pg_marvin pgdocs /Users/dpage/postgres.txt
Generating embeddings: 100%|██████████████████████████████████████████████████████████████| 58/58 [05:52<00:00,  6.07s/it]
Loading data         : 100%|██████████████████████████████████████████████████████| 14626/14626 [00:05<00:00, 2615.17it/s]
```

### query_qa_vectors.py

This script is used to answer a question, with the text being generated from
the embedding vectors stored by _load_qa_vetors.py_. 

To operate, it will use the _flax-sentence-embeddings/all_datasets_v3_mpnet-base_
to generate an embedding vector for the question being asked. That vector is 
then used to query the PostgreSQL database to find the most relevant paragraphs
of text from the original material.

This data is then passed into the _vblagoje/bart_lfqa_ model which has been
trained to answer questions given the input question and contextual text.
For reasons that have not been explored in depth, this model seems to always
generate text up to _max_length_ tokens, and will cut off the output 
mid-sentence or descend into gibberish or repetition if the length is set too
high.

To overcome this, the output can optionally be passed to HuggingFace's 
_summarization_ pipeline using the _sshleifer/distilbart-cnn-12-6_ model
to produce more useful output.

The available options are follows:

```shell
python3 query_qa_vectors.py --help
usage: query_qa_vectors.py [-h] [-m MAX] [-s] [-t TOP_K] [-o HOST] [-p PORT] [-d DB] [-u USER] store question

Ask a question to be answered from vectors of text stored in PostgreSQL.

positional arguments:
  store                 the name of the vector store to use. Equates to a table in PostgreSQL
  question              the question to be answered

options:
  -h, --help            show this help message and exit
  -m MAX, --max MAX     maximum length of the output in tokens (default: 100)
  -s, --summary         include summarised output
  -t TOP_K, --top_k TOP_K
                        number of top results to consider
  -o HOST, --host HOST  the hostname or IP address of the PostgreSQL server (default: 127.0.0.1)
  -p PORT, --port PORT  the port number for the PostgreSQL server (default: 5432)
  -d DB, --db DB        the name of the database to use (default: postgres)
  -u USER, --user USER  the name of the database to use (default: postgres)```
```

An example of the usage of the script is shown below:

```shell
python3 ./query_qa_vectors.py --summary -d pg_marvin pgdocs "What is PostgreSQL?"
== Result =====================================================================
Question: 
    What is PostgreSQL?

Answer: 
    PostgreSQL is a relational database management system (RDBMS). That means
    it is a system for managing data stored in relations. Relation is a
    mathematical term for describing how data is organized in a database. For
    example, if you want to store data in a table, you can store it in a row,
    a column, or a column. You can also store the data in rows, columns, or
    columns. A relational database is just a way of organizing data in
    relation to other data. For instance, if I want to record the location of
    a person, I can store that information in a column in PostgreSQL. If I
    wanted to store a person's name, I could store that data in the column
    "John Doe". If I needed to store the name of a car, I would store that in
    the columns "John" and "Car". PostgreSQL is an ORDBMS, which means that it
    is an object-oriented database. This means that instead of storing

Summary: 
     Postgres is a relational database management system (RDBMS) That means it
    is a system for managing data stored in relations . Relation is a
    mathematical term for describing how data is organized in a database . For
    example, if you want to store data in a table, you can store it in a row,
    a column, or a column .
===============================================================================
```

It should be noted of course, that whilst not entirely wrong, the answer given
here is far from perfect. The models in use have not had any fine-tuning on
database or PostgreSQL related topics, and the vector embeddings that have been
loaded into the database consist only of single copy of the PostgreSQL 
documentation - hastily pre-processed to demonstrate the technology. It would 
certainly be possible to improve the quality of the output, however this work
is intended to show how PostgreSQL can be used in such systems, not to re-create
ChatGPT!

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
