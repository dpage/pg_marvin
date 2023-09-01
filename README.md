# pg_marvin

pg_marvin is a PostgreSQL extension that provides machine learning and analytic functionality.

**WARNING**: This is proof of concept code! Do **NOT** deploy it into production.

## Installation

This extension uses PGXS for its build system. Currently there is no support
for VC++ on Windows. To build/install, ensure that `pg_config` is in the path,
and run `make install` to build the code and install it:

```bash
$ PATH=/path/to/postgresql/bin make install
```

On Windows, manually copy the *pg_marvin.control* and *pg_marvin--1.0.sql* files into the extension installation 
directory on the server, e.g. *\<PGINSTDIR\>\share\extension*

Create the extension in whatever database you want to use virtual environments:

```sql
pg_marvin=# CREATE EXTENSION pg_marvin CASCADE;
NOTICE:  installing required extension "plpy_venv"
NOTICE:  installing required extension "plpython3u"
CREATE EXTENSION
```

> **_NOTE:_** 
> pg_marvin is dependent on two additional extensions:
>
> plpython3u: Normally shipped with PostgreSQL<br>
> plpy_venv: Available from https://github.com/dpage/plpy_venv

Once installed, the Python environment needs to be setup. Note that this step may take a few minutes to complete:

```sql
pg_marvin=# SELECT marvin.create_venv();
NOTICE:  Collecting transformers[torch]
  Using cached transformers-4.32.1-py3-none-any.whl (7.5 MB)
Collecting filelock
  Using cached filelock-3.12.3-py3-none-any.whl (11 kB)
Collecting huggingface-hub<1.0,>=0.15.1
  Using cached huggingface_hub-0.16.4-py3-none-any.whl (268 kB)
Collecting numpy>=1.17
  Using cached numpy-1.25.2-cp311-cp311-macosx_11_0_arm64.whl (14.0 MB)
Collecting packaging>=20.0
  Using cached packaging-23.1-py3-none-any.whl (48 kB)
Collecting pyyaml>=5.1
  Using cached PyYAML-6.0.1-cp311-cp311-macosx_11_0_arm64.whl (167 kB)
Collecting regex!=2019.12.17
  Using cached regex-2023.8.8-cp311-cp311-macosx_11_0_arm64.whl (289 kB)
Collecting requests
  Using cached requests-2.31.0-py3-none-any.whl (62 kB)
Collecting tokenizers!=0.11.3,<0.14,>=0.11.1
  Using cached tokenizers-0.13.3-cp311-cp311-macosx_12_0_arm64.whl (3.9 MB)
Collecting safetensors>=0.3.1
  Using cached safetensors-0.3.3-cp311-cp311-macosx_13_0_arm64.whl (406 kB)
Collecting tqdm>=4.27
  Using cached tqdm-4.66.1-py3-none-any.whl (78 kB)
Collecting torch!=1.12.0,>=1.9
  Using cached torch-2.0.1-cp311-none-macosx_11_0_arm64.whl (55.8 MB)
Collecting accelerate>=0.20.3
  Using cached accelerate-0.22.0-py3-none-any.whl (251 kB)
Collecting psutil
  Using cached psutil-5.9.5-cp38-abi3-macosx_11_0_arm64.whl (246 kB)
Collecting fsspec
  Using cached fsspec-2023.6.0-py3-none-any.whl (163 kB)
Collecting typing-extensions>=3.7.4.3
  Using cached typing_extensions-4.7.1-py3-none-any.whl (33 kB)
Collecting sympy
  Using cached sympy-1.12-py3-none-any.whl (5.7 MB)
Collecting networkx
  Using cached networkx-3.1-py3-none-any.whl (2.1 MB)
Collecting jinja2
  Using cached Jinja2-3.1.2-py3-none-any.whl (133 kB)
Collecting charset-normalizer<4,>=2
  Using cached charset_normalizer-3.2.0-cp311-cp311-macosx_11_0_arm64.whl (122 kB)
Collecting idna<4,>=2.5
  Using cached idna-3.4-py3-none-any.whl (61 kB)
Collecting urllib3<3,>=1.21.1
  Using cached urllib3-2.0.4-py3-none-any.whl (123 kB)
Collecting certifi>=2017.4.17
  Using cached certifi-2023.7.22-py3-none-any.whl (158 kB)
Collecting MarkupSafe>=2.0
  Using cached MarkupSafe-2.1.3-cp311-cp311-macosx_10_9_universal2.whl (17 kB)
Collecting mpmath>=0.19
  Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
Installing collected packages: tokenizers, safetensors, mpmath, urllib3, typing-extensions, tqdm, sympy, regex, pyyaml, psutil, packaging, numpy, networkx, MarkupSafe, idna, fsspec, filelock, charset-normalizer, certifi, requests, jinja2, torch, huggingface-hub, transformers, accelerate
Successfully installed MarkupSafe-2.1.3 accelerate-0.22.0 certifi-2023.7.22 charset-normalizer-3.2.0 filelock-3.12.3 fsspec-2023.6.0 huggingface-hub-0.16.4 idna-3.4 jinja2-3.1.2 mpmath-1.3.0 networkx-3.1 numpy-1.25.2 packaging-23.1 psutil-5.9.5 pyyaml-6.0.1 regex-2023.8.8 requests-2.31.0 safetensors-0.3.3 sympy-1.12 tokenizers-0.13.3 torch-2.0.1 tqdm-4.66.1 transformers-4.32.1 typing-extensions-4.7.1 urllib3-2.0.4

 create_venv 
-------------
 t
(1 row)
```


## Usage

pg_marvin includes various machine learning functions. These are divided into related groups in the following sections.

> **_NOTE:_** 
>
> Many of the following functions will download and cache a pre-trained model upon first use. This means they may take
> additional time to run the first time they are executed.

### Low Level Functions

These functions provide low-level access to Hugging Face Transformers through an SQL interface. See
https://huggingface.co/docs/transformers/pipeline_tutorial for more information.

```sql
pg_marvin=# SELECT marvin.run_pipeline('{"I am very happy", "I am very sad"}'::text[], 'sentiment-analysis', model => 'bhadresh-savani/distilbert-base-uncased-emotion');
                                            run_pipeline                                            
----------------------------------------------------------------------------------------------------
 [{"label": "joy", "score": 0.9986562728881836}, {"label": "sadness", "score": 0.9982013702392578}]
(1 row)
```

The parameters accepted are:

* A *text[]* array of input data to analyse.
* The task to run. See https://huggingface.co/docs/transformers/v4.32.1/en/main_classes/pipelines#transformers.pipeline.task
  for the available options. Note that only tasks for which there are corresponding high-level functions below in the
  following sections have been tested with pg_marvin.
* An optional model to use from https://huggingface.co/models.

The analysis results are returned as a single JSON document.

### Sentiment Analysis

Sentiment analysis is used to gauge whether a text string is positive or negative or attreibute an emotion to it. The 
following functions can be used as shown in the examples to analyse a single or multiple strings. They return values 
indicating the sentiment in the form of a label and a score.

```sql
pg_marvin=# SELECT * FROM marvin.analyse_sentiment('I am very happy today');
  label   |       score        
----------+--------------------
 POSITIVE | 0.9998797178268433
(1 row)

pg_marvin=# SELECT * FROM marvin.analyse_sentiment('My fingers hurt from all the typing today.');
  label   |       score       
----------+-------------------
 NEGATIVE | 0.999701201915741
(1 row)
```

You can also pass an array of strings, for example:

```sql
pg_marvin=# SELECT * FROM marvin.analyse_sentiment('{"I am very happy", "I am very sad"}'::text[]);
  label   |       score        
----------+--------------------
 POSITIVE | 0.9998795986175537
 NEGATIVE | 0.9994852542877197
(2 rows)
```

You can also specify an alternative model to use (from the Hugging Face collection at https://huggingface.co/models).
For example:

```sql
pg_marvin=# SELECT * FROM marvin.analyse_sentiment('I am very happy', model => 'cardiffnlp/twitter-roberta-base-sentiment-latest');
  label   |       score        
----------+--------------------
 positive | 0.9763537645339966
(1 row)

pg_marvin=# SELECT * FROM marvin.analyse_sentiment('{"I am very happy", "I am very sad"}'::text[], model => 'cardiffnlp/twitter-roberta-base-sentiment-latest'); 
  label   |       score        
----------+--------------------
 positive | 0.9763537645339966
 negative | 0.8382462859153748
(2 rows)
```

Note that different models may return different values for labels - as can be seen in the examples above, the default
model returns *POSITIVE* or *NEGATIVE*, whilst the Cardiff NLP model returns *positive* or *negative*.

Some models may offer more complex analysis, e.g:

```sql
pg_marvin=# SELECT * FROM marvin.analyse_sentiment('{"I am very happy", "I am very sad", "I am outraged!"}'::text[], model => 'bhadresh-savani/distilbert-base-uncased-emotion');
  label  |       score        
---------+--------------------
 joy     | 0.9986562728881836
 sadness | 0.9982013702392578
 anger   | 0.9968796968460083
(3 rows)
```

### Text Translation

These functions translate text from one spoken language to another. To translate a single string, from English to 
French:

```sql
pg_marvin=# SELECT * FROM marvin.translate_text('Hello, how are you?', 'en', 'fr');
         translation         
-----------------------------
 Bonjour, comment êtes-vous?
(1 row)
```

Or to translate an array of strings from English to German:

```sql
pg_marvin=# SELECT * FROM marvin.translate_text('{"Hello, how are you?", "I am very well, thank you."}'::text[], 'en', 'de');
          translation           
--------------------------------
 Hallo, wie sind Sie?
 Ich bin sehr gut, danke Ihnen.
(2 rows)
```

You can also pass the *model* parameter to use a specific model for the translation. In this case, the source and
target language specifiers will have no effect. 