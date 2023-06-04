# Error-Free Data Collection

> Detail README will be released later

## Introduction

This is a simple, error-free data collector made with Pynecone. We aim to provide an application to remove errors during the data collection stage.

## Setup

### 1. Modify the model path

* Change line **25** in `ImageBind/data.py` to:

```python
10| BPE_PATH = "ImageBind/bpe/bpe_simple_vocab_16e6.txt.gz"
```

### 2. Download pretrained model and dependencies

```bash
$ wget https://github.com/Justin900429/error-free-data-collection/releases/download/v1.0.0/model.pt -p web/model.pt
$ pip install -r requirements.txt
```

### 3. Modify the deploy IP address

Copy `.env.template` to `.env` and modify the `IP_ADDRESS` to your server IP address.
