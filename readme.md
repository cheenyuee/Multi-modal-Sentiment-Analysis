# Multi-modal Sentiment Analysis

This is the official repository of DaSE 2022 Contemporary-Artificial-Intelligence course  project *Multi-modal Sentiment Analysis*.

## Setup

This implemetation is based on Python3. To run the code, you need the following dependencies:

- torch==1.9.0+cu111
- torchvision==0.10.0+cu111
- transformers==4.19.2
- sklearn==0.0
- numpy==1.22.3
- pandas==1.4.2
- chardet==4.0.0
- Pillow==9.2.0

You can simply run 

```shell
pip install -r requirements.txt
```

## Repository structure

We select some important files for detailed description.

```python
|-- checkpoints/ # the directory to save checkpoints
|-- dataset/  # the datasets
    |-- data/ # the original data
    |-- train.txt # the original data with label
    |-- test_without_label.txt # the original test data without label
    |-- train.json # the processed train set
    |-- dev.json # the processed dev set
    |-- test.json # the processed test set
|-- model/  # the implemented model
    |-- Multimodal_model.py # the main part of this model
    |-- utils.py # provide some useful functions
    |-- module/ # the sub-module of this model
    	|-- Img_module.py # the iamge module
    	|-- Text_module.py # the text module
|-- main.py # the main code
|-- split_dataset.py # the code for data processing
|-- test_with_label.txt # the prediction result of test_without_label.txt
|-- requirements.txt # dependencies
|-- README.md
```

## Data Process

We process the original dataset for this task by following steps:

- Read the origin dataset and pre-process the data.
- Divide the Training set and Dev set.
- Save the processed datasets as the json-format files.

**Actually, We have finished this step. The Train set, Dev set and Test set are already in the dataset directory.** 

**You can directly skip this step, unless you want to re-process or re-divide the original dataset.** 

Data statistic about the processed dataset for Multi-modal Sentiment Analysis is shown in this Table. 

| Dataset | Negative | Neutral | Positive | Total |
| ------- | -------- | ------- | -------- | ----- |
| Train   | 1,074    | 384     | 2142     | 3600  |
| Dev     | 119      | 35      | 246      | 400   |
| Total   | 1,193    | 419     | 2388     | 4000  |

| Dataset | Negative | Neutral | Positive | Total |
| ------- | -------- | ------- | -------- | ----- |
| Test    | unknown  | unknown | unknown  | 511   |

If you want to re-process and re-divide the Train set and Dev set, you can simply run:

```shell
python split_dataset.py --train_file ./dataset/train.json --dev_file ./dataset/dev.json --test_file ./dataset/test.json --dev_size 0.1 --random_state 6
```

## Train and Evaluation

You can simply try to train our model by the script (using default arguments):

```shell
python main.py --do_train
```
You can train our model on processed dataset by the script (using your designated  arguments):

```shell
python main.py --do_train \
--train_file ./dataset/train.json \
--dev_file ./dataset/dev.json \
--checkpoints_dir ./checkpoints \
--pretrained_model roberta-base \
--img_size 384 \
--text_size 64 \ 
--lr 1e-5 \
--dropout 0.0 \
--epoch 10 \
--batch_size 4        
```

## Test and Prediction

 You can simply try to test our model by the script, if you have simply tried in Train stage (using default arguments):

```shell
python main.py --do_test
```

You can test our model on processed dataset by the script (using your designated  arguments):

```shell
python main.py --do_test \
--test_output_file ./test_with_label.txt \
--dev_file ./dataset/dev.json \
--test_file ./dataset/test.json \
--checkpoints_dir ./checkpoints \
--batch_size 4 \
--img_size 384 \
--text_size 64
```

Please note that some arguments you have designated in Training and testing stages should be same, such as train_file, dev_file, checkpoints_dir....etc. Or else the file may can't be found due to path errors.

If you just want to try this model, you would better use the default arguments to avoid unexpected errors !

## Results

The results are shown in this Table(**Accuracy**):

| Fusion strategy           | Only Text | Only Image | Multi-modal | Ensemble |
| ------------------------- | --------- | ---------- | ----------- | -------- |
| Directly Concatenate      | 0.7000    | 0.6625     | 0.7300      | 0.7300   |
| Transformer Encoder       | 0.7350    | 0.6400     | 0.7300      | 0.7250   |
| Multi-head self-attention | 0.7100    | 0.6475     | 0.7475      | 0.7500   |

If you want to reproduce the experiment results, you can simply try (using the default arguments) and don't need modify any hyper-parameters.
