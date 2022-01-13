# Prototypical Prompt Verbalizer

The repository contains the code for Paper "Eliciting Knowledge from Pretrained Language Models for Prototypical Prompt Verbalizer"

## Introduction
This repo includes general prompt-tuning, fine-tuning, soft verbalizer(based on [OpenPrompt](https://github.com/thunlp/OpenPrompt)) and our prototypical prompt verbalizer. We offer three data processor for three different datasets: AGNews, DBPedia, Yahoo Answers. More details can be seen in our paper.

## Usage
Using `run.py` to train or test with different settings, i.e., fine-tuning, prompt-tuning, prototypical prompt verbalizer.

Using `soft_train.py` and `soft_test.py` to train or test with soft verbalizer (Based on [OpenPrompt](https://github.com/thunlp/OpenPrompt)).

Using `./pretrain/contrastive.py` to run pretraining process with sample contents. (We haven't provided an argumenat parser for this process, the exact parameters need to be changed in the code)

Using `./pretrain/wiki_model.get***content` to get sentences containing a specific word from Wikidata or AGNews or Yahoo Answers or DBPedia. (We haven't provided an argumenat parser for this process, the exact parameters need to be changed in the code)

Using `./utils/sample_utils` to sample training set for few-shot scenario.

This repo now only support roberta.

## Main APP
| Arguments                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| data-dir                 | The input data directory                                     |
| prompt-dir               | File for prompts                                             |
| prompt-index             | Decide whice prompt to be used                               |
| model_name_or_path       | Path to pretrained model or shortcut name of the model       |
| pretrain_path            | Pretrained model path for prototypical prompt verbalizer     |
| sample_num               | Number of training set                                       |
| task_name                | Determine using which dataprocessor                          |
| space_before_mask        | Whether there is a deterministic space before mask           |
| mask_must_lower          | Whether the mask word must be lowercase                      |
| output_dir               | The output directory                                         |
| max_seq_length           | The maximum total input sequence length                      |
| fine_tune                | Using finetunemodel                                          |
| prompt_tune              | Using prompttunemodel                                        |
| contrastive_tune         | Using contrastivetunemodel                                   |
| using_pretrain           | Whether to use pretrained prototypical prompt verbalizer     |
| do_train                 | Whether to run training                                      |
| do_eval                  | Whether to run eval                                          |
| zero_eval                | Whether to run eval in zero-shot scenario                    |
| evaluate_during_training | Whether to run evaluation during training at each logging step |
| evaluate_after_epoch     | Whether to run evaluation after every epoch                  |
| per_gpu_train_batch_size | Batch size per GPU/CPU for training                          |
| per_gpu_eval_batch_size  | Batch size per GPU/CPU for evaluation                        |
| learning_rate            | The initial learning rate for Adam                           |
| seed                     | Random seed for initialization                               |
| overwrite_output_dir     | Overwrite the content of the output directory                |
| overwrite_cache          | Overwrite the cached training and evaluation sets            |
