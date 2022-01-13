import argparse
import torch
import transformers
from transformers import BertConfig, BertTokenizer, BertModel, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,\
                         AlbertTokenizer, AlbertConfig, AlbertModel, \
                         BertForMaskedLM, RobertaForMaskedLM, AlbertForMaskedLM
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaLMHead
from transformers.models.albert.modeling_albert import AlbertMLMHead

_GLOBAL_ARGS = None

_MODEL_CLASSES = {
    'bert': {
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertModel,
        'masked_lm':BertForMaskedLM,
        'head':BertOnlyMLMHead
    },
    'roberta': {
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaModel,
        'masked_lm':RobertaForMaskedLM,
        'head':RobertaLMHead,
        'classification':RobertaForSequenceClassification
    },
    'albert': {
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model':AlbertModel,
        'masked_lm':AlbertForMaskedLM,
        'head':AlbertMLMHead
    }
}

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Prompt-based text classification.")
    
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data directory, should be specifed to specific task file."
    )

    parser.add_argument(
        "--prompt_dir",
        default=None,
        type=str,
        required=True,
        help="File for prompts."
    )

    parser.add_argument(
        "--prompt_index",
        default=None,
        type=int,
        required=True,
        help="Decide whice prompt to be used."
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Select the model type selected to be used from bert, roberta, albert, now only support roberta."
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or shortcut name of the model."
    )

    parser.add_argument("--text_num", default=2, type=int,
                    help="Number of text to be filled in prompt.")

    parser.add_argument("--pretrain_path", default=None, type=str, 
                        help="Pretrained model path for prototypical prompt verbalizer.")
    
    parser.add_argument("--sample_num", default=10, type=int,
                    help="Number of training set.")

    parser.add_argument("--space_before_mask", action="store_true", help="Whether there is a deterministic space before mask.")
    parser.add_argument("--mask_must_lower", action="store_true", help="Whether the mask word must be lowercase.")

    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="Determine using which dataprocessor.")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--fine_tune", action="store_true", help="Using finetunemodel.")
    parser.add_argument("--prompt_tune", action="store_true", help="Using prompttunemodel.")
    parser.add_argument("--contrastive_tune", action="store_true", help="Using contrastivetunemodel.")

    parser.add_argument("--using_pretrain", action="store_true", help="Whether to use pretrained prototypical prompt verbalizer.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval.")
    parser.add_argument("--zero_eval", action="store_true", help="Whether to run eval in zero-shot scenario.")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--evaluate_after_epoch",
        action="store_true",
        help="Whether to run evaluation after every epoch.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--keep_accents", action="store_const", const=True, help="Set this flag if model is trained with accents."
    )
    parser.add_argument(
        "--strip_accents", action="store_const", const=True, help="Set this flag if model is trained without accents."
    )
    parser.add_argument("--use_fast", action="store_const", const=True, help="Set this flag to use fast tokenization.")
    
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--bert_lr", default=3e-5, type=float, help="Learning rate for bert model.")
    parser.add_argument("--cls_lr", default=3e-5, type=float, help="Learning rate for head above bert model.")

    parser.add_argument("--adv_training", default=None, choices=['fgm', 'pgd'], help="Using adversarial training.")

    parser.add_argument("--warmup_proportion", default=0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=str, default='0.1', help="Log every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available.")

    parser.add_argument("--seed", type=int, default=8, help="Random seed for initialization.")

    parser.add_argument("--dropout_prob", type=float, default=0.1, help="Dropout rate.")

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory."
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets."
    )

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS

def get_model_classes():
    return _MODEL_CLASSES