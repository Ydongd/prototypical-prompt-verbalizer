import logging
from torch.utils.data._utils.collate import default_collate
import torch
from torch.utils.data import TensorDataset
import re
import os

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, text_a='', text_b='', label=None, is_single=False):
        self.guid = guid
        self.is_single = is_single
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
    
class InputFeature(object):
    def __init__(self, guid, input_ids, token_type_ids, attention_mask, prompt_masks, label_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.prompt_masks = prompt_masks
        self.label_ids = label_ids
        
    @staticmethod
    def collate_fct(batch):
        r'''
        This function is used to collate the input_features.
        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.
        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        '''
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        batch_lens = torch.sum(batch_tuple[2], dim=-1, keepdim=False)
        max_len = batch_lens.max().item()
        results = ()
        for item in batch_tuple:
            if item.dim() >= 2:
                results += (item[:, :max_len],)
            else:
                results += (item,)
        return results


def convert_examples_to_features(
    examples,
    max_seq_length,
    tokenizer,
    prompt_util,
    prompt_index,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    task_name=''
):
    features = []
    for (ex_index, example) in enumerate(examples):
        guid = example.guid
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        if example.label == -1:
            label_ids = None
        else:
            label_ids = example.label
        
        prompt = prompt_util.prompts[prompt_index]
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        # adding the prompt tokens to special tokens
        special_tokens_count += prompt.prompt_length
        # truncate tokens_b first, if task is dbpedia, truncate tokens_a first
        if task_name == 'dbpedia':
            if (len(tokens_a) + len(tokens_b)) > (max_seq_length - special_tokens_count):
                max_length = max_seq_length - special_tokens_count
                # truncate
                for_a_nums = max_length - len(tokens_b)
                if for_a_nums >= 0:
                    tokens_a = tokens_a[:for_a_nums]
                else:
                    tokens_a = []
                    tokens_b = tokens_b[:max_length]
        else:
            if (len(tokens_a) + len(tokens_b)) > (max_seq_length - special_tokens_count):
                max_length = max_seq_length - special_tokens_count
                # truncate
                for_b_nums = max_length - len(tokens_a)
                if for_b_nums >= 0:
                    tokens_b = tokens_b[:for_b_nums]
                else:
                    tokens_b = []
                    tokens_a = tokens_a[:max_length]
        
        tokens, prompt_masks = prompt.wrap_text(tokenizer, tokens_a, tokens_b)
        # text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
        # print(text)
        # exit()
        # text_a = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens_a))
        # text_b = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens_b))

        # prompt = prompt_util.wrap_text(text_a=text_a, text_b=text_b, prompt_index=prompt_index, is_single=example.is_single, mask_num=1)
        # print(prompt)
        # tokens = tokenizer.tokenize(prompt)

        # reset speical tokens count
        special_tokens_count = 3 if sep_token_extra else 2
        assert len(tokens) <= max_seq_length - special_tokens_count

        tokens += [sep_token]
        prompt_masks += [0]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            prompt_masks += [0]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            prompt_masks += [0]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            prompt_masks = [0] + prompt_masks
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            prompt_masks = ([0] * padding_length) + prompt_masks
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids += [pad_token] * padding_length
            prompt_masks += [0] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(prompt_masks) == max_seq_length

        features.append(
           InputFeature(guid=guid,
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        prompt_masks=prompt_masks,
                        label_ids=label_ids)
        )
    return features

def convert_examples_to_features4finetnue(
    examples,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    task_name=''
):
    features = []
    for (ex_index, example) in enumerate(examples):
        guid = example.guid
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        if example.label == -1:
            label_ids = None
        else:
            label_ids = example.label
        
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        # truncate tokens_b first, if task is dbpedia, truncate tokens_a first
        if task_name == 'dbpedia':
            if (len(tokens_a) + len(tokens_b)) > (max_seq_length - special_tokens_count):
                max_length = max_seq_length - special_tokens_count
                # truncate
                for_a_nums = max_length - len(tokens_b)
                if for_a_nums >= 0:
                    tokens_a = tokens_a[:for_a_nums]
                else:
                    tokens_a = []
                    tokens_b = tokens_b[:max_length]
        else:
            if (len(tokens_a) + len(tokens_b)) > (max_seq_length - special_tokens_count):
                max_length = max_seq_length - special_tokens_count
                # truncate
                for_b_nums = max_length - len(tokens_a)
                if for_b_nums >= 0:
                    tokens_b = tokens_b[:for_b_nums]
                else:
                    tokens_b = []
                    tokens_a = tokens_a[:max_length]
        
        if tokens_b:
            tokens_b[0] = 'Ġ' + tokens_b[0]
        tokens = tokens_a + tokens_b
        # text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens))
        # print(text)
        # exit()
        # text_a = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens_a))
        # text_b = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens_b))

        # prompt = prompt_util.wrap_text(text_a=text_a, text_b=text_b, prompt_index=prompt_index, is_single=example.is_single, mask_num=1)
        # print(prompt)
        # tokens = tokenizer.tokenize(prompt)

        # reset speical tokens count
        special_tokens_count = 3 if sep_token_extra else 2
        assert len(tokens) <= max_seq_length - special_tokens_count

        tokens += [sep_token]

        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
        
        prompt_masks = [0] * max_seq_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(prompt_masks) == max_seq_length

        features.append(
           InputFeature(guid=guid,
                        input_ids=input_ids,
                        token_type_ids=segment_ids,
                        attention_mask=input_mask,
                        prompt_masks=prompt_masks,
                        label_ids=label_ids)
        )
    return features

def load_examples(args, data_processor, prompt_util, prompt_index, tokenizer, split):
    logger.info("Loading and converting data from data_utils.py...")
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            split, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length), str(args.prompt_index)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = data_processor.get_examples(split=split)
        features = convert_examples_to_features(
            examples,
            args.max_seq_length,
            tokenizer,
            prompt_util,
            prompt_index,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            task_name=args.task_name
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_prompt_masks = torch.tensor([f.prompt_masks for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids, all_prompt_masks)
    return dataset

def load_examples_4_finetune(args, data_processor, prompt_util, prompt_index, tokenizer, split):
    logger.info("Loading and converting data from data_utils.py...")
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "fine_cached_{}_{}_{}".format(
            split, list(filter(None, args.model_name_or_path.split("/"))).pop(), str(args.max_seq_length)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = data_processor.get_examples(split=split)

        features = convert_examples_to_features4finetnue(
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            task_name=args.task_name
        )
        # features = []
        # for example in examples:
        #     guid = example.guid
        #     text_a = example.text_a
        #     text_b = example.text_b
        #     label_ids = example.label

        #     if args.task_name != 'dbpedia':
        #         inputs = tokenizer.encode_plus(text=text_a, text_pair=text_b, max_length=args.max_seq_length, padding='max_length', truncation='only_second')
        #     else:
        #         inputs = tokenizer.encode_plus(text=text_a, text_pair=text_b, max_length=args.max_seq_length, padding='max_length', truncation='only_first')

        #     input_ids = inputs['input_ids']
        #     attention_mask = inputs['attention_mask']
        #     token_type_ids = [0] * args.max_seq_length
        #     prompt_masks = [0] * args.max_seq_length
        #     assert len(input_ids) == args.max_seq_length
        #     assert len(attention_mask) == args.max_seq_length
        #     assert len(token_type_ids) == args.max_seq_length
        #     assert len(prompt_masks) == args.max_seq_length
        #     features.append(
        #     InputFeature(guid=guid,
        #                     input_ids=input_ids,
        #                     token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask,
        #                     prompt_masks=prompt_masks,
        #                     label_ids=label_ids)
        #     )
        
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_prompt_masks = torch.tensor([f.prompt_masks for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids, all_prompt_masks)
    return dataset

