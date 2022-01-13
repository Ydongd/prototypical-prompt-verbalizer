import sys

sys.path.append("..")
import os
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
import torch.nn as nn
import random
import time
from torch.utils.data import TensorDataset
from utils.prompt_utils import Prompt
import math
from tqdm import tqdm, trange
from utils.utils_adversarial import FGM, PGD
import numpy as np
from transformers import AdamW, AutoConfig, get_linear_schedule_with_warmup
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
import argparse

class InputExample(object):
    def __init__(self, text_a='', text_b='', label=None, is_single=False):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
    
class InputFeature(object):
    def __init__(self, input_ids, token_type_ids, attention_mask, prompt_masks, label_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids

class Contrastive_model(torch.nn.Module):
    def __init__(self, labels, model_path, seed, sample_num=30, batch_size=8):
        super().__init__()
        self.labels = labels # labels is a 1-dimensional list， like ['company', 'book publication']
        self.label2idx = {k: i for (i, k) in enumerate(labels)}
        self.idx2label = {i: k for (i, k) in enumerate(labels)}
        self.num_labels = len(labels)

        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        # self.masked_model = RobertaForMaskedLM.from_pretrained(model_path)
        self.masked_model = RobertaModel.from_pretrained(model_path)
        self.mask_token = self.tokenizer.mask_token
        self.mask_idx = self.tokenizer.convert_tokens_to_ids(self.mask_token)

        prompt_text = '{} In this sentence, {} means <mask>.'
        self.prompt = Prompt(prompt_text, 2)
        self.prompt.set_attributes(self.tokenizer.tokenize(prompt_text))
        self.sample_num = 60
        self.batch_size = 8
        self.seed = seed
        self.embedding_size = 256
        self.max_seq_length = 512
        self.all_size = self.sample_num * self.num_labels
        self.batch_num = self.all_size // self.batch_size

        # self.classify_embedding = nn.Linear(self.embedding_size, self.num_labels)
        self.transform_matrix = nn.Linear(1024, self.embedding_size)
        self.label_embedding = nn.Embedding(self.num_labels, self.embedding_size)
        # zeros = torch.zeros([self.num_labels, self.embedding_size])
        # self.label_embedding.weight.data.copy_(zeros)
        # w = torch.empty(self.num_labels, self.embedding_size)
        # self.label_embedding.weight = nn.Parameter(nn.init.uniform_(w,-0.5/self.embedding_size,0.5/self.embedding_size))

    def forward(self, input_ids, attention_mask, labels):
        batch_size = input_ids.size(0)
        mask_index = input_ids.eq(self.mask_idx)
        mask_index = torch.nonzero(mask_index)
        seq = mask_index[:, 0]
        mask_index = mask_index[:, 1]
        
        output = self.masked_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output['last_hidden_state'] # [batch_size, max_seq_length, hidden_size]
        embedding = last_hidden_state[seq, mask_index] # [batch_size, hidden_size]
        loss1, loss2, loss3 = self.sample_proto_loss(labels, embedding)
        a1 = 0.5
        a2 = 0.5
        # print(loss1)
        # print(loss2)
        # print(loss3)
        # exit()
        loss = a1 * loss1 + a2 * (loss2 + loss3)
        return loss

    def sample_proto_loss(self, labels, embedding):
        # labels: [batch_size]
        # embedding: [batch_size, hidden_size]
        embedding = self.transform_matrix(embedding)
        label_index = {} # key:label value:indexes of label
        bs = len(labels)
        for i in range(bs):
            ll = labels[i].item()
            if ll not in label_index:
                label_index[ll] = []
            label_index[ll].append(i)
        
        # inter sample loss
        inter_sample_loss = 0.0
        for i in range(bs):
            ll = labels[i].item()
            positive_index = label_index[ll]
            negative_index = [k for k in range(bs) if k not in positive_index]
            if negative_index == []:
                break
            negative_embedding = torch.index_select(embedding, 0, torch.tensor(negative_index).cuda())
            for j in range(bs):
                if i == j:
                    continue
                pos_loss = torch.tensor(0.0)
                neg_loss = 0.0
                if labels[j].item() == ll:
                    pos_loss = torch.cosine_similarity(embedding[i].unsqueeze(0), embedding[j].unsqueeze(0))[0]
                neg_loss = torch.sum(torch.exp(torch.cosine_similarity(embedding[i].unsqueeze(0), negative_embedding)))
                inter_sample_loss = inter_sample_loss - pos_loss + torch.log(neg_loss + torch.exp(pos_loss))
        inter_sample_loss = inter_sample_loss / (bs * bs)

        sample_proto_loss_1 = 0.0
        sample_proto_loss_2 = 0.0
        # for i in range(bs):
        #     ll = labels[i].item()
        #     for k in range(self.num_labels):
        #         pos_loss = torch.tensor(0.0)
        #         neg_loss_1 = 0.0
        #         if k == ll:
        #             pos_loss = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight[ll].unsqueeze(0))[0]
        #         neg_index_1 = [j for j in range(self.num_labels) if j != ll]
        #         negative_embedding_1 = torch.index_select(self.label_embedding.weight, 0, torch.tensor(neg_index_1).cuda())
        #         neg_loss_1 = torch.sum(torch.exp(torch.cosine_similarity(embedding[i].unsqueeze(0), negative_embedding_1)))
        #         sample_proto_loss_1 = sample_proto_loss_1 - pos_loss + torch.log(neg_loss_1 + torch.exp(pos_loss))
        # for k in range(self.num_labels):
        #     for i in range(bs):
        #         ll = labels[i].item()
        #         pos_loss = torch.tensor(0.0)
        #         neg_loss_2 = 0.0
        #         if k == ll:
        #             pos_loss = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight[k].unsqueeze(0))[0]
        #         neg_index_2 = [j for j in range(bs) if labels[j].item()!=k]
        #         negative_embedding_2 = torch.index_select(embedding, 0, torch.tensor(neg_index_2).cuda())
        #         neg_loss_2 = torch.sum(torch.exp(torch.cosine_similarity(self.label_embedding.weight[k].unsqueeze(0), negative_embedding_2)))
        #         sample_proto_loss_2 = sample_proto_loss_2 - pos_loss + torch.log(neg_loss_2 + torch.exp(pos_loss))
        # sample_proto_loss_1 = sample_proto_loss_1 / (bs * self.num_labels)
        # sample_proto_loss_2 = sample_proto_loss_2 / (bs * self.num_labels)
                
        for i in range(bs):
            ll = labels[i].item()
            pos_loss = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight[ll].unsqueeze(0))[0]
            
            neg_index_1 = [j for j in range(self.num_labels) if j != ll]
            negative_embedding_1 = torch.index_select(self.label_embedding.weight, 0, torch.tensor(neg_index_1).cuda())
            neg_loss_1 = torch.sum(torch.exp(torch.cosine_similarity(embedding[i].unsqueeze(0), negative_embedding_1)))
            
            neg_index_2 = [j for j in range(bs) if labels[j].item()!=ll]
            if neg_index_2 == []:
                neg_loss_2 = torch.tensor(0.0)
            else:
                negative_embedding_2 = torch.index_select(embedding, 0, torch.tensor(neg_index_2).cuda())
                neg_loss_2 = torch.sum(torch.exp(torch.cosine_similarity(self.label_embedding.weight[ll].unsqueeze(0), negative_embedding_2)))

            sample_proto_loss_1 = sample_proto_loss_1 - pos_loss + torch.log(neg_loss_1 + torch.exp(pos_loss))
            sample_proto_loss_2 = sample_proto_loss_2 - pos_loss + torch.log(neg_loss_2 + torch.exp(pos_loss))
            # ll = labels[i].item()
            # pos_loss = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight[ll].unsqueeze(0))
            
            # neg_index_1 = [j for j in range(self.num_labels) if j != ll]
            # negative_embedding_1 = torch.index_select(self.label_embedding.weight, 0, torch.tensor(neg_index_1).cuda())
            # neg_loss_1 = torch.sum(torch.exp(torch.cosine_similarity(embedding[i].unsqueeze(0), negative_embedding_1)))
            
            # neg_index_2 = [j for j in range(bs) if labels[j].item()!=ll]
            # negative_embedding_2 = torch.index_select(embedding, 0, torch.tensor(neg_index_2).cuda())
            # neg_loss_2 = torch.sum(torch.exp(torch.cosine_similarity(self.label_embedding.weight[ll].unsqueeze(0), negative_embedding_2)))

            # sample_proto_loss_1 = sample_proto_loss_1 - pos_loss + torch.log(neg_loss_1 + torch.exp(pos_loss))
            # sample_proto_loss_2 = sample_proto_loss_2 - pos_loss + torch.log(neg_loss_2 + torch.exp(pos_loss))

        sample_proto_loss_1 = sample_proto_loss_1 / (bs)
        sample_proto_loss_2 = sample_proto_loss_2 / (bs)
                
        # for label in label_index: 
        #     positive_index = label_index[label]
        #     positive_embedding = torch.index_select(embedding, 0, torch.tensor(positive_index).cuda())
        #     negative_index = [i for i in range(bs) if i not in positive_index]
        #     negative_embedding = torch.index_select(embedding, 0, torch.tensor(negative_index).cuda())
        #     positive_num = len(positive_index)

        #     for i in range(positive_num):
        #         for j in range(positive_num):
        #             if i == j:
        #                 continue
        #             pos_loss = torch.cosine_similarity(positive_embedding[i].unsqueeze(0), positive_embedding[j].unsqueeze(0))
        #             neg_loss = torch.sum(torch.exp(torch.cosine_similarity(positive_embedding[i].unsqueeze(0), negative_embedding)))
        #             inter_sample_loss = inter_sample_loss - pos_loss + torch.log(neg_loss + torch.exp(pos_loss))
            # for i in range(positive_num):
            #     for j in range(positive_num):
            #         if i == j:
            #             continue
            #         pos_loss = torch.cosine_similarity(positive_embedding[i].unsqueeze(0), positive_embedding[j].unsqueeze(0))
            #         neg_loss = torch.sum(torch.exp(torch.cosine_similarity(positive_embedding[i].unsqueeze(0), negative_embedding)))
            #         inter_sample_loss = inter_sample_loss - pos_loss + torch.log(neg_loss)
            # pos_loss = 0.0
            # neg_loss = 0.0
            # for i in range(positive_num):
            #     pos_loss = pos_loss + torch.sum(torch.cosine_similarity(positive_embedding[i].unsqueeze(0), positive_embedding)) - 1
            #     cos = torch.cosine_similarity(positive_embedding[i].unsqueeze(0), negative_embedding)
            #     cos_sum = torch.sum(torch.exp(cos))
            #     neg_loss = neg_loss + torch.log(cos_sum)
            # inter_sample_loss = inter_sample_loss - pos_loss + neg_loss
        # inter_sample_loss = inter_sample_loss / (bs * 2)

        # sample-prototype loss
        # sample_proto_loss_1 = torch.tensor([0.0]).cuda()
        # sample_proto_loss_2 = torch.tensor([0.0]).cuda()
        # for i in range(bs):
        #     ll = labels[i].item()
        #     pos_loss = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight[ll].unsqueeze(0))
            
        #     # neg_index_1 = [j for j in range(self.num_labels) if j != ll]
        #     # negative_embedding_1 = torch.index_select(self.label_embedding.weight, 0, torch.tensor(neg_index_1).cuda())
        #     neg_loss_1 = torch.sum(torch.exp(torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight)))
            
        #     neg_index_2 = [j for j in range(bs) if labels[j].item()!=ll]
        #     negative_embedding_2 = torch.index_select(embedding, 0, torch.tensor(neg_index_2).cuda())
        #     neg_loss_2 = torch.sum(torch.exp(torch.cosine_similarity(self.label_embedding.weight[ll].unsqueeze(0), negative_embedding_2)))

        #     sample_proto_loss_1 = sample_proto_loss_1 - pos_loss + torch.log(neg_loss_1)
        #     sample_proto_loss_2 = sample_proto_loss_2 - pos_loss + torch.log(neg_loss_2 + torch.exp(pos_loss))
        #     # ll = labels[i].item()
        #     # pos_loss = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight[ll].unsqueeze(0))
            
        #     # neg_index_1 = [j for j in range(self.num_labels) if j != ll]
        #     # negative_embedding_1 = torch.index_select(self.label_embedding.weight, 0, torch.tensor(neg_index_1).cuda())
        #     # neg_loss_1 = torch.sum(torch.exp(torch.cosine_similarity(embedding[i].unsqueeze(0), negative_embedding_1)))
            
        #     # neg_index_2 = [j for j in range(bs) if labels[j].item()!=ll]
        #     # negative_embedding_2 = torch.index_select(embedding, 0, torch.tensor(neg_index_2).cuda())
        #     # neg_loss_2 = torch.sum(torch.exp(torch.cosine_similarity(self.label_embedding.weight[ll].unsqueeze(0), negative_embedding_2)))

        #     # sample_proto_loss_1 = sample_proto_loss_1 - pos_loss + torch.log(neg_loss_1 + torch.exp(pos_loss))
        #     # sample_proto_loss_2 = sample_proto_loss_2 - pos_loss + torch.log(neg_loss_2 + torch.exp(pos_loss))

        # sample_proto_loss_1 = sample_proto_loss_1 / (bs)
        # sample_proto_loss_2 = sample_proto_loss_2 / (bs)

        # classify loss
        # logits = self.classify_embedding(self.label_embedding.weight) # [num_labels, num_labels]
        # classify_labels = torch.tensor([i for i in range(self.num_labels)]).cuda()
        # loss_fn = torch.nn.CrossEntropyLoss()
        # classify_loss = loss_fn(logits, classify_labels)
        # logits = self.classify_embedding(embedding)
        # loss_fn = torch.nn.CrossEntropyLoss()
        # classify_loss = loss_fn(logits, labels)
        # logits = torch.tensor([]).cuda()
        # for i in range(bs):
        #     dist = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight).unsqueeze(0)
        #     logits = torch.cat((logits, dist), dim=0) # [batch_size, num_labels]
        # loss_fn = torch.nn.CrossEntropyLoss()
        # classify_loss = loss_fn(logits, labels)
        #return classify_loss,classify_loss,classify_loss,classify_loss


        return inter_sample_loss, sample_proto_loss_1, sample_proto_loss_2

    def get_sample_content(self):
        if os.path.exists("./content_select/select_" + str(self.seed) + ".txt"):
            print("reading content from file...")
            examples = []
            with open("./content_select/select_" + str(self.seed) + ".txt", 'r') as f:
                line = f.readline()
                while line:
                    splits = line.split('\t')
                    examples.append((splits[0], splits[1], int(splits[2])))
                    line = f.readline()
            return examples

        print("creating content...")
        contents = {label:[] for label in self.labels}
        contents_labels = {label:[] for label in self.labels}
        for label in self.labels:
            label_list = label.split(' ')
            sample_each_word = self.sample_num // len(label_list)
            for label_word in label_list:
                lines = []
                with open("./yahoo_contents/{}.txt".format(label_word), 'r') as f:
                    line = f.readline()
                    while(line):
                        # if label_word in line:
                        lines.append(line.strip())
                        line = f.readline()
                random.seed(self.seed)
                lines = random.sample(lines, sample_each_word)
                contents[label].extend(lines)
                contents_labels[label].extend([label_word]*sample_each_word)

        for label in self.labels:
            random.seed(self.seed)
            random.shuffle(contents[label])
            random.seed(self.seed)
            random.shuffle(contents_labels[label])
        
        examples = []
        
        f = open("./content_select/select_" + str(self.seed) + ".txt", 'w')
        # for i in range(self.batch_num):
        #     for label in self.labels:
        #         for j in range(sample_each):
        #             content = contents[label][i*sample_each+j]
        #             content_label = contents_labels[label][i*sample_each+j]
        #             label1 = self.label2idx[label]
        #             examples.append((content,content_label,label1))
        #             f.write(content + '\t' + content_label + '\t' + str(label1)+'\n')
        for label in self.labels:
            for i in range(len(contents[label])):
                content = contents[label][i]
                content_label = contents_labels[label][i]
                label1 = self.label2idx[label]
                examples.append((content,content_label,label1))
                f.write(content + '\t' + content_label + '\t' + str(label1)+'\n')
        f.close()
        return examples

    def convert_examples_to_features(self, examples, max_seq_length):
        special_tokens_count = 3
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_label_ids = []
        special_tokens_count += self.prompt.prompt_length
        for example in examples:
            content = example[0]
            word = example[1]
            label_ids = example[2]

            # for using < and > to wrap words up
            # words = [word, word.title(), word.upper()]
            # for w in words:
            #     content = content.replace(w, '<'+w+'>')
            # word = '<'+word+'>'

            tokens_a = self.tokenizer.tokenize(content)
            tokens_b = self.tokenizer.tokenize(word)

            if (len(tokens_a) + len(tokens_b)) > (max_seq_length - special_tokens_count):
                max_length = max_seq_length - special_tokens_count
                # truncate
                for_a_nums = max_length - len(tokens_b)
                if for_a_nums >= 0:
                    tokens_a = tokens_a[:for_a_nums]
                else:
                    tokens_a = []
                    tokens_b = tokens_b[:max_length]

            tokens, _ = self.prompt.wrap_text(self.tokenizer, tokens_a, tokens_b)
            # text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens))
            # print(text)
            # exit()

            assert len(tokens) <= max_seq_length - 3

            tokens += [sep_token]
            tokens += [sep_token]
            tokens = [cls_token] + tokens
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            all_input_ids.append(input_ids)
            all_attention_mask.append(input_mask)
            all_label_ids.append(label_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
        
        return TensorDataset(all_input_ids, all_attention_mask, all_label_ids)

    def convert_examples_to_features_not_prompt(self, examples, max_seq_length):
        special_tokens_count = 3
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_label_ids = []
        all_word_ids = []
        special_tokens_count += self.prompt.prompt_length
        for example in examples:
            content = example[0]
            word = example[1]
            label_ids = example[2]

            words = []
            words.append(word)
            words.append(word.title())
            words.append(word.upper())
            # using <mask>
            # for w in words:
            #     if w in content:
            #         content = content.replace(w, '<mask>', 1)
            #         break
            # assert '<mask>' in content

            tokens_a = self.tokenizer.tokenize(content)

            if len(tokens_a) > (max_seq_length - special_tokens_count):
                max_length = max_seq_length - special_tokens_count
                # truncate
                tokens_a = tokens_a[:max_length]

            # tokens, _ = self.prompt.wrap_text(self.tokenizer, tokens_a, tokens_b)
            # text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens))
            tokens = tokens_a

            assert len(tokens) <= max_seq_length - 3

            tokens += [sep_token]
            tokens += [sep_token]
            tokens = [cls_token] + tokens
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding_length = max_seq_length - len(input_ids)
            input_ids += [pad_token] * padding_length
            input_mask += [0] * padding_length

            # using word itself
            # wordss = []
            # for w in words:
            #     wordss.append(w)
            #     wordss.append('Ġ'+w)
            # for w in wordss:
            #     if w in tokens:
            #         all_word_ids.append(tokens.index(w))
            #         # print(tokens[tokens.index(w)])
            #         # exit()
            #         break

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length

            all_input_ids.append(input_ids)
            all_attention_mask.append(input_mask)
            all_label_ids.append(label_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
        all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
        
        return TensorDataset(all_input_ids, all_attention_mask, all_label_ids)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_args_parser():
    parser = argparse.ArgumentParser(description="Command line interface for Prompt-based text classification.")
    args = parser.parse_args()
    args.n_gpu = 1
    args.fp16 = False
    return args

def train(train_dataset, model, seed, batch_size=8, epochs=10, lr=3e-5, weight_decay=0, adam_epsilon=1e-8, max_grad_norm=1.0, warmup_proportion=0):
    train_batch_size = batch_size
    set_seed(seed)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=train_batch_size)
    t_total = len(train_dataloader)  * epochs
    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = []
    other_parameters = []
    for param in model.named_parameters():
        if "masked_model" in param[0]:
            bert_parameters += [param]
        else:
            other_parameters += [param]
    
    assert len(bert_parameters) != 0 and len(other_parameters) != 0
    
    bert_lr = lr
    cls_lr = lr

    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": bert_lr},
        {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": bert_lr},

        {"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": cls_lr},
        {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": cls_lr}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)
    warmup_steps = int(t_total * warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )
    # adversarial_training
    # if adv_training == 'fgm':
    #     adv = FGM(model=model, param_name='word_embeddings')
    # elif adv_training == 'pgd':
    #     adv = PGD(model=model, param_name='word_embeddings')
    
    # Train
    print("***** begin training *****")
    print("***** Num examples: {} *****".format(len(train_dataset)))
    
    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(epochs), desc="Epoch"
    )
    set_seed(seed)
    epoch_num = 1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.cuda() for t in batch)
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "labels":batch[2]
            }
            loss = model(**inputs)
            loss.backward()
            
            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
        epoch_num += 1
    
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir = os.path.join(output_dir, "last_checkpoint")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, "model"))
    print("saving model !!!")
    

def main():
    model_path = '../../../roberta-large'
    seed = 8
    dbpedia_labels = ['company', 'school university', 'artist', 'athlete', 'politics', 'transportation', 'building', 'river mountain lake', 'village', 'animal', 'plant tree', 'album', 'film', 'book publication']
    agnews_labels = ['world', 'sports', 'business', 'technology science']
    yahoo_labels = ['society culture','science mathematics','health','education reference','computers internet','sports','business finance','entertainment music','family relationships','politics government']
    
    set_seed(seed)
    model = Contrastive_model(yahoo_labels, model_path, seed=seed)
    examples = model.get_sample_content()
    train_dataset = model.convert_examples_to_features(examples, max_seq_length=512)

    model = model.cuda()

    train(train_dataset,model,seed=seed, batch_size=model.batch_size)


if __name__ == "__main__":
    main()