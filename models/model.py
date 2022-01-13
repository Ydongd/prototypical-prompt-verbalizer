from logging import log
import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from arguments import get_model_classes, get_args
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import re

class FinetuneModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.label2id = args.label2id
        self.id2label = args.id2label
        self.num_labels = args.num_labels
        self.labels = self.label2id.keys()
        # # label to bert vocab idx
        # self.label2idx = {label:self.tokenizer.convert_tokens_to_ids(label) for label in self.labels}
        # self.idx2label = {v:k for k,v in self.label2idx.items()}

        self.masked_model = model_config['classification'].from_pretrained(
            args.model_name_or_path, num_labels=self.num_labels
        )
    
    def forward(self, input_ids, token_type_ids, attention_mask, labels, prompt_mask, mode):
        outputs  = self.masked_model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids, labels=labels)
        loss = outputs[0]
        logits = outputs[1]
        preds = torch.argmax(logits, dim=-1)
        return loss, preds


class PromptTuneModel(torch.nn.Module):
    def __init__(self, args, tokenizer, verbalizer):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.label2id = args.label2id
        self.id2label = args.id2label
        self.tokenizer = tokenizer
        self.num_labels = args.num_labels
        self.labels = self.label2id.keys()
        self.space = args.space_before_mask
        self.lowercase = args.mask_must_lower
        # # label to bert vocab idx
        # self.label2idx = {label:self.tokenizer.convert_tokens_to_ids(label) for label in self.labels}
        # self.idx2label = {v:k for k,v in self.label2idx.items()}

        self.masked_model = model_config['masked_lm'].from_pretrained(
            args.model_name_or_path
        )

        mask_token = self.tokenizer.mask_token
        self.mask_idx = self.tokenizer.convert_tokens_to_ids(mask_token)

        # self.words = word_utils.words

        related_words = {label:[] for label in verbalizer.labels}

        for i, label in enumerate(verbalizer.labels):
            similar_words = label.split(' ')
            similar_values = list(range(2, len(similar_words)+2))

            ##########
            # if ' ' not in label:
            #     with open('./pretrain/related_words/dbpedia_refine/{}.txt'.format(label)) as f:
            #         similar_words = f.readlines()
            #         similar_words = [line.strip().split('\t')[0] for line in similar_words][:10]
            #         similar_values = list(range(2, len(similar_words)+2))
            # else:
            #     ll = label.split(' ')
            #     similar_words = []
            #     for l in ll:
            #         with open('./pretrain/related_words/dbpedia_refine/{}.txt'.format(l)) as f:
            #             ss = f.readlines()
            #             ss = [line.strip().split('\t')[0] for line in ss][:10]
            #             similar_words.extend(ss)
            #     similar_values = list(range(2, len(similar_words)+2))
            ##########

            print(similar_words)
            verbalizer.set_verbalizer(label, similar_words, similar_values, direct=True, space=self.space, lowercase=self.lowercase)
        self.verbalizer = verbalizer
        print(verbalizer.label2words)

    def forward(self, input_ids, token_type_ids, attention_mask, labels, prompt_mask, mode):
        batch_size = input_ids.size(0)
        mask_index = input_ids.eq(self.mask_idx) # size:[batch_size, max_seq_length]
        mask_index = torch.nonzero(mask_index) # size:[batch_size, 2]
        seq = mask_index[:, 0] # size:[batch_size]
        mask_index = mask_index[:, 1] # size:[batch_size]
        # if mode == 'train':
        # labels_idx = input_ids.clone()
        # all_words = [self.verbalizer.label2words[self.id2label[label.item()]][0] for label in labels]
        # all_words_idx = self.tokenizer.convert_tokens_to_ids(all_words)
        # labels_idx[seq, mask_index] = torch.tensor(all_words_idx).cuda()

        output = self.masked_model(input_ids=input_ids,
                                   token_type_ids=token_type_ids,
                                   attention_mask=attention_mask,)
                                   # labels=labels_idx)
        # print(output)
        # print("loss1: {}".format(output['loss'].item()))
        # output['logits'] size:[batch_size, max_seq_length, vocab_size]
        # output['logits'][seq, mask_index] size:[batch_size, vocab_size]
        logits = output['logits'][seq, mask_index]

        if mode == 'train':
            loss = self.mlm_loss(input_ids, output['logits'], labels, mask_index, self.space)
            max_prob = self.max_head(logits)
            loss_fn = torch.nn.CrossEntropyLoss()
            cross_loss = loss_fn(max_prob, labels)
            preds = torch.argmax(max_prob, dim=-1)
            alpha = 0.0004
            loss = alpha * loss + (1-alpha) * cross_loss
            return loss, preds
        else: # model is test or eval
            max_prob = self.max_head(logits)
            # [batch_size,]
            preds = torch.argmax(max_prob, dim=-1)
            loss = torch.tensor(0)
            # print("loss2: {}".format(loss.item()))
            return loss, preds
    
    def max_head(self, logits):
        # logits: [batch_size, vocab_size]
        max_prob = []
        for label in self.labels:
            original_words = self.verbalizer.label2words[label]
            words = self.tokenizer.convert_tokens_to_ids(original_words)
            # [batch_size, word_nums]
            words_logits = logits[:, words]
            maxx = torch.max(words_logits, dim=-1)
            max_words = maxx[0]
            max_prob.append(max_words)
        # [batch_size, num_labels]
        max_prob = torch.stack(max_prob, dim=-1)
        return max_prob
    
    def average_head(self, logits):
        max_prob = []
        for label in self.labels:
            original_words = self.verbalizer.label2words[label]
            words = self.tokenizer.convert_tokens_to_ids(original_words)
            values = self.verbalizer.label2values[label]
            # [batch_size, word_num]
            words_logits = logits[:, words]
            prob = []
            for i in range(words_logits.size(0)):
                max_word = 0
                max_value = 0
                max_words = []
                max_values = []
                for j in range(words_logits.size(1)):
                    if j == 0:
                        max_word = words_logits[i][j]
                        max_value = values[j]
                    elif j != 0 and values[j] == values[j-1]:
                        max_word = max(max_word, words_logits[i][j])
                    else:
                        max_words.append(max_word)
                        max_values.append(max_value)
                        max_word = words_logits[i][j]
                        max_value = values[j]
                max_words.append(max_word)
                max_values.append(max_value)

                max_words = F.log_softmax(torch.tensor(max_words))
                # print(max_words)
                max_values = torch.tensor([1.0] * max_words.size(0))
                # print(max_values)

                prob.append(torch.dot(max_words, max_values))
            max_prob.append(prob)
        max_prob = torch.tensor(max_prob)
        max_prob = max_prob.T
        print(max_prob)
        return max_prob
    
    def weight_head(self, logits):
        max_prob = []
        for label in self.labels:
            original_words = self.verbalizer.label2words[label]
            words = self.tokenizer.convert_tokens_to_ids(original_words)
            values = self.verbalizer.label2values[label]
            # [batch_size, word_num]
            words_logits = logits[:, words]
            prob = []
            for i in range(words_logits.size(0)):
                max_word = 0
                max_value = 0
                max_words = []
                max_values = []
                for j in range(words_logits.size(1)):
                    if j == 0:
                        max_word = words_logits[i][j]
                        max_value = values[j]
                    elif j != 0 and values[j] == values[j-1]:
                        max_word = max(max_word, words_logits[i][j])
                    else:
                        max_words.append(max_word)
                        max_values.append(max_value)
                        max_word = words_logits[i][j]
                        max_value = values[j]
                max_words.append(max_word)
                max_values.append(max_value)

                max_words = torch.tensor(max_words)
                max_values = F.softmax(torch.tensor(max_values))

                prob.append(torch.dot(max_words, max_values))
            max_prob.append(prob)
        max_prob = torch.tensor(max_prob)
        max_prob = max_prob.T
        print(max_prob)
        return max_prob

    def mlm_loss(self, input_ids, prediction_scores, labels, mask_index, space):
        # space indicates whether there is a space before <mask> 
        # input_ids: [batch_size, max_seq_length]
        # prediction_scores: [batch_size, max_seq_length, vocab_size]
        # labels: [batch_size]
        # mask_index: [batch_size]
        vocab_size = prediction_scores.size(-1)
        all_words = {}
        for label_word in self.labels:
            similar_words = self.verbalizer.label2words[label_word]
            words = []
            for word in similar_words:
                if space:
                    if word.startswith('Ġ'):
                        words.append(word)
                else:
                    words.append(word)
            all_words[label_word] = words
        
        batch_size = len(labels)
        loss_fn = torch.nn.CrossEntropyLoss()

        # [batch_size,]
        # idx_labels = torch.tensor([self.label2idx[label] for label in labels]).cuda()
        # [batch_size, max_seq_length]
        # labels_idx = input_ids.clone()
        # labels_idx[seq, mask_index] = idx_labels
        all_loss = torch.tensor(0.0).cuda()
        assert batch_size == input_ids.size(0)
        # for each sample, calculate its all possible words' cross entropy
        for i in range(batch_size):
            label = labels[i]
            label_word = self.id2label[label.item()]
            num_words = len(all_words[label_word])
            input_tmp = input_ids[i]
            mask_tmp = mask_index[i]
            # [num_words, max_seq_length]
            input_tmp = input_tmp.repeat(num_words, 1)
            labels_idx = input_tmp.clone()
            label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(all_words[label_word]))
            assert label_ids.size(0) == num_words
            labels_idx[:, mask_tmp] = label_ids
            labels_idx = labels_idx.cuda()

            # predictions = prediction_scores[i].clone()
            # predictions = predictions.repeat(num_words, 1, 1)
            # predictions = predictions.cuda()
            tmp_loss = 0
            for j in range(num_words):
                tmp_loss += loss_fn(prediction_scores[i].view(-1, vocab_size), labels_idx[j].view(-1))
            tmp_loss /= num_words
            # loss = loss_fn(predictions.view(-1, vocab_size), labels_idx.view(-1))
            all_loss += tmp_loss
        
        all_loss = all_loss / batch_size

        return all_loss

class ContrastiveModel(torch.nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]
        self.label2id = args.label2id
        self.id2label = args.id2label
        self.tokenizer = tokenizer
        self.num_labels = args.num_labels
        self.labels = self.label2id.keys()
        self.space = args.space_before_mask
        self.lowercase = args.mask_must_lower
        self.embedding_size = 256

        self.masked_model = model_config['model'].from_pretrained(args.model_name_or_path)
        mask_token = self.tokenizer.mask_token
        self.mask_idx = self.tokenizer.convert_tokens_to_ids(mask_token)

        # self.classify_embedding = nn.Linear(self.embedding_size, self.num_labels)
        self.transform_matrix = nn.Linear(1024, self.embedding_size)
        self.label_embedding = nn.Embedding(self.num_labels, self.embedding_size)
    
    def forward(self, input_ids, token_type_ids, attention_mask, labels, prompt_mask, mode):
        batch_size = input_ids.size(0)
        mask_index = input_ids.eq(self.mask_idx)
        mask_index = torch.nonzero(mask_index)
        seq = mask_index[:, 0]
        mask_index = mask_index[:, 1]

        output = self.masked_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output['last_hidden_state'] # [batch_size, max_seq_length, hidden_size]
        embedding = last_hidden_state[seq, mask_index] # [batch_size, hidden_size]
        embedding = self.transform_matrix(embedding)

        logits = torch.tensor([]).cuda()
        for i in range(batch_size):
            dist = torch.cosine_similarity(embedding[i].unsqueeze(0), self.label_embedding.weight).unsqueeze(0)
            logits = torch.cat((logits, dist), dim=0) # [batch_size, num_labels]

        assert batch_size == embedding.size(0)
        if mode == 'train':
            loss1, loss2, loss3 = self.sample_proto_loss(labels, embedding)
            # loss_fn = torch.nn.CrossEntropyLoss()
            # loss4 = loss_fn(logits, labels)
            a1 = 0.5
            a2 = 0.5
            # a3 = 0.5
            loss = a1 * loss1 + a2 * (loss2 + loss3)
            return loss, []
        else:
            values, preds = torch.max(logits, dim=-1)
            loss = torch.tensor(0)
            return loss, preds

    def sample_proto_loss(self, labels, embedding):
        # labels: [batch_size]
        # embedding: [batch_size, hidden_size]
        # embedding = self.transform_matrix(embedding)
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

        sample_proto_loss_1 = sample_proto_loss_1 / (bs)
        sample_proto_loss_2 = sample_proto_loss_2 / (bs)

        return inter_sample_loss, sample_proto_loss_1, sample_proto_loss_2
