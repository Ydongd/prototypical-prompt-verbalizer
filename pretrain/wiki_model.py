import sys

from nltk.corpus.reader.wordlist import WordListCorpusReader
sys.path.append("..")
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from utils.prompt_utils import Prompt
import torch
from transformers import AutoModel, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM
import torch.nn as nn
import random
import time
from torch.utils.data import TensorDataset
import math
from tqdm import tqdm
import json
import pandas as pd

class Wiki_word():
    def __init__(self, wiki_path, model_path):
        t1 = time.time()
        self.content = []
        with open(wiki_path, 'r') as f:
            line = f.readline()
            while line:
                self.content.append(line.strip())
                line = f.readline()
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.masked_model = RobertaForMaskedLM.from_pretrained(model_path).cuda()
        self.mask_token = self.tokenizer.mask_token
        self.mask_idx = self.tokenizer.convert_tokens_to_ids(self.mask_token)
        self.stopwords = list(set(stopwords.words('english')))

        # self.content_num = 100
        # self.word_num = 50
        # self.threshold = 0.5
        # self.batch_size = 5
        # prompt_text = "{} In this sentence, {} means <mask>."
        # self.prompt = Prompt(prompt_text, 2)
        # self.prompt.set_attributes(self.tokenizer.tokenize(prompt_text))
        with open('./somestop.txt', 'r') as f:
            somestop = f.readlines()
        self.somestop = [stop.strip() for stop in somestop]
        print("initialized taking {} secs...".format(time.time()-t1))
    
    def get_agnews_content(self, data_path):
        path = os.path.join(data_path, 'train.csv')
        labels = ['world', 'sports', 'business', 'technology', 'science']
        contents = []
        data = pd.read_csv(path, header=None)
        for i in range(len(data)):
            if data.iloc[i][2] == data.iloc[i][2]:
                contents.append(data.iloc[i][2])
        for word in labels:
            if os.path.exists('./agnews_contents/{}.txt'.format(word)):
                print("existing content words file of {}".format(word))
                all_sents = []
                with open('./agnews_contents/{}.txt'.format(word), 'r') as f:
                    line = f.readline()
                    while line:
                        all_sents.append(line.strip())
                        line = f.readline()
                continue
            print("creating content words file of {}".format(word))
            all_sents = []
            for c in contents:
                if word in c.lower():
                    sents = sent_tokenize(c)
                    for sent in sents:
                        if word in sent.lower():
                            splits = word_tokenize(sent.lower())
                            # if word in splits and len(splits) > 10 and len(splits) < 150:
                            if word in splits and len(splits) > 10:
                                all_sents.append(sent)
            with open('./agnews_contents/{}.txt'.format(word), 'w') as f:
                for sent in all_sents:
                    f.write(sent + '\n')
        return all_sents
    
    def get_dbpedia_content(self, data_path):
        path = os.path.join(data_path, 'train.txt')
        labels = ['company', 'school', 'university', 'artist', 'athlete', 'politics', 'transportation', 'building', 'river', 'mountain', 'lake', 'village', 'animal', 'plant', 'tree', 'album', 'film', 'book', 'publication']
        with open(path, 'r') as f:
            data = f.readlines()
        contents = []
        for i in range(len(data)):
            content = data[i].split('.', 1)[1]
            if data[i].startswith('...'):
                content = data[i][3:].split('.', 1)[1]
            if data[i].startswith('. . .'):
                content = data[i][6:].split('.', 1)[1]
            contents.append(content)
        for word in labels:
            if os.path.exists('./dbpedia_contents/{}.txt'.format(word)):
                print("existing content words file of {}".format(word))
                all_sents = []
                with open('./dbpedia_contents/{}.txt'.format(word), 'r') as f:
                    line = f.readline()
                    while line:
                        all_sents.append(line.strip())
                        line = f.readline()
                continue
            print("creating content words file of {}".format(word))
            all_sents = []
            for c in contents:
                if word in c.lower():
                    sents = sent_tokenize(c)
                    for sent in sents:
                        if word in sent.lower():
                            splits = word_tokenize(sent.lower())
                            # if word in splits and len(splits) > 10 and len(splits) < 150:
                            if word in splits and len(splits) > 10:
                                all_sents.append(sent)
            with open('./dbpedia_contents/{}.txt'.format(word), 'w') as f:
                for sent in all_sents:
                    f.write(sent + '\n')
        return all_sents

    def get_yahoo_content(self, data_path):
        path = os.path.join(data_path, 'train.csv')
        labels = ['society','culture','science','mathematics','health','education','reference','computers','internet','sports','business','finance','entertainment','music','family','relationships','politics','government']
        contents = []
        data = pd.read_csv(path, header=None)
        for i in range(len(data)):
            if data.iloc[i][1] == data.iloc[i][1]:
                contents.append(data.iloc[i][1])
                
            if data.iloc[i][2] == data.iloc[i][2]:
                contents.append(data.iloc[i][2])
            
            if data.iloc[i][3] == data.iloc[i][3]:
                contents.append(data.iloc[i][3])
            
        for word in labels:
            if os.path.exists('./yahoo_contents/{}.txt'.format(word)):
                print("existing content words file of {}".format(word))
                all_sents = []
                with open('./yahoo_contents/{}.txt'.format(word), 'r') as f:
                    line = f.readline()
                    while line:
                        all_sents.append(line.strip())
                        line = f.readline()
                continue
            print("creating content words file of {}".format(word))
            all_sents = []
            for c in contents:
                if word in c.lower():
                    sents = sent_tokenize(c)
                    for sent in sents:
                        if word in sent.lower():
                            splits = word_tokenize(sent.lower())
                            # if word in splits and len(splits) > 10 and len(splits) < 150:
                            if word in splits and len(splits) > 10:
                                all_sents.append(sent)
            with open('./yahoo_contents/{}.txt'.format(word), 'w') as f:
                for sent in all_sents:
                    f.write(sent + '\n')
        return all_sents

    def get_content(self, word, sample=False):
        if os.path.exists('./content_words/{}.txt'.format(word)):
            print("existing content words file of {}".format(word))
            all_sents = []
            with open('./content_words/{}.txt'.format(word), 'r') as f:
                line = f.readline()
                while line:
                    all_sents.append(line.strip())
                    line = f.readline()
            return all_sents

        print("creating content words file of {}".format(word))
        all_sents = []
        for c in self.content:
            if word in c.lower():
                sents = sent_tokenize(c)
                for sent in sents:
                    if word in sent.lower():
                        splits = word_tokenize(sent.lower())
                    #     splits = len(sent.split(' '))
                        if word in splits and len(splits) > 10 and len(splits) < 150:
                            all_sents.append(sent)
        if sample:
            all_sents = random.sample(all_sents, min(self.content_num, len(all_sents)))
        with open('./content_words/{}.txt'.format(word), 'w') as f:
            for sent in all_sents:
                f.write(sent + '\n')
        return all_sents

    def get_related_words(self, word):
        if os.path.exists('./words_select/{}.txt'.format(word)):
            print("existing related words file of {}".format(word))
            related_words = []
            with open('./words_select/{}.txt'.format(word), 'r') as f:
                line = f.readline()
                while line:
                    related_words.append(line.strip())
                    line = f.readline()
            return related_words
        
        print("creating related words of {}".format(word))
        contents = self.get_content(word)

        if not contents:
            related_words = []
            with open('./words_select/{}.txt'.format(word),'w') as f:
                for cand in related_words:
                    f.write(cand + '\n')
            return related_words

        all_input_ids, all_attention_mask = self.convert_contents_to_features(word, contents, 256)        
        batch_size = self.batch_size
        batch = math.ceil(len(all_input_ids)/batch_size)
        cands_words = {}
        for i in tqdm(range(batch), ncols=80):
            input_ids = all_input_ids[i*batch_size : (i+1)*batch_size]
            attention_mask = all_attention_mask[i*batch_size : (i+1)*batch_size]
            input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
            outputs = self.masked_model(input_ids=input_ids, attention_mask=attention_mask)
            
            mask_index = input_ids.eq(self.mask_idx)
            mask_index = torch.nonzero(mask_index)
            seq = mask_index[:, 0]
            mask_index = mask_index[:, 1]
            logits = outputs['logits'][seq, mask_index]
            a, b = torch.topk(logits, self.word_num, -1)
            # for each content in a batch
            for bb in b:
                cands = self.tokenizer.convert_ids_to_tokens(bb)
                # cands = [cand[1:].lower() if cand.startswith('Ġ') else cand for cand in cands]
                # cands_words.extend(cands)
                for cand in cands:
                    cand = cand[1:].lower() if cand.startswith('Ġ') else cand.lower()
                    if cand in self.stopwords:
                        continue
                    if cand not in cands_words:
                        cands_words[cand] = 0
                    cands_words[cand] += 1
                
        # cands_words = list(set(cands_words))
        # related_words = []
        # eliminate_words = []
        # for cand in cands_words:
        #     if cand in eliminate_words or cand in related_words or cand in self.stopwords:
        #         continue
        #     if self.test_cand_word(cand, word):
        #         related_words.append(cand)
        #     else:
        #         eliminate_words.append(cand)
        # with open('./words_select/{}.txt'.format(word),'w') as f:
        #     for cand in related_words:
        #         f.write(cand + '\n')

        cands_words = sorted(cands_words.items(),key=lambda x:x[1],reverse=True)
        cands_words = cands_words[:1000]
        with open('./words_select/{}.txt'.format(word),'w') as f:
            for cand in cands_words:
                f.write(cand[0] + '\t' + str(cand[1]) + '\n')

        related_words = cands_words[:10]

        return related_words
    
    def jaccard(self, set1, set2):
        union_set = len(list(set(set1)|set(set2)))
        if union_set == 0:
            union_set = 1
        intersection_set = len(list(set(set1)&set(set2)))
        # print(list(set(set1)&set(set2)))
        j = intersection_set/union_set
        return j

    def denoise_related_words(self, label):
        with open('./words_select/{}.txt'.format(label), 'r') as f:
            lines = f.readlines()
            all_words = [line.strip() for line in lines]

        with open('../../../roberta-large/vocab.json') as f:
            roberta_vocab = json.load(f)
        roberta_vocab = list(roberta_vocab.keys())
        roberta_vocab = [word[1:].lower() if word.startswith('Ġ') else word.lower() for word in roberta_vocab]
        roberta_vocab = list(set(roberta_vocab))
        with open('../words/all_related_words.json', 'r') as f:
                related_words_dict = json.load(f)

        related_words = []
        label_words = [t[0] for t in related_words_dict[label]]
        label_words = [word.lower() for word in label_words if word.lower() in roberta_vocab]
        label_words.append(label)
        # with open('./words_select/{}.txt'.format(label), 'r') as f:
        #     lines = f.readlines()
        #     label_words = [line.strip().split('\t')[0] for line in lines]
        for word_freq in all_words:
            splits = word_freq.split('\t')
            word = splits[0]
            freq = splits[1]
            if word in self.somestop:
                continue
            # using relatedwords.org to denoise the related_words
            cand_words = [t[0] for t in related_words_dict[word]]
            cand_words = [word.lower() for word in cand_words if word.lower() in roberta_vocab]
            cand_words.append(word)
            # with open('./words_select/{}.txt'.format(label), 'r') as f:
            #     lines = f.readlines()
            #     label_words = [line.strip().split('\t')[0] for line in lines]
            minlen = min(len(cand_words), len(label_words))
            if self.jaccard(cand_words[:minlen], label_words[:minlen]) >= 0.01:
                related_words.append((word, freq))
        
        with open('./related_words/dbpedia_not_refine/{}.txt'.format(label),'w') as f:
            for cand in related_words:
                f.write(cand[0] + '\t' + cand[1] + '\n')
        # cand_words = [t[0] for t in related_words_dict[word]]
        # cand_words = [word.lower() for word in cand_words if word.lower() in roberta_vocab]
        # cand_words.append(word)
        # minlen = min(len(cand_words), len(label_words))
        # print(self.jaccard(cand_words[:minlen], label_words[:minlen]))

    def test_cand_word(self, word, source):
        pass
            
    def test_cand_word_old(self, word, source):
        print("test word of {}".format(word))
        contents = self.get_content(word, sample=True)
        if not contents:
            return False
        all_input_ids, all_attention_mask = self.convert_contents_to_features(word, contents, 256)        
        batch_size = self.batch_size
        batch = math.ceil(len(all_input_ids)/batch_size)
        have_source = 0
        for i in range(batch):
            input_ids = all_input_ids[i*batch_size : (i+1)*batch_size]
            attention_mask = all_attention_mask[i*batch_size : (i+1)*batch_size]
            input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
            outputs = self.masked_model(input_ids=input_ids, attention_mask=attention_mask)
            
            mask_index = input_ids.eq(self.mask_idx)
            mask_index = torch.nonzero(mask_index)
            seq = mask_index[:, 0]
            mask_index = mask_index[:, 1]
            logits = outputs['logits'][seq, mask_index]
            a, b = torch.topk(logits, self.word_num, -1)
            for bb in b:
                cands = self.tokenizer.convert_ids_to_tokens(bb)
                cands = [cand[1:].lower() if cand.startswith('Ġ') else cand.lower() for cand in cands]
                if source in cands: # having source words
                    have_source += 1
        if have_source >= len(contents)*self.threshold:
            print('{} is a related word of {}'.format(word, source))
            return True
        else:
            print('{} is not a related word of {}'.format(word, source))
            return False     

    # todo: select content based on related words (enough candidate words are predicted from mask)     
    def refine_related_words(self, labels, num=100):
        # labels is a list        
        # get length of contents
        softmax = nn.Softmax()
        content_length = {}
        for label in labels:
            content_length[label] = self.get_content_length(label)
        # get related words and its probabilities
        related_words = {}
        related_freqs = {}
        all_words = []
        for label in labels:
            words = []
            freqs = []
            with open('./related_words/dbpedia_not_refine/{}.txt'.format(label), 'r') as f:
                line = f.readline()
                while line:
                    line = line.strip().split('\t')
                    words.append(line[0])
                    freqs.append(int(line[1])/content_length[label])
                    line = f.readline()
            freqs = softmax(torch.tensor(freqs))
            related_words[label] = words
            related_freqs[label] = freqs
            all_words.extend(words)
        # put all the words together, enumerate each word to allocate it to corresponding label
        all_words = list(set(all_words))
        new_related_words = {label:[] for label in labels}
        for word in all_words:
            max_freq = 0
            max_label = ''
            for label in labels:
                if word in related_words[label]:
                    word_index = related_words[label].index(word)
                    word_freq = related_freqs[label][word_index]
                    if word_freq > max_freq:
                        max_freq = word_freq
                        max_label = label
            new_related_words[max_label].append((word, max_freq.item()))

        for label in labels:
            words = new_related_words[label]
            words = sorted(words,key=lambda x:x[1],reverse=True)
            with open('./related_words/dbpedia_refine/{}.txt'.format(label),'w') as f:
                for cand in words:
                    f.write(cand[0] + '\t' + str(cand[1]) + '\n')  

    def get_content_length(self, word):
        num = 0
        with open('./content_words/{}.txt'.format(word), 'r') as f:
            line = f.readline()
            while(line):
                num += 1
                line = f.readline()
        return num

    # this function now only supports roberta
    def convert_contents_to_features(self, word, contents, max_seq_length):
        special_tokens_count = 3
        cls_token = self.tokenizer.cls_token
        sep_token = self.tokenizer.sep_token
        pad_token = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        special_tokens_count += self.prompt.prompt_length
        for content in contents:
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
        
        return all_input_ids, all_attention_mask


def main():
    wiki_path = '../../../wiki_content.txt'
    model_path = '../../../roberta-large'
    wiki_word = Wiki_word(wiki_path, model_path)
    words = ['sports', 'world', 'business', 'technology', 'company', 'school', 'university', 'artist', 'athlete', 'politics', 'transportation', 'building', 'river', 'mountain', 'lake', 'village', 'animal', 'plant', 'tree', 'album', 'film', 'book', 'publication']
    # for word in words:
    #     wiki_word.get_related_words(word)
    # agnews_labels = ['world', 'sports', 'business', 'technology']
    dbpedia_labels = ['company', 'school', 'university', 'artist', 'athlete', 'politics', 'transportation', 'building', 'river', 'mountain', 'lake', 'village', 'animal', 'plant', 'tree', 'album', 'film', 'book', 'publication']
    words = ['society', 'culture', 'science', 'mathematics', 'health', 'education', 'reference', 'computers' ,'internet', 'sports', 'business' ,'finance', 'entertainment','music', 'family','relationships', 'politics','government']
    # for word in words:
    #     wiki_word.get_content(word)
    # for label in dbpedia_labels:
    #     wiki_word.denoise_related_words(label)
    # wiki_word.refine_related_words(dbpedia_labels)
    # wiki_word.get_agnews_content('../dataset/agnews')
    # wiki_word.get_dbpedia_content('../dataset/dbpedia')
    wiki_word.get_content('nation')


if __name__ == "__main__":
    main()