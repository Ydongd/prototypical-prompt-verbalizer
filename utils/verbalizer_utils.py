import json

class Verbalizer(object):
    def __init__(self, labels, tokenizer):
        self.labels = labels
        self.label2words = {}
        self.label2values = {}
        self.words2values = {}
        self.label2originwords = {}
        # self.word2label = {}
        self.tokenizer = tokenizer
        all_labels = [label.strip().split(' ') for label in labels]
        print(all_labels)
        # for label in all_labels:
        #     for label_word in label:
        #         if label_word.lower() not in words:
        #             print("word not in dictionary: {}".format(label_word))
        #         assert label_word.lower() in words
    
    def set_verbalizer(self, label_word, similar_words, similar_values, direct, space, lowercase):
        assert label_word in self.labels
        assert len(similar_words) == len(similar_values)
        if direct:
            self.label2originwords[label_word] = similar_words
            words = []
            values = []
            for i in range(len(similar_words)):
                word = similar_words[i]
                if 'Ġ'+word in self.tokenizer.get_vocab():
                    words.append('Ġ'+word)
                    values.append(similar_values[i])
            self.label2words[label_word] = words
            self.label2values[label_word] = values
            for i in range(len(words)):
                self.words2values[words[i]] = values[i]
            # for word in similar_words:
            #     self.word2label[word] = label_word
            return
        
        words = []
        values = []
        self.label2originwords[label_word] = similar_words
        # words = ['Ġ' + word for word in similar_words]
        # values = similar_values
        for i in range(len(similar_words)):
            word = similar_words[i]
            value = similar_values[i]
            if lowercase:
                cands = [word.lower()]
            else:
                cands = [word.lower(), word.upper(), word.title()]
            if space:
                cands = ['Ġ'+w for w in cands] # for prompts that have a space befor mask
            else:
                cands.extend(['Ġ'+w for w in cands])
            for cand in cands:
                if cand in self.tokenizer.get_vocab():
                    words.append(cand)
                    values.append(value)
    
        self.label2words[label_word] = words
        self.label2values[label_word] = values
        for i in range(len(words)):
            self.words2values[words[i]] = values[i]
        # for word in similar_words:
        #     self.word2label[word] = label_word

    def write_verbalizer_to_file(self, file):
        with open(file + '_words', 'w') as f:
            json.dump(self.label2words, f)
        with open(file + '_values', 'w') as f:
            json.dump(self.label2values, f)
    
    def set_verbalizer_from_file(self, file, direct, space, lowercase):
        with open(file + '_words', 'r') as f:
            label2words = json.load(f)
        with open(file + '_values', 'r') as f:
            label2values = json.load(f)

        for label_word in label2words:
            words = label2words[label_word]
            values = label2values[label_word]
            similar_words = []
            similar_values = []
            for i,word in enumerate(words):
                if word.startswith('Ġ'):
                    word = word[1:]
                word = word.lower()
                if word not in similar_words:
                    similar_words.append(word)
                    similar_values.append(values[i])
            self.set_verbalizer(label_word, similar_words, similar_values, direct, space, lowercase)
        # print("new label2words: {}".format(self.label2words))
