import re

class Prompt_utils(object):
    def __init__(self, path, tokenizer, text_num):
        self.path = path
        self.tokenizer = tokenizer
        self.prompts = []
        with open(self.path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    self.prompts.append(Prompt(line, text_num))
        
        for prompt in self.prompts:
            tokens = tokenizer.tokenize(prompt.prompt_text)
            prompt.set_attributes(tokens)
    
    def get_prompts_num(self):
        return len(self.prompts)
    
    # def wrap_text(self, text_a='', text_b='', prompt_index=0, is_single=False, mask_num=1):
    #     # here should make sure that all the token lengths do not add up to 512
    #     assert mask_num == self.prompts_mask_num[prompt_index]
    #     prompt = self.prompts[prompt_index]
    #     if is_single:
    #         prompt = prompt.format(text_a)
    #     else:
    #         prompt = prompt.format(text_a, text_b)
        
    #     prompt = re.sub("\s+", " ", prompt)

    #     return prompt
    
# this class now only works for roberta
class Prompt(object):
    def __init__(self, prompt_text, text_num):
        self.prompt_text = prompt_text
        self.text_num = text_num
    
    def set_attributes(self, tokens):
        nums = tokens.count('Ġ{}')
        slot1 = 0
        slot2 = 0
        # whether there is a space before slot1
        space1 = False
        if self.text_num == 2:
            if nums == 2:
                slot = [i for i, x in enumerate(tokens) if x == 'Ġ{}']
                slot1 = slot[0]
                slot2 = slot[1]
            elif nums == 1:
                for i, x in enumerate(tokens):
                    if tokens[i] == '{' and tokens[i+1] == '}':
                        slot1 = i
                        space1 = True
                        break
                tokens.pop(slot1+1)
                tokens[slot1] = '{}'
                slot2 = tokens.index('Ġ{}')
            else: # nums1 == 0
                print("Wrong Prompt!!!!!!")
                exit()
        
        else: # self.text_num == 1
            if nums == 1:
                slot1 = tokens.index('Ġ{}')
                slot2 = -1
            else: # nums1 == 0
                for i, x in enumerate(tokens):
                    if tokens[i] == '{' and tokens[i+1] == '}':
                        slot1 = i
                        space1 = True
                        break
                tokens.pop(slot1+1)
                tokens[slot1] = '{}'
                slot2 = -1
        
        if slot1 == 0 and slot2 == 0:
            print('Wrong Prompt!!!!!!')
            exit()
        
        self.slot1 = slot1
        self.slot2 = slot2
        self.space1= space1
        self.tokens = tokens

        self.prompt_length = len(tokens) - 2
        self.mask_nums = tokens.count('<mask>')
    
    def wrap_text(self, tokenizer, tokens_a=[], tokens_b=[]):
        new_tokens = self.tokens.copy()
        ret_tokens = []
        prompt_mask = []
        if self.text_num == 1:
            if not self.space1:
                tokens_a[0] = 'Ġ' + tokens_a[0]
            new_tokens[self.slot1] = tokens_a
            for tt in new_tokens:
                if type(tt) == list:
                    ret_tokens.extend(tt)
                    prompt_mask.extend([0] * len(tt))
                else:
                    ret_tokens.append(tt)
                    prompt_mask.append(1)

        elif self.text_num == 2:
            if not self.space1:
                tokens_a[0] = 'Ġ' + tokens_a[0]
            # if not tokens_b:
            #     print(tokens_a)
            new_tokens[self.slot1] = tokens_a
            if tokens_b:
                tokens_b[0] = 'Ġ' + tokens_b[0]
                new_tokens[self.slot2] = tokens_b
            else: # tokens_b is empty
                new_tokens.pop(self.slot2)
            for tt in new_tokens:
                if type(tt) == list:
                    ret_tokens.extend(tt)
                    prompt_mask.extend([0] * len(tt))
                else:
                    ret_tokens.append(tt)
                    prompt_mask.append(1)
        
        # print(ret_tokens)
        # print(prompt_mask)
        # print(tokenizer.decode(tokenizer.convert_tokens_to_ids(ret_tokens)))
        
        return ret_tokens, prompt_mask
