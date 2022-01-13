from arguments import get_args_parser, get_model_classes
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, YahooProcessor, DBpediaProcessor
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import SoftVerbalizer
import torch
from openprompt import PromptForClassification
from transformers import  AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
import os
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm, trange

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_acc_f1(labels, predictions, num_labels):
    labels = labels.reshape(-1)
    predictions = predictions.reshape(-1)
    # accuracy = (predictions == labels).sum() / len(predictions)
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')

    f1 = {i:0 for i in range(num_labels)}
    tp = {i:0 for i in range(num_labels)}
    fp = {i:0 for i in range(num_labels)}
    fn = {i:0 for i in range(num_labels)}

    assert labels.shape[0] == predictions.shape[0]
    for i in range(len(labels)):
        l = labels[i]
        p = predictions[i]
        if l == p:
            tp[l] += 1
        else:
            fn[l] += 1
            fp[p] += 1
    
    for i in range(num_labels):
        precision = tp[i] / (tp[i] + fp[i] + 1e-8)
        recall = tp[i] / (tp[i] + fn[i] + 1e-8)
        f1[i] = 2 * precision * recall / (precision + recall + 1e-8)
    
    tps = 0
    fps = 0
    fns = 0
    f1s = 0
    for i in range(num_labels):
        tps += tp[i]
        fps += fp[i]
        fns += fn[i]
        f1s += f1[i]
    
    ps = tps / (tps + fps)
    rs = tps / (tps + fns)
    micro_f11 = 2 * ps * rs / (ps + rs)
    macro_f11 = f1s / num_labels

    if micro_f1 != micro_f11 or macro_f1 != macro_f11:
        print("micro_f1 is {}".format(micro_f1))
        print("micro_f11 is {}".format(micro_f11))
        print("macro_f1 is {}".format(macro_f1))
        print("macro_f11 is {}".format(macro_f11))

    return micro_f1, macro_f1, f1

def main():
    args = get_args_parser()

    args.device = torch.device("cuda")

    set_seed(args)

    plm, tokenizer, model_config, WrapperClass = load_plm("roberta", args.model_name_or_path)
    prompt_index = args.prompt_index

    if args.task_name == 'agnews':
        print('Task is AGnews.')
        data_processor = AgnewsProcessor()
        template0 = ManualTemplate(tokenizer=tokenizer, text='A {"mask"} news: {"placeholder":"text_a"} {"placeholder":"text_b"}')
        template1 = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} {"placeholder":"text_b"} This topic is about {"mask"}.')
        template2 = ManualTemplate(tokenizer=tokenizer, text='[Category: {"mask"}] {"placeholder":"text_a"} {"placeholder":"text_b"}')
        template3 = ManualTemplate(tokenizer=tokenizer, text='[Topic: {"mask"}] {"placeholder":"text_a"} {"placeholder":"text_b"}')
        templates = [template0, template1, template2, template3]
        num_classes = 4
    elif args.task_name == 'yahoo':
        print('Task is Yahoo Answers.\n')
        data_processor = YahooProcessor()
        template0 = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_b"} {"placeholder":"text_a"} is a {"mask"}.')
        template1 = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_b"} In this sentence, {"placeholder":"text_a"} is a {"mask"}.')
        template2 = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_b"} The type of {"placeholder":"text_a"} is {"mask"}.')
        template3 = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_b"} The category of {"placeholder":"text_a"} is {"mask"}.')
        templates = [template0, template1, template2, template3]
        num_classes = 10
    elif args.task_name == 'dbpedia':
        print('Task is DBPedia.\n')
        data_processor = DBpediaProcessor()
        template0 = ManualTemplate(tokenizer=tokenizer, text='A {"mask"} question: {"placeholder":"text_a"} {"placeholder":"text_b"}')
        template1 = ManualTemplate(tokenizer=tokenizer, text='{"placeholder":"text_a"} {"placeholder":"text_b"} This topic is about {"mask"}.')
        template2 = ManualTemplate(tokenizer=tokenizer, text='[Category: {"mask"}] {"placeholder":"text_a"} {"placeholder":"text_b"}')
        template3 = ManualTemplate(tokenizer=tokenizer, text='[Topic: {"mask"}] {"placeholder":"text_a"} {"placeholder":"text_b"}')
        templates = [template0, template1, template2, template3]
        num_classes = 14
    else:
        print('Wrong Task.\n')
        exit()
    
    dataset = {}
    # dataset['train'] = data_processor.get_examples("./dataset/agnews/", 'train_sample_' + str(args.sample_num) + '_' + str(args.seed))
    dataset['test'] = data_processor.get_examples(args.data_dir, "test")

    verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=num_classes)

    model = PromptForClassification(plm=plm, template=templates[prompt_index], verbalizer=verbalizer, freeze_plm=False)
    model=  model.cuda()

    checkpoint = os.path.join(args.output_dir, 'last_checkpoint')
    # tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
    state_dict = torch.load(os.path.join(checkpoint, "model"))
    model.load_state_dict(state_dict)

    model.eval()

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=templates[prompt_index], tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=3, 
        batch_size=args.per_gpu_eval_batch_size,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    allpreds = []
    alllabels = []
    for inputs in tqdm(test_dataloader, desc="Evaluating"):
        inputs = inputs.cuda()
        with torch.no_grad():
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    
    allpreds = np.array(allpreds)
    alllabels = np.array(alllabels)
    micro_f1, macro_f1, f1 = get_acc_f1(alllabels, allpreds, num_classes)
    results = {}
    results['eval_loss'] = 0
    results['micro_f1'] = micro_f1
    results['macro_f1'] = macro_f1
    print("***** Eval results  *****")
    result_str = "Eval loss is {}\nMicro f1 is {}\nMacro f1 is {}\n".format(0, micro_f1, macro_f1)
    for i in range(num_classes):
        result_str += "f1 score of Class {} is: {}\n".format(i, f1[i])
    result_str += "\n\n"
    print(result_str)

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as f:
        f.write('***** Predict Result for {} sample_num {} seed {} prompt index {} *****\n'.format(args.task_name, args.sample_num, args.seed, args.prompt_index))
        f.write(result_str)

if __name__ == "__main__":
    main()