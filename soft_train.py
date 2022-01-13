from logging import exception
from arguments import get_args_parser, get_model_classes
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, YahooProcessor, DBpediaProcessor
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import SoftVerbalizer
import torch
from openprompt import PromptForClassification
from transformers import  AdamW, get_linear_schedule_with_warmup
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
    dataset['train'] = data_processor.get_examples(args.data_dir, 'train_sample_' + str(args.sample_num) + '_' + str(args.seed))
    # dataset['test'] = data_processor.get_examples("./dataset/agnews/", "test_sample_100_1")

    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=templates[prompt_index], tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=3, 
        batch_size=args.per_gpu_train_batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")
    
    verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=num_classes)

    model = PromptForClassification(plm=plm, template=templates[prompt_index], verbalizer=verbalizer, freeze_plm=False)
    model=  model.cuda()

    # training
    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5, "lr": 3e-5},
        {'params': [p for n, p in model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 5e-5, "lr": 3e-5}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': model.verbalizer.group_parameters_1, 'weight_decay': 5e-5, "lr":3e-5},
        {'params': model.verbalizer.group_parameters_2, 'weight_decay': 5e-5, "lr":3e-5},
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5, eps=args.adam_epsilon)
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=3e-5, eps=args.adam_epsilon)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)
    epoch_num = 1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, inputs in enumerate(epoch_iterator):
            model.train()
            inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            try:
                loss.backward()
            except RuntimeError:
                print(loss)
            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            model.zero_grad()
            global_step += 1
        epoch_num += 1

    output_dir = os.path.join(args.output_dir, "last_checkpoint")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, "model"))
    print("saving model to {}".format(output_dir))

if __name__ == "__main__":
    main()
