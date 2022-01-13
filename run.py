from arguments import get_args_parser, get_model_classes
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
from utils.prompt_utils import Prompt_utils, Prompt
from utils.data_processor import AgnewsProcessor, YahooAnswersProcessor, DBPediaProcessor
from utils.data_utils import InputExample, InputFeature, load_examples, load_examples_4_finetune
from utils.utils_adversarial import FGM, PGD
from models.model import FinetuneModel, PromptTuneModel, ContrastiveModel
from sklearn.metrics import f1_score
from utils.verbalizer_utils import Verbalizer
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoConfig
from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger(__name__)
writer = SummaryWriter(log_dir=f"./runs")

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

def train(args, train_dataset, dev_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=InputFeature.collate_fct
                                  )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    if args.fine_tune or args.prompt_tune:
        bert_parameters = []
        cls_parameters = []
        for param in model.named_parameters():
            if args.model_type in param[0]:
                bert_parameters += [param]
            else:
                cls_parameters += [param]
        
        args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
        args.cls_lr = args.cls_lr if args.cls_lr else args.learning_rate

        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.bert_lr},

            {"params": [p for n, p in cls_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.cls_lr},
            {"params": [p for n, p in cls_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.cls_lr},
        ]
    
    elif args.contrastive_tune:
        bert_parameters = []
        other_parameters = []
        for param in model.named_parameters():
            if "masked_model" in param[0]:
                bert_parameters += [param]
            else:
                other_parameters += [param]

        assert len(bert_parameters) != 0 and len(other_parameters) != 0
        
        args.bert_lr = args.bert_lr if args.bert_lr else args.learning_rate
        args.cls_lr = args.cls_lr if args.cls_lr else args.learning_rate
        
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.bert_lr},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.bert_lr},

            {"params": [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.cls_lr},
            {"params": [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.cls_lr}
        ]
    
    else:
        logger.info("Wrong Model!")
        exit()

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # adversarial_training
    if args.adv_training == 'fgm':
        adv = FGM(model=model, param_name='word_embeddings')
    elif args.adv_training == 'pgd':
        adv = PGD(model=model, param_name='word_embeddings')

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    steps_trained_in_current_epoch = 0
    logging_step = 0
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     try:
    #         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     except ValueError:
    #         global_step = 0
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproductibility
    epoch_num = 1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[2],
                "labels":batch[3],
                "mode":"train"
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[1] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            if not args.fine_tune:
                inputs['prompt_mask'] = batch[4]
            else:
                inputs['prompt_mask'] = None
            loss, _ = model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            if args.adv_training:
                adv.adversarial_training(args, inputs, optimizer)

            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                writer.add_scalar("loss", loss.item(), global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        results, _ = evaluate(args, dev_dataset, model, tokenizer)
                        writer.add_scalar("micro_f1", results['micro_f1'], logging_step)
                        writer.add_scalar("macro_f1", results['macro_f1'], logging_step)
                        writer.add_scalar("eval_loss", results['eval_loss'], logging_step)
                        logging_step += 1
                        if best_score < results['micro_f1']:
                            best_score = results['micro_f1']
                            output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))

                            if not args.fine_tune and not args.contrastive_tune:
                                verbalizer_file = os.path.join(output_dir, 'verbalizer')
                                model.verbalizer.write_verbalizer_to_file(verbalizer_file)

                            logger.info("Saving model checkpoint to %s", output_dir)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        
        # evaluate after every epoch
        if args.evaluate_after_epoch:
            results, result_str = evaluate(args, dev_dataset, model, tokenizer)
            output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
            with open(output_eval_file, "a") as f:
                f.write('***** Predict Result *****\n')
                f.write(result_str)
            writer.add_scalar("micro_f1", results['micro_f1'], epoch_num)
            writer.add_scalar("macro_f1", results['macro_f1'], epoch_num)
            writer.add_scalar("eval_loss", results['eval_loss'], epoch_num)
            if best_score < results['micro_f1']:
                best_score = results['micro_f1']
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                if not args.fine_tune and not args.contrastive_tune:
                    verbalizer_file = os.path.join(output_dir, 'verbalizer')
                    model.verbalizer.write_verbalizer_to_file(verbalizer_file)

                logger.info("Saving model checkpoint to %s", output_dir)

        epoch_num += 1

    return global_step, tr_loss / global_step

def evaluate(args, dev_dataset, model, tokenizer):
    all_preds = np.array([])
    all_labels = np.array([])
    
    # Eval!
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset,
                                 sampler=dev_sampler,
                                 batch_size=args.per_gpu_eval_batch_size)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()
    for batch in tqdm(dev_dataloader, desc='Evaluating'):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[2],
                "labels":batch[3],
                "mode":'test'
            }
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[1] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            if not args.fine_tune:
                inputs['prompt_mask'] = batch[4]
            else:
                inputs['prompt_mask'] = None
            loss, preds = model(**inputs)
            labels = batch[3]
            all_preds = np.concatenate((all_preds, preds.detach().cpu().numpy()), axis=0)
            all_labels = np.concatenate((all_labels, labels.detach().cpu().numpy()), axis=0)
            eval_loss += loss.item()
        nb_eval_steps += 1

    eval_loss /= nb_eval_steps
    micro_f1, macro_f1, f1 = get_acc_f1(all_labels, all_preds, args.num_labels)
    results = {}
    results['eval_loss'] = eval_loss
    results['micro_f1'] = micro_f1
    results['macro_f1'] = macro_f1
    logger.info("***** Eval results  *****")
    result_str = "Eval loss is {}\nMicro f1 is {}\nMacro f1 is {}\n".format(eval_loss, micro_f1, macro_f1)
    for i in range(args.num_labels):
        result_str += "f1 score of Class {} is: {}\n".format(i, f1[i])
    result_str += "\n\n"
    print(result_str)
    return results, result_str

def main():
    args = get_args_parser()

    args.device = torch.device("cuda")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    set_seed(args)
    model_classes = get_model_classes()
    model_config = model_classes[args.model_type]

    tokenizer = model_config['tokenizer'].from_pretrained(
        args.model_name_or_path
    )

    # words_utils = Words_utils(args)

    prompt_util = Prompt_utils(args.prompt_dir, tokenizer, args.text_num)
    prompt_index = args.prompt_index

    logger.info("Loading dataset from run.py...")

    if args.task_name == 'agnews':
        logger.info('Task is AGnews.\n')
        data_processor = AgnewsProcessor(args.data_dir)
    elif args.task_name == 'yahoo':
        logger.info('Task is Yahoo Answers.\n')
        data_processor = YahooAnswersProcessor(args.data_dir)
    elif args.task_name == 'dbpedia':
        logger.info('Task is DBPedia.\n')
        data_processor = DBPediaProcessor(args.data_dir)
    else:
        logger.info('Wrong Task.\n')
        exit()
    args.label2id = data_processor.label_mapping
    args.id2label = data_processor.id2label
    labels = data_processor.labels
    args.num_labels = len(labels)
    

    verbalizer = Verbalizer(labels, tokenizer)

    if args.fine_tune:
        model = FinetuneModel(args)
    elif args.prompt_tune:
        model = PromptTuneModel(args, tokenizer, verbalizer)
    elif args.contrastive_tune:
        model = ContrastiveModel(args, tokenizer)
    else:
        logger.info('Wrong Model.\n')
        exit()
    # prompt = prompt_util.prompts[prompt_index]
    # model.set_prompt_embedding(prompt)
    if args.contrastive_tune and args.using_pretrain:
        checkpoint = args.pretrain_path
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)

    model = model.cuda()
    # for param in model.named_parameters():
    #     if 'label_embedding' in param[0]:
    #         print(param)
    # exit()
    
    # Training
    if args.do_train:
        if not args.fine_tune:
            train_dataset = load_examples(args, data_processor, prompt_util, prompt_index, tokenizer, 'train_sample_' + str(args.sample_num) + '_' + str(args.seed))
            # for evalute during training
            dev_dataset = load_examples(args, data_processor, prompt_util, prompt_index, tokenizer, 'test_sample_100_1')
        else:
            train_dataset = load_examples_4_finetune(args, data_processor, prompt_util, prompt_index, tokenizer, 'train_sample_' + str(args.sample_num) + '_' + str(args.seed))
            # for evalute during training
            dev_dataset = load_examples_4_finetune(args, data_processor, prompt_util, prompt_index, tokenizer, 'test_sample_100_1')
        global_step, tr_loss = train(args, train_dataset, dev_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # model_to_save = (
        #     model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        # model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)
        #
        # # Good practice: save your training arguments together with the trained model
        # torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        output_dir = os.path.join(args.output_dir, "last_checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(model.state_dict(), os.path.join(output_dir, "model"))
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))

        if not args.fine_tune and not args.contrastive_tune:
            verbalizer_file = os.path.join(output_dir, 'verbalizer')
            model.verbalizer.write_verbalizer_to_file(verbalizer_file)
        
        logger.info("Saving model checkpoint to %s", output_dir)
    
    # Evaluation
    if args.do_eval:
        if not args.zero_eval:
            checkpoint = os.path.join(args.output_dir, 'last_checkpoint')
            tokenizer = model_config['tokenizer'].from_pretrained(checkpoint)
            state_dict = torch.load(os.path.join(checkpoint, "model"))
            model.load_state_dict(state_dict)
            if not args.fine_tune and not args.contrastive_tune:
                verbalizer_file = os.path.join(checkpoint, 'verbalizer')
                space = args.space_before_mask
                if args.prompt_tune:
                    direct = True
                else:
                    direct = False
                model.verbalizer.set_verbalizer_from_file(verbalizer_file, direct, space, args.mask_must_lower)

            model.to(args.device)

        if not args.fine_tune:
            dev_dataset = load_examples(args, data_processor, prompt_util, prompt_index, tokenizer, 'test')
        else:
            dev_dataset = load_examples_4_finetune(args, data_processor, prompt_util, prompt_index, tokenizer, 'test')
        results, result_str = evaluate(args, dev_dataset, model, tokenizer)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            f.write('***** Predict Result for {} sample_num {} seed {} prompt index {} *****\n'.format(args.task_name, args.sample_num, args.seed, args.prompt_index))
            f.write(result_str)
    

if __name__ == "__main__":
    main()
