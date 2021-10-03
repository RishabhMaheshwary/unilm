#!/usr/bin/env python
# coding=utf-8

import logging
import os
import math
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
import numpy as np
from datasets import ClassLabel, load_dataset, load_metric
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from detectron2.structures import ImageList
from matplotlib.colors import ListedColormap
import layoutlmft.data.datasets.docvqa
import transformers
from layoutlmft import AutoModelForRelationExtraction
from layoutlmft import AutoModelForTokenClassification
#from layoutlm import LayoutLMForTokenClassification
from layoutlmft.data import DataCollatorForKeyValueExtraction
from layoutlmft.data.data_args import DataTrainingArguments
from layoutlmft.models.model_args import ModelArguments
from layoutlmft.trainers import FunsdTrainer as Trainer
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    set_seed,
)
from similarity.normalized_levenshtein import \
    NormalizedLevenshtein

from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from pathlib import Path

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)

EditDistance = NormalizedLevenshtein()

def plot_boxes(inputs, predictions, attentions, test_set, tokenizer, label_list, dir):
    #breakpoint()
    Path(str(dir)+"/attn_images/").mkdir(parents=True, exist_ok=True)
    Path(str(dir)+"/images/").mkdir(parents=True, exist_ok=True)
    #attentions = np.mean(attentions, axis = 0)
    for sample_idx in range(len(inputs["bbox"])):
        bbox = inputs["bbox"][sample_idx]
        input_ids = inputs["input_ids"][sample_idx]
        true_labels = inputs["labels"][sample_idx]
        predicted_labels = predictions[sample_idx]
        cur_idx = 0
        final_boxes, final_true_labels, final_predicted_labels = [], [], []
        attention_indices = []
        for id in input_ids:
            if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]):
                cur_idx+=1
                continue
            else:
                attention_indices.append(cur_idx)
                final_predicted_labels.append(predicted_labels[cur_idx])
                final_boxes.append(bbox[cur_idx])
                final_true_labels.append(true_labels[cur_idx])
            cur_idx+=1
        #print(final_true_labels)
        #print(final_predicted_labels)
        assert len(final_true_labels) == len(final_predicted_labels)

        image = Image.open(inputs["image_paths"][sample_idx])
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)
        #attns = attentions[sample_idx]
        #attns = np.mean(attns, axis=0)

        for i in range(len(final_boxes)):
            if int(final_true_labels[i]) == -100:
                continue

            #image1 = Image.open(inputs["image_paths"][sample_idx])
            #image1 = image1.convert("RGB")
            #draw1 = ImageDraw.Draw(image1)
            #attn_scores = []

            #for j in range(len(attention_indices)):
            #    if i == j:
            #        continue
            #    attn_scores.append((attns[i][attention_indices[j]], final_boxes[j]))
            #attn_scores.sort()
            #attn_scores.reverse()
            #draw1.rectangle(final_boxes[i], outline='green', width=3)

            #for j in range(10):
            #    if j <= 5:
            #        draw1.rectangle(attn_scores[j][1], outline='red', width=3)
            #    else:
            #       draw1.rectangle(attn_scores[j][1], outline='red', width=2)
            #attn_path = str(dir)+"/attn_images/"+str(sample_idx)+"_"+str(i)+".png"
            #image1.save(attn_path)
            if final_true_labels[i] == final_predicted_labels[i]:
                draw.rectangle(final_boxes[i], outline='green', width=3)
            else:
                orig_label = label_list[final_true_labels[i]]
                pred_label = label_list[final_predicted_labels[i]]
                if len(orig_label) > 1:
                    text_mark = orig_label[2]
                else:
                    text_mark = "F"
                if len(pred_label) > 1:
                    text_mark1 = pred_label[2]
                else:
                    text_mark1 = "F"
                #draw.text((final_boxes[i][0], final_boxes[i][1]-15), text_mark, fill="black", font=fnt)
                #draw.text((final_boxes[i][2], final_boxes[i][1]-15), text_mark1, fill="black", font=fnt)
                draw.rectangle([final_boxes[i][0]-10,final_boxes[i][1], final_boxes[i][2]-10, final_boxes[i][3]], outline='red', width=3)

        boxes_path = str(dir)+"/images/"+str(sample_idx)+".png"
        image.save(boxes_path)

def get_attention_scores(attentions, inputs, tokenizer, layer):
    #breakpoint()
    final_euc_dist, final_attn_score, final_far  = [], [], []
    for sample_idx in range(17):
        avg_attn_score = 0
        label_map = {}
        #token_predictions = outputs[sample_idx].logits.argmax(-1).squeeze().tolist() # the predictions are at the token level
        words_indices = []
        labels = inputs["labels"][sample_idx]
        input_ids = inputs["input_ids"][sample_idx]
        bboxes = inputs["bbox"][sample_idx]
        #breakpoint()
        attns = attentions[layer][sample_idx]
        attns = np.mean(attns, axis=0)
        #attns = attns.tolist()
        cur_idx = 0
        final_labels = []
        for id in input_ids:
            if (tokenizer.decode([id]).startswith("##")) or (id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]):
            # skip prediction + bounding box
                #if id == tokenizer.pad_token_id:
                #    print(id)
                continue
            else:
                words_indices.append(cur_idx)
                final_labels.append(labels[cur_idx])
            cur_idx+=1
        attention_scores = []
        x_distance = []
        y_distance = []
        euc_distance = []
        label_pairs = []
        far = []
        total_cnt = 0
        for i in range(512):
            for j in range(i+1, 512):
                if i in words_indices and j in words_indices and final_labels[i] !=-100 and final_labels[j]!=-100:
                    if abs(bboxes[i][0] - bboxes[j][0])<= 50 and abs(bboxes[i][1] - bboxes[j][1]) <= 50:
                        far.append(0)
                    else:
                        far.append(1)
                    attention_scores.append(attns[i][j])
                    total_cnt+=1
                    avg_attn_score += attns[i][j]
                    x_distance.append(abs(bboxes[i][0] - bboxes[j][0]))
                    y_distance.append(abs(bboxes[i][1] - bboxes[j][1]))
                    euc_distance.append(int(math.sqrt((bboxes[i][0]-bboxes[j][0])**2 + (bboxes[i][1] - bboxes[j][1])**2)))
                    label_pairs.append((labels[i],labels[j]))
                    label_map[(labels[i], labels[j])] = 1
        #breakpoint()
        avg_attn_score = avg_attn_score / total_cnt
        #final_euc_dist, final_attn_score, final_far  = [], [], []
        for k in range(len(attention_scores)):
            if attention_scores[k] <= avg_attn_score:
                continue
            final_attn_score.append(attention_scores[k])
            final_euc_dist.append(euc_distance[k])
            final_far.append(far[k])
    classes = ['Near','Far']
    colors = ListedColormap(['r','b'])
    plt.figure()
    scatter = plt.scatter(final_attn_score, final_euc_dist, s = 1, c = final_far, cmap=colors)
    plt.xlabel("Attention Scores")
    plt.ylabel("Euclidean distance")
    plt.title("LayoutLM")
    plt.legend(handles=scatter.legend_elements()[0], labels=classes)
    plt.savefig('plots/plot_all_'+str(layer)+'.jpg')
    plt.close()


def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    print("===========Now Load Dataset==================")
    datasets = load_dataset(os.path.abspath(layoutlmft.data.datasets.docvqa.__file__))

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "tokens" if "tokens" in column_names else column_names[0]
    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        print("==========Tokenization==========")
        tokenized_inputs = tokenizer(
            examples["question"],
            examples[text_column_name],
            padding=padding,
            truncation=True,
            return_overflowing_tokens=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        #breakpoint()
        labels = []
        bboxes = []
        images = []
        img_paths = []
        data_args.label_all_tokens = True
        print(len(tokenized_inputs["input_ids"]))
        cnn = 0
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]
            image = examples["image"][org_batch_index]
            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            cur_cnt, cntr = 0, 0
            for word_idx in word_ids:
                if word_idx is None and cntr > 0:
                    break
                cntr+=1
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if cur_cnt < cntr:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])

                elif word_idx is None:
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    #print(len(label), word_idx)
                    #label_ids.append(-100)
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                previous_word_idx = word_idx
                cur_cnt+=1
            cnn+=1
            print(cnn,end =" ")
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            images.append(image)
            img_paths.append(examples["image_path"][org_batch_index])
        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["image"] = images
        tokenized_inputs["image_paths"] = img_paths
        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        exs = tokenize_and_align_labels(test_dataset)
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        #breakpoint()

    # Data collator
    print("================Done=============")
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    # Metrics
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        #breakpoint()
        results = metric.compute(predictions=true_predictions, references=true_labels)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

    # Initialize our Trainer
    #breakpoint()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        predictions, labels, metrics = trainer.predict(eval_dataset)
        predictions = np.argmax(predictions, axis=2)
        final_indices, true_labels, pred_labels = [], [], []
        #breakpoint()
        exs = eval_dataset["input_ids"]
        anls_score = []
        for i in range(len(predictions)):

            cur_idx = 0
            cur_indices, cur_labels, cur_pred_labels = [], [], []
            toks = exs[i]
            gt_answer, pred_answer = [], []

            for j in range(512):
                cur_token = toks[j]
                if int(labels[i][j]) != -100:
                    #if int(labels[i][j]) != 0:
                    #cur_indices.append(cur_idx)
                    if int(labels[i][j]) >=1:
                        gt_answer.append(tokenizer.decode([cur_token]))
                    if int(predictions[i][j]) >=1:
                        pred_answer.append(tokenizer.decode([cur_token]))

            final_pred_string = []
            for tok in pred_answer:
                if len(tok) > 2 and tok[0:2] == "##":
                    if len(final_pred_string) == 0:
                        final_pred_string.append(tok[2:])
                        continue
                    final_pred_string[-1] = final_pred_string[-1]+tok[2:]
                else:
                    final_pred_string.append(tok)

            final_gt_string = []

            for tok in gt_answer:
                if len(tok) > 2 and tok[0:2] == "##":
                    final_gt_string[-1] = final_gt_string[-1]+tok[2:]
                else:
                    final_gt_string.append(tok)

            sim_score = EditDistance.similarity(" ".join(final_pred_string), " ".join(final_gt_string))
            if sim_score < 0.5:
                sim_score = 0
            print(final_gt_string, final_pred_string, sim_score)
            anls_score.append(sim_score)
                    #cur_labels.append(int(labels[i][j]))
                    #cur_pred_labels.append(int(predictions[i][j]))
                    #cur_idx+=1
            #final_indices.append(cur_indices)
            #true_labels.append(cur_labels)
            #pred_labels.append(cur_pred_labels)
        print("Done")
        print(np.mean(anls_score))
        #for i in range(len(predictions)):
        #    doc_tokens = tokenizer.decode(exs[i])
        #    breakpoint()
            #assert len(doc_tokens) == len(final_indices[i])
        #    for j in range(len(doc_tokens)):
        #        if true_labels[i][j] >= 1:
        #            gt_answer.append(doc_tokens[i][j])
        #        if pred_labels[i][j] >= 1:
        #            pred_answer.append(doc_tokens[i][j])
        #    sim_score = EditDistance.similarity(" ".join(pred_answer), " ".join(gt_answer))
        #    if sim_score < 0.5:
        #        sim_score = 0
        #    anls_scores.append(sim_score)
        #    res.append(pred_answer, gt_answer)
        #print("ANLS score: ", np.mean(anls_scores))
        breakpoint()
        #metrics = trainer.evaluate()

        #max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        #metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        #trainer.log_metrics("eval", metrics)
        #trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        #breakpoint()
        predictions, labels, metrics, hidden_states = trainer.predict(test_dataset)
        cur_preds = predictions
        #breakpoint()
        predictions = np.argmax(predictions, axis=2)
        trainer.log_metrics("test", metrics)
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        orig_predictions = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        #breakpoint()
        #true_preds = []
        #input_ids = exs["input_ids"]
        #cntr = 0
        #docs = []
        #all_hidden_states = []
        #for prediction, label in zip(cur_preds, labels):
        #    arr = []
        #    words = []
        #    h_states = []
        #    input_id = input_ids[cntr]
        #    tok_cntr = 0
        #    for (p, l) in zip(prediction, label):
        #        if l !=-100:
        #            h_states.append(hidden_states[11][cntr][tok_cntr])
        #            words.append(tokenizer.decode([input_id[tok_cntr]]))
        #            arr.append(p)
        #        tok_cntr+=1
        #    cntr+=1
        #    true_preds.append(arr)
        #    docs.append(words)
        #    all_hidden_states.append(h_states)
        #true_preds = [
        #    [p for (p, l) in zip(prediction, label) if l != -100]
        #    for prediction, label in zip(cur_preds, labels)
        #]
        img = []
        for img_path in exs["image_paths"]:
            iddx = img_path.rfind("/")
            img.append(img_path[iddx+1:])
        with open("image_prediction_sequence_train.txt", "w") as ff:
            ff.write("\n".join(img))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

            #with open("original_predictions.txt", "w") as wrt:
            #    for prediction in orig_predictions:
            #        wrt.write(" ".join(prediction) + "\n")

            #np.save("confidence_train.npy", np.asarray(true_preds))
            #np.save("words_train.npy", np.asarray(docs))
            #np.save("all_hidden_states_train.npy", np.asarray(all_hidden_states))
        plot_boxes(exs, predictions, attns ,datasets["test"], tokenizer, label_list, training_args.output_dir)
        #np.save("attention_original/part1.npy", attns)
        #np.save("hidden_states_original/part1.npy", hidden_states)
        #device = torch.device("cpu")
        #model.to(device)
        #model.eval()
        #breakpoint()
        #preds = model(input_ids=torch.tensor(exs["input_ids"], device=device),\
        #             attention_mask=torch.tensor(exs["attention_mask"], device=device),\
        #             token_type_ids=torch.tensor(exs["token_type_ids"], device=device),\
        #             labels=torch.tensor(exs["labels"], device=device), output_attentions=True, output_hidden_states=True)
        #attns, hidden_states = [], []
        #for i in range(12):
            #attns.append(preds.attentions[i].detach().numpy())
            #hidden_states.append(preds.hidden_states[i].detach().numpy())
       #     np.save("attention_pre/test.npy", preds.attentions[i].detach().numpy())
       #     np.save("hidden_states_pre/test.npy", preds.hidden_states[i].detach().numpy())
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #trainer.log_metrics("test", metrics)
        #breakpoint()
        #test_embeddings = []
        #num_categories = 12
        #cmap = cm.get_cmap('tab20')
        #layers = []
        #for i in range(12):
        #    embeddings = preds.hidden_states[i]
        #    embeddings = torch.mean(embeddings, axis = 1)
        #    for j in range(52):
        #        doc_embeddings = embeddings[j]
        #        layers.append(i)
        #        test_embeddings.append(doc_embeddings.detach().numpy())

        #tsne = TSNE(2, verbose=1)
        #fig, ax = plt.subplots(figsize=(8,8))
        #tsne_proj = tsne.fit_transform(test_embeddings)
        #for lab in range(num_categories):
        #    indices = np.asarray(layers) == lab
        #    ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        #ax.legend(fontsize='large', markerscale=2)
        #plt.savefig('plots/test_embeddings_original_1.jpg')
        #plt.close()
        #for i in range(12):
        #get_attention_scores(attns, exs, tokenizer, i)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
