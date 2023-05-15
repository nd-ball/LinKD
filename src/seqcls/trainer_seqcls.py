# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A subclass of `Trainer` specific to Question-Answering tasks
"""

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput

import torch
import torch.nn as nn


if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

import mlmt

class SeqClsTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=None,
                ignore_keys=ignore_keys,
        )
        # self.label_names = label_names
        self.compute_metrics = compute_metrics

        # metrics = output.metrics
        metrics = self.compute_metrics(output, eval_dataset)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, predict_dataset, ignore_keys=None, metric_key_prefix: str = "test"):
        predict_dataloader = self.get_test_dataloader(predict_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            predict_dataloader,
            description="Prediction",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
        )

        # self.label_names = label_names
        self.compute_metrics = compute_metrics

        # metrics = output.metrics
        metrics = self.compute_metrics(output, predict_dataset)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics) #Added

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=metrics)

class SeqClsDiffTrainer(SeqClsTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        attn_masks = inputs.get("attention_mask")
        difficulties = torch.sum(attn_masks, -1)
        difficulties = torch.softmax(difficulties, dim=0)
        #print(inputs.keys())
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, labels)
        loss = torch.sum(loss * difficulties) / torch.sum(difficulties)
        return (loss, outputs) if return_outputs else loss



pretrained_model_name = 'bert-base-uncased'
pretrained_model_name = 'michiyasunaga/BioLinkBERT-base'
#pretrained_model_name = 'emilyalsentzer/Bio_ClinicalBERT'
scorer = mlmt.MLMScorer(pretrained_model_name, use_cuda=True)


class CustomPerpTrainer(SeqClsTrainer):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        print(input_ids.shape)
        scores = []
        reconstructed_inputs = []
        for z in input_ids:
            recon = self.tokenizer.decode(z, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #print(recon)
            #reconstructed_inputs.append(recon)
            score = scorer.score_sentences([recon])
            scores.extend(score)    
        #print(reconstructed_inputs)
        #print(scores)
        scores = torch.tensor(scores, device=labels.device)
        difficulties = 1 + torch.exp(scores/128)
        #print(difficulties)

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss 
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(logits, labels)
        loss = torch.sum(loss * difficulties) / torch.sum(difficulties)
        return (loss, outputs) if return_outputs else loss
