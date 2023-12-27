import logging
from typing import Any

import spacy
import torch
from lightning import LightningModule
from torch.nn import ParameterDict
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.chat_gpt import ChatGPT
from src.models.components.vertex_ai import PaLM

# A logger for this file
log = logging.getLogger(__name__)


class LanguageTaskOnTheFlyLitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        llm,
        data_path: str,
        num_classes: int,
        seed: int,
        characters: list[str] = ["2 year old", "4 year old"],
        template="""If you were a {character}, would you answer the following question with A, B, or C?
{context}""",
        max_tries: int = 10,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.characters = characters
        self.llm = llm
        self.template = template
        self.seed = seed
        self.num_classes = num_classes
        self.data_path = data_path
        self.max_tries = max_tries

        log.info(f"Template: {self.template}")

        log.info("load spacy for some sentence cleaning")
        self.nlp = spacy.load("en_core_web_sm")

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["llm", "model"])

        # metrics:
        # metric objects for calculating and averaging accuracy across batches
        # Wrapping them with ParameterDict's nicely handles all the moving to devices, etc.
        self.train_accs = ParameterDict(
            {
                character: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
            }
        )
        self.val_accs = ParameterDict(
            {
                character: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
            }
        )
        self.test_accs = ParameterDict(
            {
                character: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
            }
        )

        # for averaging loss across batches
        self.train_losses = ParameterDict(
            {character: MeanMetric() for character in characters}
        )
        self.val_losses = ParameterDict(
            {character: MeanMetric() for character in characters}
        )
        self.test_losses = ParameterDict(
            {character: MeanMetric() for character in characters}
        )

        # for tracking best so far validation accuracy
        self.val_acc_bests = ParameterDict(
            {character: MaxMetric() for character in characters}
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def module_step(self, batch: dict, batch_idx: int):
        if isinstance(self.llm, ChatGPT) or isinstance(self.llm, PaLM):
            return self.module_step_chatgpt(batch, batch_idx)

        # one method to do it all
        text = batch["text"]
        label = batch["label"]
        task = list(set(batch["task"]))
        assert len(task) == 1
        task = task[0]

        # obtain ordered list of descriptions
        ordered_answers = [
            self.trainer.datamodule.data_test.idx_to_class[i]
            for i in range(self.num_classes)
        ]
        target_tokens = self.llm.tokenizer(ordered_answers, return_tensors="pt")
        target_tokens = target_tokens.input_ids[:, -1].to(label.device)

        # now we want to run this for each character
        return_values = {}
        for character in self.characters:
            prompts = [
                self.template.format(character=character, context=t, task=task)
                for t in text
            ]
            if (
                hasattr(self.trainer.datamodule.data_test, "postfix_token")
                and self.trainer.datamodule.data_test.postfix_token is not None
            ):
                logits = self.llm.get_logits(
                    prompts,
                    postfix_token=self.trainer.datamodule.data_test.postfix_token,
                )
            else:
                logits = self.llm.get_logits(prompts)
            logits_per_class = logits[:, -1, target_tokens]

            # we can take the softmax to get the label probabilities
            probs = logits_per_class.softmax(dim=1)
            pred_classes = probs.argmax(dim=1)

            # compute the loss and return it
            loss = self.criterion(probs, label)

            return_values[character] = {
                "loss": loss,
                "probs": probs,
                "pred_classes": pred_classes,
            }

        return return_values

    def query_chatgpt(self, prompt, ordered_answers):
        t = 0
        sucess = False
        response = None
        while not sucess and t < self.max_tries:
            responses = self.llm.generate([prompt])
            response = responses[0]
            sucess = response in ordered_answers
            t += 1

        return response, sucess

    def module_step_chatgpt(self, batch: dict, batch_idx: int):
        text = batch["text"]
        label = batch["label"]
        task = list(set(batch["task"]))
        assert len(task) == 1
        task = task[0]

        # obtain ordered list of descriptions
        ordered_answers = [
            self.trainer.datamodule.data_test.idx_to_class[i]
            for i in range(self.num_classes)
        ]
        # now we want to run this for each character
        return_values = {}
        for character in self.characters:
            discarded = 0
            prompts = [
                self.template.format(character=character, context=t, task=task)
                for t in text
            ]
            responses = self.llm.generate(prompts)

            filtered_responses = []
            filtered_labels = []
            for i in range(len(responses)):
                if responses[i] not in ordered_answers:
                    new_response, sucess = self.query_chatgpt(
                        prompts[i], ordered_answers
                    )
                    if sucess:
                        filtered_responses.append(new_response)
                        filtered_labels.append(label[i])
                    else:
                        print(f"Sample {i} in batch {batch_idx} discarded")
                        discarded += 1
                else:
                    filtered_responses.append(responses[i])
                    filtered_labels.append(label[i])

            return_values[character] = {
                "pred_classes": torch.tensor(
                    [
                        self.trainer.datamodule.data_test.class_to_idx[r]
                        for r in filtered_responses
                    ],
                    dtype=torch.long,
                ).to(label.device),
                "labels": torch.stack(filtered_labels),
            }
            print(f"Discarded {discarded} samples for character {character}")
        return return_values

    def training_step(self, batch: dict, batch_idx: int):
        label = batch["label"]
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            probs = results["probs"]

            # update and log metrics
            self.train_losses[character](loss)
            self.train_accs[character](probs, label)
            self.log(
                f"train/{self.trainer.datamodule.data_test.task}/{character}/loss",
                self.train_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"train/{self.trainer.datamodule.data_test.task}/{character}/acc",
                self.train_accs[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def validation_step(self, batch: dict, batch_idx: int):
        label = batch["label"]
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            probs = results["probs"]
            pred_classes = results["pred_classes"]

            # update and log metrics
            self.val_losses[character](loss)
            self.val_accs[character](probs, label)
            self.log(
                f"val/{self.trainer.datamodule.data_test.task}/{character}/loss",
                self.val_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"val/{self.trainer.datamodule.data_test.task}/{character}/acc",
                self.val_accs[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def test_step(self, batch: dict, batch_idx: int):
        label = batch["label"]
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            if isinstance(self.llm, ChatGPT) or isinstance(self.llm, PaLM):
                pred_classes = results["pred_classes"]
                label = results["labels"]
                self.test_accs[character](pred_classes, label)
            else:
                loss = results["loss"]
                probs = results["probs"]
                pred_classes = results["pred_classes"]

                self.test_losses[character](loss)
                self.test_accs[character](probs, label)

                self.log(
                    f"test/{self.trainer.datamodule.data_test.task}/{character}/loss",
                    self.test_losses[character],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )
            self.log(
                f"test/{self.trainer.datamodule.data_test.task}/{character}/acc",
                self.test_accs[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def on_validation_epoch_end(self):
        for character in self.characters:
            acc = self.val_accs[character].compute()  # get current val acc
            self.val_acc_bests[character](acc)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(
                f"val/{self.trainer.datamodule.data_test.task}/{character}/acc_best",
                self.val_acc_bests[character].compute(),
                prog_bar=True,
            )
