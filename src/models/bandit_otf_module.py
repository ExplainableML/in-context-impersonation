import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import spacy
import torch
from lightning import LightningModule
from torch.nn import ParameterDict
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.chat_gpt import ChatGPT
from src.models.components.vertex_ai import PaLM
from src.models.components.vicuna import VicundaModel

# A logger for this file
log = logging.getLogger(__name__)


class BanditTaskOnTheFlyLitModule(LightningModule):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        llm,
        data_path: str,
        num_classes: int,
        seed: int,
        temperature: float = 1.0,
        characters: list[str] = ["2 year old", "4 year old"],
        impersonation_prompt="If you were a {character}",  # "Imagine you are a {character}"
        template="\n\nQuestion: You are now performing trial {trial}. {impersonation_prompt}, which machine do you choose between machine 1 and machine 2?",
        data_path_root="./results_rebuttal_forceanssys_llama2-70b_tmp",
        storing=True,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.characters = characters
        self.llm = llm
        self.impersonation_prompt = impersonation_prompt
        self.template = template
        self.seed = seed
        self.temperature = temperature
        self.num_classes = num_classes
        self.data_path = data_path
        self.storing = storing
        self.data_path_root = data_path_root
        if not os.path.exists(data_path_root):
            os.makedirs(data_path_root)

        self.llm_start_msg = "Answer: Machine"
        self.none_template = "\n\nQuestion: You are now performing trial {trial}. Which machine do you choose between machine 1 and machine 2?"
        self.instruction = (
            "In this game, you have a choice between two slot machines, represented by machine 1 and machine 2."
            " Choosing the same slot machine will not always give you the same points, but one slot machine is always better than the other. The outcomes will tend to give points around their average value."
            " After making your choice, you will receive feedback about the outcome. Your goal is to choose the slot machine that will give you the most points over the course of {num_trials} trials.\n"
        )
        self.question = (
            "Question: Which machine do you choose between machine 1 and machine 2?"
        )

        if isinstance(self.llm, VicundaModel):
            self.model_type = self.llm.system_prompt
        elif isinstance(self.llm, ChatGPT):
            self.model_type = "chatgpt"
        elif isinstance(self.llm, PaLM):
            self.model_type = "palm"

        if self.model_type != "llama-2":
            self.template += f"\n{self.llm_start_msg}"
            self.none_template += f"\n{self.llm_start_msg}"
            self.question += f"\n{self.llm_start_msg}"

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
                character
                + acc_type: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
                for acc_type in ["_last", "_avg"]
            }
        )
        self.val_accs = ParameterDict(
            {
                character
                + acc_type: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
                for acc_type in ["_last", "_avg"]
            }
        )
        self.test_accs = ParameterDict(
            {
                character
                + acc_type: Accuracy(task="multiclass", num_classes=num_classes)
                for character in characters
                for acc_type in ["_last", "_avg"]
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

    def query_chatgpt(self, prompt, ordered_answers, max_tries=20):
        t = 0
        success = False
        action = -1
        while not success and t < max_tries:
            response = self.llm.generate([prompt])[0]
            success = response in ordered_answers
            t += 1

        if success:
            action = int(response) - 1
        else:
            print("BANDIT TRIAL FAILED!")

        print(success, t)

        return action, success

    def module_step(self, batch: dict, batch_idx: int):
        # one method to do it all
        env = batch["env"][0]

        # obtain ordered list of descriptions
        ordered_answers = [
            self.trainer.datamodule.data_test.idx_to_class[i]
            for i in range(self.num_classes)
        ]

        if not isinstance(self.llm, ChatGPT):
            # Currently assumes answers are single token
            target_tokens = self.llm.tokenizer(ordered_answers, return_tensors="pt")
            # assert target_tokens.input_ids.shape[1] == 2
            if target_tokens.input_ids.shape[1] == 2:
                postfix_token = None
            elif target_tokens.input_ids.shape[1] == 3:
                postfix_token = target_tokens.input_ids[0, 1]
                assert (target_tokens.input_ids[:, 1] == postfix_token).all()
            else:
                raise NotImplementedError
            target_tokens = target_tokens.input_ids[:, -1].to(self.llm.model.device)

        # now we want to run this for each character
        regression_data = {c: defaultdict(list) for c in self.characters}
        return_values = {}
        for character in self.characters:
            env.reset()
            instruction = self.instruction.format(num_trials=env.no_trials)
            impersonation_prompt = self.impersonation_prompt.format(character=character)
            if character == "none":
                prompt = (
                    instruction
                    + "Let's start with the first trial.\n\n"
                    + self.question
                )
            else:
                prompt = (
                    instruction + "Let's start with the first trial.\n\n"
                    f"Question: {impersonation_prompt}, which machine do you choose between machine 1 and machine 2?"
                )
                if self.model_type != "llama-2":
                    prompt += f"\n{self.llm_start_msg}"

            history = (
                "\n\nYou have received the following points when playing in the past:\n"
            )
            history_dict = {0: [], 1: []}

            successful_trial = True
            for trial in range(env.no_trials):
                if trial != 0:
                    history_str = (
                        history
                        + f"- List of points received from machine 1: {history_dict[0]}\n- List of points received from machine 2: {history_dict[1]}"
                    )
                    prompt = (
                        instruction
                        + history_str
                        + (
                            self.none_template.format(trial=trial + 1)
                            if character == "none"
                            else self.template.format(
                                trial=trial + 1,
                                impersonation_prompt=impersonation_prompt,
                            )
                        )
                    )

                if isinstance(self.llm, ChatGPT):
                    action, success = self.query_chatgpt(prompt, ordered_answers)
                    if not success:
                        successful_trial = False
                        break
                else:
                    logits = self.llm.get_logits(
                        [prompt],
                        postfix_token=postfix_token,
                        llm_start_msg=self.llm_start_msg
                        if self.model_type == "llama-2"
                        else None,
                        character=character,
                        change_system_prompt=True,
                    )
                    logits_per_class = logits[:, -1, target_tokens]
                    probs = (logits_per_class / self.temperature).softmax(dim=1)
                    action = torch.multinomial(probs.squeeze(), num_samples=1).item()

                _, reward, done, _ = env.step(action)
                history_dict[action].append(np.round(reward.numpy(), 1))

            if successful_trial:
                # Store data at the end of each game in dataframe:
                regression_data[character]["V"] = env.V
                regression_data[character]["RU"] = env.RU
                regression_data[character]["V/TU"] = list(
                    np.array(env.V) / np.array(env.TU)
                )
                regression_data[character]["action1"] = env.action1
                regression_data[character]["regret"] = env.Regret_list
                regression_data[character]["reward"] = env.reward
                regression_data[character]["mu1"] = [
                    env.mean1.numpy() for _ in range(env.no_trials)
                ]
                regression_data[character]["mu2"] = [
                    env.mean2.numpy() for _ in range(env.no_trials)
                ]
                regression_data[character]["trial"] = list(range(env.no_trials))
                regression_data[character]["prompt"] = [
                    self.impersonation_prompt for _ in range(env.no_trials)
                ]

                return_values[character] = {
                    "loss": np.mean(env.Regret_list),
                    "pred_last": 1 if env.action1 else 2,
                    "pred_avg": 1 if np.mean(env.action1) > 0.5 else 2,
                    "label": 1 if env.mean1 > env.mean2 else 2,
                }

                # Store dataframe in csv file
                if self.storing:
                    data_path = os.path.join(self.data_path_root, f"{character}.csv")
                    reg_data = pd.DataFrame.from_dict(regression_data[character])
                    if os.path.exists(data_path):
                        reg_data.to_csv(
                            data_path, mode="a", header=False, index=False
                        )  # ? If already create
                    else:
                        reg_data.to_csv(data_path, mode="a", index=False)

        return return_values

    def training_step(self, batch: dict, batch_idx: int):
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            pred_last = torch.tensor(results["pred_last"]).unsqueeze(0)
            pred_avg = torch.tensor(results["pred_avg"]).unsqueeze(0)
            label = torch.tensor(results["label"]).unsqueeze(0)

            # update and log metrics
            self.train_losses[character](loss)
            self.train_accs[character + "_last"](pred_last, label)
            self.train_accs[character + "_avg"](pred_avg, label)
            self.log(
                f"train/{character}/loss",
                self.train_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"train/{character}/acc_avg",
                self.train_accs[character + "_avg"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"train/{character}/acc_last",
                self.train_accs[character + "_last"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def validation_step(self, batch: dict, batch_idx: int):
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            pred_last = torch.tensor(results["pred_last"]).unsqueeze(0)
            pred_avg = torch.tensor(results["pred_avg"]).unsqueeze(0)
            label = torch.tensor(results["label"]).unsqueeze(0)

            # update and log metrics
            self.val_losses[character](loss)
            self.val_accs[character + "_last"](pred_last, label)
            self.val_accs[character + "_avg"](pred_avg, label)
            self.log(
                f"val/{character}/loss",
                self.val_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"val/{character}/acc_last",
                self.val_accs[character + "_last"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"val/{character}/acc_avg",
                self.val_accs[character + "_avg"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def test_step(self, batch: dict, batch_idx: int):
        out = self.module_step(batch, batch_idx)

        for character, results in out.items():
            loss = results["loss"]
            pred_last = torch.tensor(results["pred_last"]).unsqueeze(0)
            pred_avg = torch.tensor(results["pred_avg"]).unsqueeze(0)
            label = torch.tensor(results["label"]).unsqueeze(0)

            # update and log metrics
            self.test_losses[character](loss)
            self.test_accs[character + "_last"](pred_last, label)
            self.test_accs[character + "_avg"](pred_avg, label)
            self.log(
                f"test/{character}/loss",
                self.test_losses[character],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"test/{character}/acc_last",
                self.test_accs[character + "_last"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
            self.log(
                f"test/{character}/acc_avg",
                self.test_accs[character + "_avg"],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return out

    def on_validation_epoch_end(self):
        for character in self.characters:
            acc = self.val_accs[character + "_last"].compute()  # get current val acc
            self.val_acc_bests[character](acc)  # update best so far val acc
            # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
            # otherwise metric would be reset by lightning after each epoch
            self.log(
                f"val/{character}/acc_best",
                self.val_acc_bests[character].compute(),
                prog_bar=True,
            )
