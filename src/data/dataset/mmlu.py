from typing import Any

from datasets import load_dataset
import torch
from torch.utils.data import Dataset

TASKS = [
    "high_school_european_history",
    "business_ethics",
    "clinical_knowledge",
    "medical_genetics",
    "high_school_us_history",
    "high_school_physics",
    "high_school_world_history",
    "virology",
    "high_school_microeconomics",
    "econometrics",
    "college_computer_science",
    "high_school_biology",
    "abstract_algebra",
    "professional_accounting",
    "philosophy",
    "professional_medicine",
    "nutrition",
    "global_facts",
    "machine_learning",
    "security_studies",
    "public_relations",
    "professional_psychology",
    "prehistory",
    "anatomy",
    "human_sexuality",
    "college_medicine",
    "high_school_government_and_politics",
    "college_chemistry",
    "logical_fallacies",
    "high_school_geography",
    "elementary_mathematics",
    "human_aging",
    "college_mathematics",
    "high_school_psychology",
    "formal_logic",
    "high_school_statistics",
    "international_law",
    "high_school_mathematics",
    "high_school_computer_science",
    "conceptual_physics",
    "miscellaneous",
    "high_school_chemistry",
    "marketing",
    "professional_law",
    "management",
    "college_physics",
    "jurisprudence",
    "world_religions",
    "sociology",
    "us_foreign_policy",
    "high_school_macroeconomics",
    "computer_security",
    "moral_scenarios",
    "moral_disputes",
    "electrical_engineering",
    "astronomy",
    "college_biology",
]


class MMLU(Dataset):
    def __init__(
        self,
        task,
        cache_dir: str,
        split: str = "train",
        options: list[str] = ["A", "B", "C", "D"],
        option_separator: str = ":",
        postfix_token: int = None,
        num_classes: int = 4,
    ) -> None:
        super().__init__()

        assert task in TASKS
        assert len(options) == num_classes
        assert split in ["train", "validation", "test"]

        self.task = task
        self.split = split
        self.options = options
        self.option_separator = option_separator
        if postfix_token is not None:
            self.postfix_token = torch.ones((1,), dtype=torch.long) * postfix_token
        else:
            self.postfix_token = postfix_token

        self.dataset = load_dataset(
            "lukaemon/mmlu",
            self.task,
            cache_dir=cache_dir,
        )

        self.idx_to_class: dict[int, str] = {
            i: name for i, name in enumerate(self.options)
        }
        self.class_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(self.options)
        }
        self.target_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(["A", "B", "C", "D"])
        }

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, index) -> Any:
        data = self.dataset[self.split][index]
        text = (
            data["input"]
            + f"\n{self.options[0]}{self.option_separator} "
            + data["A"]
            + f"\n{self.options[1]}{self.option_separator} "
            + data["B"]
            + f"\n{self.options[2]}{self.option_separator} "
            + data["C"]
            + f"\n{self.options[3]}{self.option_separator} "
            + data["D"]
            + "\n"
        )
        return {
            "text": text,
            "label": int(self.target_to_idx[data["target"]]),
            "task": self.task.replace("_", " "),
        }


if __name__ == "__main__":
    import numpy as np

    print(len(TASKS))
    cache_dir = "a/path/to/a/.cache"
    # compute imbalance
    with open("mmlu_stats.txt", "w") as o:
        for t in TASKS:
            print(t)
            sc = MMLU(t, cache_dir=cache_dir, split="test")
            targets = [item["label"] for item in sc]
            values, counts = np.unique(targets, return_counts=True)
            counts = counts * 100.0 / len(targets)
            print(f"Task {t}: {counts}", file=o)

    # compute number of instances in test sets
    max_len = 0
    max_t = None
    min_len = 100000000
    min_t = None
    all_lens = []
    for t in TASKS:
        sc = MMLU(t, cache_dir=cache_dir, split="test")
        all_lens.append(len(sc))
        if len(sc) > max_len:
            max_len = len(sc)
            max_t = t
        if len(sc) < min_len:
            min_len = len(sc)
            min_t = t

    print(f"Max dataset  {max_t} with len {max_len}")
    print(f"Min dataset  {min_t} with len {min_len}")
    with open("mmlu_lens.txt", "w") as o:
        for i, t in enumerate(TASKS):
            print(f"{t}: {all_lens[i]}", file=o)
