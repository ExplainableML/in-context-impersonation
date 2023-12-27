from typing import Any
from torch.utils.data import Dataset
from datasets import load_dataset


class FGVCAircraft(Dataset):
    def __init__(
        self,
        cache_dir: str,
        train: bool = True,
        num_classes: int = 100,
    ) -> None:
        super().__init__()

        if train:
            self.split = "train"
        else:
            self.split = "test"

        dataset_name = f"Multimodal-Fatima/FGVC_Aircraft_{self.split}"
        self.dataset = load_dataset(
            dataset_name,
            cache_dir=cache_dir,
        )

        self.idx_to_class: dict[int, str] = {
            i: name
            for i, name in enumerate(self.dataset[self.split].features["label"].names)
        }

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, index) -> Any:
        item = self.dataset[self.split][index]

        image = item["image"]
        # Crop away copyrighted information (bottom 20px)
        # See https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/#format
        cropped_image = image.crop((0, 0, image.width, image.height - 20))
        item["image"] = cropped_image

        # this HF version of the dataset comes with a bunch of stuff that we do not need
        item.pop("clip_tags_ViT_L_14", None)
        item.pop("LLM_Description_opt175b_downstream_tasks_ViT_L_14", None)
        item.pop("LLM_Description_gpt3_downstream_tasks_ViT_L_14", None)
        item.pop("LLM_Description_gpt3_downstream_tasks_visual_genome_ViT_L_14", None)
        item.pop("blip_caption", None)
        item.pop("blip_caption_beam_5", None)
        item.pop("Attributes_ViT_L_14_text_davinci_003_full", None)
        item.pop("Attributes_ViT_L_14_text_davinci_003_fgvc", None)
        item.pop("clip_tags_ViT_L_14_with_openai_classes", None)
        item.pop("clip_tags_ViT_L_14_wo_openai_classes", None)
        item.pop("clip_tags_ViT_L_14_simple_specific", None)
        item.pop("clip_tags_ViT_L_14_ensemble_specific", None)
        item.pop("clip_tags_ViT_B_16_simple_specific", None)
        item.pop("clip_tags_ViT_B_16_ensemble_specific", None)
        item.pop("clip_tags_ViT_B_32_simple_specific", None)
        item.pop("clip_tags_ViT_B_32_ensemble_specific", None)
        item.pop("Attributes_ViT_B_16_descriptors_text_davinci_003_full", None)
        item.pop("Attributes_LAION_ViT_H_14_2B_descriptors_text_davinci_003_full", None)
        item.pop("clip_tags_LAION_ViT_H_14_2B_simple_specific", None)
        item.pop("clip_tags_LAION_ViT_H_14_2B_ensemble_specific", None)

        return item


if __name__ == "__main__":
    sc = FGVCAircraft(cache_dir="/mnt/character-based-classification/.cache")
    item1 = sc[0]
    item2 = sc[1]

    # collation
    bar = 9
