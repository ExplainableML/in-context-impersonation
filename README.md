<div align="center">

# In-Context Impersonation Reveals Large Language Models' Strengths and Biases

[![Paper](http://img.shields.io/badge/paper-arxiv.2305.14930-B31B1B.svg)](https://arxiv.org/abs/2305.149309)
[![NeurIPS](http://img.shields.io/badge/NeurIPS_(spotlight)-2023-4b44ce.svg)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e3fe7b34ba4f378df39cb12a97193f41-Abstract-Conference.html)
<br>
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://opensource.org/license/mit)

</div>

## Description

This repository is the official implementation of the **[NeurIPS 2023](https://neurips.cc/) spotlight** _In-Context Impersonation Reveals Large Language Models' Strengths and Biases_ by [Leonard Salewski](https://www.eml-unitue.de/people/leonard-salewski)<sup>1,2</sup>, [Stephan Alaniz](https://www.eml-unitue.de/people/stephan-alaniz)<sup>1,2</sup>, [Isabel Rio-Torto](https://www.eml-unitue.de/people/isabel-rio-torto)<sup>3,4*</sup>, [Eric Schulz](https://www.kyb.tuebingen.mpg.de/person/103915/2537)<sup>2,5</sup> and [Zeynep Akata](https://www.eml-unitue.de/people/zeynep-akata)<sup>1,2</sup>. A preprint is available on [arXiv](https://arxiv.org/abs/2305.14930) and a poster is available on the [NeurIPS website](https://neurips.cc/virtual/2023/poster/72422) and on the [project website](https://www.eml-unitue.de/publications/in-context-impersonation/NeurIPS%202023%20Poster%20In-context%20impersonation%20-%20Draft%203c-1-3.pdf).

<sup>1</sup> [University of Tübingen](https://uni-tuebingen.de/), <sup>2</sup> [Tübingen AI Center](https://tuebingen.ai/), <sup>3</sup>  [University of Porto](https://www.up.pt/portal/en/), <sup>4</sup> [INESC TEC](https://www.inesctec.pt/en), <sup>5</sup> [Max Planck Institute for Biological Cybernetics](https://www.kyb.tuebingen.mpg.de/en)
*Work done while at the University of Tübingen

## 📌 Abstract

![A schematic overview over the three tasks that we evaluated in our paper. For each task (multi-armed bandit, reasoning and vision and language) we show a complete example prompt fed to the large language model as well as example outputs and how they are evaluated.](docs/images/persona_llm.png)

<p align="justify">
In everyday conversations, humans can take on different roles and adapt their vocabulary to their chosen roles. We explore whether LLMs can take on, that is impersonate, different roles when they generate text in-context. We ask LLMs to assume different personas before solving vision and language tasks. We do this by prefixing the prompt with a persona that is associated either with a social identity or domain expertise. In a multi-armed bandit task, we find that LLMs pretending to be children of different ages recover human-like developmental stages of exploration. In a language-based reasoning task, we find that LLMs impersonating domain experts perform better than LLMs impersonating non-domain experts. Finally, we test whether LLMs' impersonations are complementary to visual information when describing different categories. We find that impersonation can improve performance: an LLM prompted to be a bird expert describes birds better than one prompted to be a car expert. However, impersonation can also uncover LLMs' biases: an LLM prompted to be a man describes cars better than one prompted to be a woman. These findings demonstrate that LLMs are capable of taking on diverse roles and that this in-context impersonation can be used to uncover their hidden strengths and biases.
</p>

## 🚀 Installation

### Conda

We exclusively use [conda](https://docs.conda.io/projects/miniconda/en/latest/) to manage all dependencies.

```bash
# clone project
git clone https://github.com/ExplainableML/in-context-impersonation
cd in-context-impersonation

# create conda environment and install dependencies
conda env create -f environment.yaml -n in_context_impersonation

# activate conda environment
conda activate in_context_impersonation

# download models for spacy
python3 -m spacy download en_core_web_sm
```

## ⚡ How to run

Within the paper we show three different impersonation evaluation schemes. To run those first the language models have to be prepared and valid paths need to be configured.

### Configuration

For all experiments [hydra](https://hydra.cc/) is used for configuration. The main config file is ``configs/eval.yaml``. All paths (e.g. for data, model weights, logging, caching, etc.), can be configured in ``configs/paths/default.yaml``.

### Language Model Setup

Use the instructions below to setup the language models. By default the experiments will run with Vicuna. This can be changed by passing ``model.llm=chat_gpt`` to the commands below.

#### Vicuna

For Vicuna please follow the instructions [here](https://github.com/lm-sys/FastChat) to obtain HuggingFace compatible weights.
Afterwards configure the path to the Vicuna weights in ``configs/model/llm/vicuna13b.yaml`` by adjusting the value of the ``model_path`` key.

#### ChatGPT

For ChatGPT please obtain an OpenAI API key, create a ``.env`` file in the project root and insert the key in the following format:

```
OPENAI_API_KEY="some_key"
```

Please note, that calls made to the OpenAI API will incur some costs billed towards your account.

### Experiments

The following commands show how to run the experiments for the three tasks studied in our paper.
Note, that in the code we sometimes use the term ``character`` for ``persona`` interchangeably.

#### Bandit Task

The following command can be used to run the bandit task

```bash
python src/eval.py model=bandit_otf data=bandit
```

which uses `configs/model/bandit_otf.yaml` and `configs/data/bandit.yaml` for further configuration.

#### Reasoning Task

The following command can be used to run one task of the MMLU reasoning experiment

```bash
python src/eval.py model=text_otf data=mmlu data.dataset_partial.task=abstract_algebra
```

which uses `configs/model/text_otf.yaml` and `configs/data/mmlu.yaml` for further configuration.

For other MMLU tasks just replace ```abstract_algebra``` with the desired task name. Task names can be found [here](https://huggingface.co/datasets/tasksource/mmlu).

#### Vision and Language Task

The following command can be used to run one task for the CUB dataset:

```bash
python src/eval.py model=clip_dotf data=cub
```

The following command can be used to run one task for the Stanford Cars dataset:

```bash
python src/eval.py model=clip_dotf data=stanford_cars
```

Further configuration (e.g. the list of personas) can be adjusted in ``configs/model/clip_dotf.yaml``. The datasets can be configured in ``configs/data/cub.yaml``and ``configs/data/stanford_cars.yaml`` respectively.

## 📖 Citation

<!-- TODO: Update bibtex once the official NeurIPS bibtex key is out -->

Please use the following bibtex entry to cite our work:

```bib
@article{Salewski2023InContextIR,
  title   = {In-Context Impersonation Reveals Large Language Models' Strengths and Biases},
  author  = {Leonard Salewski and Stephan Alaniz and Isabel Rio-Torto and Eric Schulz and Zeynep Akata},
  journal = {ArXiv},
  year    = {2023},
  volume  = {abs/2305.14930},
}
```

You can also find our work on [Google Scholar](https://scholar.google.de/citations?view_op=view_citation&hl=de&user=jJz3mXcAAAAJ&citation_for_view=jJz3mXcAAAAJ:qjMakFHDy7sC) and [Semantic Scholar](https://www.semanticscholar.org/paper/In-Context-Impersonation-Reveals-Large-Language-and-Salewski-Alaniz/19c63eade265d8a47d160098d97194b3b83d3770).

## Funding and Acknowledgments

The authors thank [IMPRS-IS](https://imprs.is.mpg.de/) for supporting Leonard Salewski. This work was partially funded by the Portuguese Foundation for Science and Technology (FCT) under PhD grant 2020.07034.BD, the Max Planck Society, the Volkswagen Foundation, the BMBF Tübingen AI Center (FKZ: 01IS18039A), DFG (EXC number 2064/1 – Project number 390727645) and ERC (853489-DEXIM).

This repository is based on the [Lightning-Hydra template](https://github.com/ashleve/lightning-hydra-template).

## Intended Use

The research software in this repository is designed for analyzing the impersonation capabilities of large language models, aiding in understanding their functionality and performance. It is meant to reproduce, understand or modify the insights of the associated paper. The software is not intended for production-ready use and its limitations should be carefully evaluated before using it for such applications.

## License

This repository is licensed under the [MIT License](LICENSE.md).
