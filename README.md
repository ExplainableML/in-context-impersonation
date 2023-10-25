<div align="center">

# In-Context Impersonation Reveals Large Language Models' Strengths and Biases

[![Paper](http://img.shields.io/badge/paper-arxiv.2305.14930-B31B1B.svg)](https://arxiv.org/abs/2305.149309)
[![NeurIPS](http://img.shields.io/badge/NeurIPS_(spotlight)-2023-4b44ce.svg)](https://papers.nips.cc/paper/2030)
<br>
[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

</div>

## Description

This repository is the official implementation of the **NeurIPS 2023 spotlight** _In-Context Impersonation Reveals Large Language Models' Strengths and Biases_ by Leonard Salewski<sup>1,2</sup>, Stephan Alaniz<sup>1,2</sup>, Isabel Rio-Torto<sup>3,4</sup>, Eric Schulz<sup>5</sup> and Zeynep Akata<sup>1,2</sup>. A preprint is available on [arXiv](https://arxiv.org/abs/2305.14930).

<sup>1</sup> University of Tübingen, <sup>2</sup> Tübingen AI Center, <sup>3</sup>  University of Porto, <sup>4</sup> INESC TEC, <sup>5</sup> Max Planck Institute for Biological Cybernetics

## 📌 Abstract

<p align="justify">
In everyday conversations, humans can take on different roles and adapt their vocabulary to their chosen roles. We explore whether LLMs can take on, that is impersonate, different roles when they generate text in-context. We ask LLMs to assume different personas before solving vision and language tasks. We do this by prefixing the prompt with a persona that is associated either with a social identity or domain expertise. In a multi-armed bandit task, we find that LLMs pretending to be children of different ages recover human-like developmental stages of exploration. In a language-based reasoning task, we find that LLMs impersonating domain experts perform better than LLMs impersonating non-domain experts. Finally, we test whether LLMs' impersonations are complementary to visual information when describing different categories. We find that impersonation can improve performance: an LLM prompted to be a bird expert describes birds better than one prompted to be a car expert. However, impersonation can also uncover LLMs' biases: an LLM prompted to be a man describes cars better than one prompted to be a woman. These findings demonstrate that LLMs are capable of taking on diverse roles and that this in-context impersonation can be used to uncover their hidden strengths and biases.
</p>

## Code

Code is coming soon.

## Citation

Please cite our work with the following bibtex key.

```bib
@article{Salewski2023InContextIR,
  title   = {In-Context Impersonation Reveals Large Language Models' Strengths and Biases},
  author  = {Leonard Salewski and Stephan Alaniz and Isabel Rio-Torto and Eric Schulz and Zeynep Akata},
  journal = {ArXiv},
  year    = {2023},
  volume  = {abs/2305.14930},
}
```

## Funding and Acknowledgments

The authors thank IMPRS-IS for supporting Leonard Salewski. This work was partially funded by the Portuguese Foundation for Science and Technology (FCT) under PhD grant 2020.07034.BD, the Max Planck Society, the Volkswagen Foundation, the BMBF Tübingen AI Center (FKZ: 01IS18039A), DFG (EXC number 2064/1 – Project number 390727645) and ERC (853489-DEXIM).

## License

This repository is licensed under the MIT License.
