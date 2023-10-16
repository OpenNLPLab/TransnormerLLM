<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  TransNormerLLM -- A Faster and Better LLM
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/OpenNLPLab/" target="_blank">Hugging Face</a> • 💬 <a href="https://discord.gg/W4Vr7AKW" target="_blank">Discord</a> 
</p>

<div align="center">

[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/OpenNLPLab/TransNormerLLM/blob/main/LICENSE)
<h4 align="center">
    <p>
        <b>English</b> |
        <a href="https://github.com/OpenNLPLab/TransNormerLLM/blob/main/README_CN.md">中文</a>
    <p>
</h4>
</div>


------
- [Introduction](#introduction)
- [Released Weights](#released-weights)
- [Benchmark Results](#benchmark-results)
  - [General Domain](#general-domain)
    - [Model Results](#model-results)
- [Inference and Deployment](#inference-and-deployment)
  - [Dependency Installation](#dependency-installation)
  - [Notice](#notice)
  - [Python Code Inference](#python-code-inference)
    - [Demonstration of Base Model Inference](#demonstration-of-base-model-inference)
- [Fine-tuning the Model](#fine-tuning-the-model)
  - [Dependency Installation](#dependency-installation-1)
  - [Training](#training)
- [Community and Ecosystem](#community-and-ecosystem)
- [Disclaimer, License and Citation](#disclaimer-license-and-citation)
  - [Disclaimer](#disclaimer)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)
  - [Citation](#citation)

# Introduction

We are re-inventing the Large Language Model (LLM). This is the official implementation of [TransNormerLLM](https://arxiv.org/pdf/2307.14995.pdf). Our opened weights of TransNormerLLM are now accessible to individuals, creators, researchers and businesses of all sizes so that they can experiment, innovate and scale their ideas responsibly.

Our release contains the TransNormerLLM model implementation, the open-source weights and the starting code for Supervised Fine-tuning (SFT). We will show examples on how to load [TransNormerLLM](https://arxiv.org/pdf/2307.14995.pdf) models, run SFT and inference on it.

- TransNormerLLM is the first linear attention-based LLM that outperforms conventional softmax attention-based models in terms of both accuracy and efficiency. It was trained on a high-quality corpus with up to **1.4 trillion** tokens.
- TransNormerLLM evolves from the previous linear attention architecture TransNormer by making advanced modifications that include LRPE positional embedding, Lightning Attention acceleration, new gating and normalization mechanisms.
- TransNormerLLM achieved competitive performance of its size on multiple well-approved Chinese, English, and multi-language general and domain-specific benchmarks.
- This release includes **Base** versions with **385M**, **1B**, and **7B** parameters.
- All versions are fully open to academic research. Developers only need to apply via email and obtain official commercial permission to use it for free commercially.
- For more information, welcome reading our academic paper [TransNormerLLM](https://arxiv.org/pdf/2307.14995.pdf).

![](./images/TransNormerLLM-arch.png)

# Released Weights

The specific released versions and download links are shown as below:

|                   |                                  Base Models                                   |
| :---------------: | :----------------------------------------------------------------------------: |
|       385M        | 🤗 [TransNormerLLM-385M](https://huggingface.co/OpenNLPLab/TransNormerLLM-385M) |
|        1B         |   🤗 [TransNormerLLM-1B](https://huggingface.co/OpenNLPLab/TransNormerLLM-1B)   |
| 7B (release soon) |   🤗 [TransNormerLLM-7B](https://huggingface.co/OpenNLPLab/TransNormerLLM-7B)   |

# Benchmark Results

To validate TransNormerLLM, we tested our 385M, 1B, and 7B models on Commonsense Reasoning Task, MMLU, CMMLU, and C-Eval. For comparison, we selected several open-source models as competitors, including Transformer-based models such as OPT, Pythia, BLOOM, GPT-Neo, GPT-J, MPT, Falcon, LLaMA1/2, OpenLLAMA v1/v2, Baichuan 1/2, ChatGLM 1/2, and non-Transformer model RWKV. It can be observed that, compared to these models, TransNormerLLM remains highly competitive.

**Commonsense Reasoning** We report BoolQ, PIQA, SIQA,
HellaSwag, WinoGrande, ARC easy and challenge, OpenBookQA and their average. We report 0-shot results for all benchmarks using LM-Eval-Harness.
All of our models achieve competitive performance compared to existing state-of-the-art LLMs, showcasing a remarkable ability to comprehend and apply commonsense reasoning.

**Aggregated Benchmarks**
We report the overall results for MMLU, CMMLU, C-Eval. Official scripts were used for evaluating MMLU, CMMLU, and C-Eval, with all evaluation results being conducted with a 5-shot setup. In comparison to top-tier open-source models available in the industry, our models have demonstrated matched performance in both English and Chinese benchmarks.

## General Domain

In the general domain, we conducted 5-shot tests on the following datasets:
- [C-Eval](https://cevalbenchmark.com/index.html#home) is a comprehensive Chinese basic model evaluation dataset, covering 52 disciplines and four levels of difficulty. Our evaluation approach followed that of [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness).
- [MMLU](https://arxiv.org/abs/2009.03300) is an English evaluation dataset comprising 57 tasks, encompassing elementary math, American history, computer science, law, etc. The difficulty ranges from high school level to expert level. It's a mainstream LLM evaluation dataset. We used its [official](https://github.com/hendrycks/test) evaluation approach.
- [CMMLU](https://github.com/haonan-li/CMMLU) is a comprehensive Chinese evaluation benchmark covering 67 topics, specifically designed to assess language models' knowledge and reasoning capabilities in a Chinese context. We adopted its [official](https://github.com/haonan-li/CMMLU) evaluation approach.


### Model Results
**Performance Comparison on Commonsense Reasoning and Aggregated Benchmarks.** For a fair comparison, we report competing methods' results reproduced by us using their released models. PS: parameter size (billion). T: tokens (trillion). HS: HellaSwag. WG: WinoGrande.

| Model       | PS   | T    | BoolQ | PIQA  | HS    | WG    | ARC-e | ARC-c | OBQA  | MMLU  | CMMLU | C-Eval |
| ----------- | ---- | ---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ |
| OPT         | 0.35 | 0.30 | 57.74 | 64.58 | 36.69 | 52.49 | 44.02 | 23.89 | 28.20 | 26.02 | 25.34 | 25.71  |
| Pythia      | 0.40 | 0.30 | 60.40 | 67.08 | 40.52 | 53.59 | 51.81 | 24.15 | 29.40 | 25.99 | 25.16 | 24.81  |
| BLOOM       | 0.56 | 0.35 | 55.14 | 64.09 | 36.97 | 52.80 | 47.35 | 23.98 | 28.20 | 24.80 | 25.35 | 27.14  |
| RWKV        | 0.43 | -    | -     | 67.52 | 40.90 | 51.14 | 52.86 | 25.17 | 32.40 | 24.85 | -     | -      |
| **Ours**    | 0.39 | 1.0  | 62.14 | 66.70 | 46.27 | 54.46 | 55.43 | 27.99 | 32.40 | 25.90 | 25.05 | 25.24  |
| GPT-Neo     | 1.3  | 0.3  | 61.99 | 71.11 | 48.93 | 54.93 | 56.19 | 25.85 | 33.60 | 24.82 | 26.03 | 23.94  |
| OPT         | 1.3  | 0.3  | 57.77 | 71.71 | 53.70 | 59.35 | 57.24 | 29.69 | 33.20 | 24.96 | 24.97 | 25.32  |
| Pythia      | 1.4  | 0.3  | 60.73 | 70.67 | 47.18 | 53.51 | 56.99 | 26.88 | 31.40 | 26.55 | 25.13 | 24.25  |
| BLOOM       | 1.1  | 0.35 | 59.08 | 67.14 | 42.98 | 54.93 | 51.47 | 25.68 | 29.40 | 27.30 | 25.09 | 26.50  |
| RWKV        | 1.5  | -    | -     | 72.36 | 52.48 | 54.62 | 60.48 | 29.44 | 34.00 | 25.77 | -     | -      |
| Falcon      | 1.0  | 0.35 | 61.38 | 75.14 | 61.50 | 60.30 | 63.38 | 32.17 | 35.60 | 25.28 | 24.88 | 25.66  |
| **Ours**    | 1.0  | 1.2  | 63.27 | 72.09 | 56.49 | 60.38 | 63.68 | 35.24 | 36.60 | 27.10 | 25.88 | 26.01  |
| GPT-J       | 6.9  | 0.3  | 65.44 | 75.41 | 66.25 | 64.09 | 66.92 | 36.60 | 38.20 | 25.40 | 26.47 | 23.39  |
| OPT         | 6.7  | 0.3  | 66.18 | 76.22 | 67.21 | 65.19 | 65.66 | 34.64 | 37.20 | 24.57 | 25.36 | 25.32  |
| Pythia      | 6.9  | 0.3  | 63.46 | 75.14 | 63.92 | 60.77 | 67.34 | 35.41 | 37.00 | 24.64 | 25.56 | 26.40  |
| BLOOM       | 7.1  | 0.35 | 62.91 | 72.69 | 62.33 | 64.01 | 65.11 | 33.45 | 35.80 | 26.25 | 24.97 | 24.25  |
| RWKV        | 7.4  | -    | -     | 76.06 | 65.51 | 61.01 | 67.80 | 37.46 | 40.20 | 24.96 | -     | -      |
| MPT         | 6.9  | 1.0  | 73.88 | 79.43 | 76.25 | 68.27 | 74.79 | 41.72 | 42.20 | 30.80 | 25.99 | 24.06  |
| Falcon      | 7.2  | 1.5  | 73.73 | 79.38 | 76.3  | 67.17 | 74.62 | 43.60 | 43.80 | 27.79 | 25.73 | 22.92  |
| Baichuan1   | 7.0  | 1.2  | 70.09 | 76.01 | 70.06 | 64.09 | 71.72 | 40.53 | 38.20 | 42.30 | 44.43 | 42.80  |
| Baichuan2   | 7.0  | 2.6  | 72.72 | 76.50 | 72.17 | 68.35 | 75.17 | 42.32 | 39.60 | 54.16 | 57.07 | 54.00  |
| ChatGLM1    | 6.7  | 1.0  | 74.74 | 68.88 | 45.57 | 52.25 | 48.78 | 31.66 | 36.80 | 40.63 | 37.48 | 40.23  |
| ChatGLM2    | 7.1  | 1.4  | 77.65 | 69.37 | 50.51 | 57.62 | 59.13 | 34.30 | 37.00 | 45.46 | 48.80 | 52.55  |
| OpenLLaMAv1 | 6.7  | 1.0  | 70.43 | 75.68 | 69.23 | 66.69 | 71.17 | 38.57 | 39.00 | 30.49 | 25.40 | 26.09  |
| OpenLLaMAv2 | 6.7  | 1.0  | 72.20 | 78.84 | 74.51 | 65.67 | 72.39 | 41.30 | 41.00 | 41.29 | 29.58 | 30.01  |
| LLaMA1      | 6.7  | 1.0  | 76.50 | 79.80 | 76.10 | 70.10 | 72.80 | 47.60 | 57.20 | 35.10 | 25.62 | 25.72  |
| LLaMA2      | 6.7  | 2.0  | 77.68 | 78.07 | 76.02 | 68.98 | 76.30 | 46.33 | 44.20 | 45.30 | 32.96 | 33.20  |
| **Ours**    | 6.8  | 1.4  | 75.87 | 80.09 | 75.21 | 66.06 | 75.42 | 44.40 | 63.40 | 43.10 | 47.99 | 43.18  |


# Inference and Deployment

The model weights, source code, and configuration needed for inference have been released on Hugging Face. Download links can be found in the table at the beginning of this document. Below, we demonstrate various inference methods using TransNormerLLM-7B-Chat as an example. The program will automatically download the required resources from Hugging Face.

## Dependency Installation

```shell
pip install -r requirements.txt
```

## Notice
If you encounter errors related to Triton, please set the following environment variables:
```
export use_triton=False
```


## Python Code Inference

### Demonstration of Base Model Inference

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("OpenNLPLab/TransNormerLLM-1B", trust_remote_code=True)
```

> In the above code snippets, the model loading specifies `device_map='auto'`, which will use all available GPUs. If you need to specify the device(s) to use, you can control it in a way similar to `export CUDA_VISIBLE_DEVICES=0,1` (using the 0 and 1 graphics cards).


# Fine-tuning the Model

## Dependency Installation

```shell
git clone https://github.com/OpenNLPLab/TransNormerLLM.git
cd TransNormerLLM/fine-tune
pip install -r requirements.txt
```
- To use lightweight fine-tuning methods like LoRA, you must additionally install [peft](https://github.com/huggingface/peft).

## Training

Below, we provide an example of fine-tuning the TransNormerLLM-1B on a single machine with ZeRO-3.

Training Data: `alpaca_data.json`. This sample data was drawn from [alpaca_data.json](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json), consisting of a selection of 52,002 entries, and has been reformatted. The main purpose is to demonstrate how to SFT our model, and effectiveness is not guaranteed.

```shell
torchrun \
    --nproc_per_node=8 \
    train.py \
    --model_name_or_path OpenNLPLab/TransNormerLLM-1B \
    --data_path ./alpaca_data.json \
    --output_dir output \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 true \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --deepspeed 'configs/zero3.json' \
    --logging_steps 1 \
    --dataloader_num_workers 24 \
    --ddp_find_unused_parameters false \
    --tf32 true \
```

# Community and Ecosystem

**📢📢📢 We will continuously update the support for TransNormerLLM from the community and ecosystem here 😀😀😀**
- [nanoTransnormer](https://github.com/Doraemonzzz/nanoTransNormer)

# Disclaimer, License and Citation

## Disclaimer
We hereby declare that our team has not developed any applications based on TransNormerLLM models, not on iOS, Android, the web, or any other platform. We strongly call on all users not to use TransNormerLLM models for any activities that harm national / social security or violate the law. Also, we ask users not to use TransNormerLLM models for Internet services that have not undergone appropriate security reviews and filings. We hope that all users can abide by this principle and ensure that the development of technology proceeds in a regulated and legal environment.

We have done our best to ensure the compliance of the data used in the model training process. However, despite our considerable efforts, there may still be some unforeseeable issues due to the complexity of the model and data. Therefore, if any problems arise due to the use of TransNormerLLM open-source models, including but not limited to data security issues, public opinion risks, or any risks and problems brought about by the model being misled, abused, spread or improperly exploited, we will not assume any responsibility.

## License
The community usage of TransNormerLLM model requires adherence to [Apache 2.0](https://github.com/OpenNLPLab/TransNormerLLM/blob/main/LICENSE) and [Community License for TransNormerLLM Model](https://huggingface.co/OpenNLPLab/TransNormerLLM-7B-Base/resolve/main/TransNormerLLM%202%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf). The TransNormerLLM model supports commercial use. If you plan to use the TransNormerLLM model or its derivatives for commercial purposes, please ensure that your entity meets the following conditions:

  1. The Daily Active Users (DAU) of your or your affiliate's service or product is less than 1 million.
  2. Neither you nor your affiliates are software service providers or cloud service providers.
  3. There is no possibility for you or your affiliates to grant the commercial license given to you, to reauthorize it to other third parties without TransNormerLLM's permission.

Upon meeting the above conditions, you need to submit the application materials required by the TransNormerLLM Model Community License Agreement via the following contact email: opennlplab@gmail.com. Once approved, TransNormerLLM will hereby grant you a non-exclusive, global, non-transferable, non-sublicensable, revocable commercial copyright license.

## Acknowledgments
Our project is developed based on the following open source projects:
- [Baichuan](https://github.com/baichuan-inc/Baichuan-7B) for the tokenizer.
- [metaseq](https://github.com/facebookresearch/metaseq) for training.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluation.

## Citation
If you wish to cite our work, please use the following reference:
```
@article{qin2023scaling,
  title={Scaling transnormer to 175 billion parameters},
  author={Qin, Zhen and Li, Dong and Sun, Weigao and Sun, Weixuan and Shen, Xuyang and Han, Xiaodong and Wei, Yunshen and Lv, Baohong and Yuan, Fei and Luo, Xiao and others},
  journal={arXiv preprint arXiv:2307.14995},
  year={2023}
}
```
