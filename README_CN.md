<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  TransNormerLLM -- A Faster and Better LLM
</h1>
</div>

<p align="center">
🤗 <a href="https://huggingface.co/OpenNLPLab/" target="_blank">Hugging Face</a> • 💬 <a href="https://discord.gg/W4Vr7AKW" target="_blank">Discord</a> • 💬 <a href="./images/contact_me_qr.png" target="_blank">微信</a> 
</p>
<div align="center">

[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/OpenNLPLab/TransNormerLLM/blob/main/LICENSE)
<h4 align="center">
    <p>
        <a href="https://github.com/OpenNLPLab/TransNormerLLM/blob/main/README.md">English</a> |
        <b>中文</b> 
    <p>
</h4>
</div>

------
- [入门简介](#入门简介)
- [开源模型](#开源模型)
- [评测结果](#评测结果)
  - [通用领域](#通用领域)
    - [模型结果](#模型结果)
- [推理部署](#推理部署)
  - [Dependency Installation](#dependency-installation)
  - [Notice](#notice)
  - [Python 推理代码](#python-推理代码)
    - [基础模型推理演示](#基础模型推理演示)
- [微调模型](#微调模型)
  - [依赖安装](#依赖安装)
  - [训练](#训练)
- [社区生态](#社区生态)
- [许可声明](#许可声明)
  - [声明](#声明)
  - [协议](#协议)
  - [致谢](#致谢)
  - [引用](#引用)

# 入门简介

我们正在重新定义大型语言模型（LLM）。该代码仓库是[TransNormerLLM](https://arxiv.org/pdf/2307.14995.pdf)的官方实现。 我们的 TransNormerLLM 开放权现在可供个人、创作者、研究人员和各种规模的企业使用，以便他们能够负责任地实验、创新和扩展他们的想法。

我们开放的版本包含 TransNormerLLM 模型实现、开源权重和监督微调 (SFT) 的起始代码。 我们将展示如何加载 [TransNormerLLM](https://arxiv.org/pdf/2307.14995.pdf) 模型、运行 SFT 并对其进行推理的示例。

- TransNormerLLM 是第一个基于线性注意力的 LLM，在准确性和效率方面均优于传统的基于 softmax 注意力的模型。 它是在具有 **1.4 万亿** 的高质量token语料库上进行训练的。
- TransNormerLLM 从之前的线性注意力架构 TransNormer 演变而来，进行了一系列的优化，包括 LRPE 位置嵌入、闪电注意力加速、新的门控和标准化机制。
- TransNormerLLM 在多个广受认可的中文、英文以及多语言通用和特定领域基准测试中取得了同类规模的非常有竞争性的性能。
- 此版本包括具有 **385M**、**1B** 和 **7B** 参数的 **Base** 版本。
- 所有版本均完全开放给学术研究。 开发者只需通过电子邮件申请并获得官方商业许可即可免费商业使用。
- 欲了解更多信息，欢迎阅读我们的学术论文[TransNormerLLM](https://arxiv.org/pdf/2307.14995.pdf)。

![](./images/TransNormerLLM-arch.png)

# 开源模型

具体发布版本及下载链接如下：

|         | 基础模型  | 
|:-------:|:-----------:|
| 385M      | 🤗 [TransNormerLLM-385M](https://huggingface.co/OpenNLPLab/TransNormerLLM-385M) | 
| 1B     | 🤗 [TransNormerLLM-1B](https://huggingface.co/OpenNLPLab/TransNormerLLM-1B) |
| 7B (release soon)     | 🤗 [TransNormerLLM-7B](https://huggingface.co/OpenNLPLab/TransNormerLLM-7B) | 

# 评测结果

为了验证 TransNormerLLM，我们在 Commonsense Reasoning Task、MMLU、CMMLU 和 C-Eval 上测试了 385M、1B 和 7B 模型。 为了进行比较，我们选择了几个开源模型作为比较，包括基于 Transformer 的模型，如 OPT、Pythia、BLOOM、GPT-Neo、GPT-J、MPT、Falcon、LLaMA1/2、OpenLLAMA v1/v2、Baichuan 1/ 2、ChatGLM 1/2，以及非Transformer模型RWKV。 可以看出，与这些模型相比，TransNormerLLM仍然具有很强的竞争力。

**常识推理** 我们报告 BoolQ、PIQA、SIQA、
HellaSwag、WinoGrande、ARC 简单和挑战、OpenBookQA 及其平均值。 我们使用 LM-Eval-Harness 报告所有基准测试的0-shot结果。
与现有最先进的大语言模型相比，我们所有的模型都取得了具有竞争力的表现，展示了理解和应用常识推理的卓越能力。

**汇总基准**
我们报告 MMLU、CMMLU、C-Eval 的总体结果。使用官方脚本来评估 MMLU、CMMLU 和 C-Eval，所有评估结果均采用 5-shot结果。 与业界顶级的开源模型相比，我们的模型在英文和中文基准测试中都表现出了相匹配的性能。


## 通用领域

在通用领域，我们对以下数据集进行了 5-shot 测试：
- [C-Eval](https://cevalbenchmark.com/index.html#home)是一个综合性的中文基础模型评估数据集，涵盖52个学科和四个难度级别。 我们使用该数据集的开发集作为小样本学习的来源，并在测试集上进行测试。 我们的评估方法遵循 [LM-Evaluation-Harness](https://github.com/EleutherAI/lm-evaluation-harness)。
- [MMLU](https://arxiv.org/abs/2009.03300)是一个英语评估数据集，包含57个任务，涵盖小学数学、美国历史、计算机科学、法律等，难度从高中水平到专家水平 。 它是主流的LLM评估数据集。 我们使用其[官方](https://github.com/hendrycks/test)评估方法。
- [CMMLU](https://github.com/haonan-li/CMMLU)是一个涵盖67个主题的综合中文评估基准，专门用于评估语言模型在中文背景下的知识和推理能力。 我们采用了其[官方](https://github.com/haonan-li/CMMLU)评估方法。


### 模型结果

**常识推理和通用领域性能比较。** 为了公平比较，我们报告了我们使用其发布的模型重现的竞争方法的结果。  PS：参数大小（十亿）。 T：Tokens（万亿）。 HS：HellaSwag。 WG：WinoGrande。

| Model       | PS   | T    | BoolQ          | PIQA           | HS             | WG             | ARC-e          | ARC-c          | OBQA           | MMLU           | CMMLU          | C-Eval         |
|-------------|------|------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| OPT         | 0.35 | 0.30 | 57.74          | 64.58          | 36.69          | 52.49          | 44.02          | 23.89          | 28.20          | 26.02          | 25.34          | 25.71          |
| Pythia      | 0.40 | 0.30 | 60.40          | 67.08          | 40.52          | 53.59          | 51.81          | 24.15          | 29.40          | 25.99          | 25.16          | 24.81          |
| BLOOM       | 0.56 | 0.35 | 55.14          | 64.09          | 36.97          | 52.80          | 47.35          | 23.98          | 28.20          | 24.80          | 25.35          | 27.14          |
| RWKV        | 0.43 | -    | -              | 67.52   | 40.90 | 51.14 | 52.86 | 25.17 | 32.40 | 24.85          | -              | -              |
| **Ours**        | 0.39 | 1.0  | 62.14          | 66.70          | 46.27          | 54.46          | 55.43          | 27.99          | 32.40          | 25.90          | 25.05          | 25.24          |
| GPT-Neo     | 1.3  | 0.3  | 61.99          | 71.11          | 48.93          | 54.93          | 56.19          | 25.85          | 33.60          | 24.82          | 26.03          | 23.94          |
| OPT         | 1.3  | 0.3  | 57.77          | 71.71          | 53.70          | 59.35          | 57.24          | 29.69          | 33.20          | 24.96          | 24.97          | 25.32          |
| Pythia      | 1.4  | 0.3  | 60.73          | 70.67          | 47.18          | 53.51          | 56.99          | 26.88          | 31.40          | 26.55          | 25.13          | 24.25          |
| BLOOM       | 1.1  | 0.35 | 59.08          | 67.14          | 42.98          | 54.93          | 51.47          | 25.68          | 29.40          | 27.30          | 25.09          | 26.50          |
| RWKV        | 1.5  | -    | -              | 72.36 | 52.48 | 54.62 | 60.48 | 29.44 | 34.00 | 25.77          | -              | -              |
| Falcon      | 1.0  | 0.35 | 61.38          | 75.14          | 61.50          | 60.30          | 63.38          | 32.17          | 35.60          | 25.28          | 24.88          | 25.66          |
| **Ours**        | 1.0  | 1.2  | 63.27          | 72.09          | 56.49          | 60.38          | 63.68          | 35.24          | 36.60          | 27.10          | 25.88          | 26.01          |
| GPT-J       | 6.9  | 0.3  | 65.44          | 75.41          | 66.25          | 64.09          | 66.92          | 36.60          | 38.20          | 25.40          | 26.47          | 23.39          |
| OPT         | 6.7  | 0.3  | 66.18          | 76.22          | 67.21          | 65.19          | 65.66          | 34.64          | 37.20          | 24.57          | 25.36          | 25.32          |
| Pythia      | 6.9  | 0.3  | 63.46          | 75.14          | 63.92          | 60.77          | 67.34          | 35.41          | 37.00          | 24.64          | 25.56          | 26.40          |
| BLOOM       | 7.1  | 0.35 | 62.91          | 72.69          | 62.33          | 64.01          | 65.11          | 33.45          | 35.80          | 26.25          | 24.97          | 24.25          |
| RWKV        | 7.4  | -    | -              | 76.06 | 65.51 | 61.01 | 67.80 | 37.46 | 40.20 | 24.96          | -              | -              |
| MPT         | 6.9  | 1.0  | 73.88          | 79.43          | 76.25          | 68.27          | 74.79          | 41.72          | 42.20          | 30.80          | 25.99          | 24.06          |
| Falcon      | 7.2  | 1.5  | 73.73          | 79.38          | 76.3           | 67.17          | 74.62          | 43.60          | 43.80          | 27.79          | 25.73          | 22.92          |
| Baichuan1   | 7.0  | 1.2  | 70.09          | 76.01          | 70.06          | 64.09          | 71.72          | 40.53          | 38.20          | 42.30 | 44.43 | 42.80 |
| Baichuan2   | 7.0  | 2.6  | 72.72          | 76.50          | 72.17          | 68.35          | 75.17          | 42.32          | 39.60          | 54.16 | 57.07 | 54.00 |
| ChatGLM1    | 6.7  | 1.0  | 74.74          | 68.88          | 45.57          | 52.25          | 48.78          | 31.66          | 36.80          | 40.63 | 37.48          | 40.23 |
| ChatGLM2    | 7.1  | 1.4  | 77.65          | 69.37          | 50.51          | 57.62          | 59.13          | 34.30          | 37.00          | 45.46 | 48.80          | 52.55 |
| OpenLLaMAv1 | 6.7  | 1.0  | 70.43          | 75.68          | 69.23          | 66.69          | 71.17          | 38.57          | 39.00          | 30.49          | 25.40          | 26.09          |
| OpenLLaMAv2 | 6.7  | 1.0  | 72.20          | 78.84          | 74.51          | 65.67          | 72.39          | 41.30          | 41.00          | 41.29          | 29.58          | 30.01          |
| LLaMA1      | 6.7  | 1.0  | 76.50 | 79.80 | 76.10 | 70.10 | 72.80 | 47.60 | 57.20 | 35.10 | 25.62          | 25.72          |
| LLaMA2      | 6.7  | 2.0  | 77.68 | 78.07 | 76.02 | 68.98 | 76.30 | 46.33 | 44.20 | 45.30 | 32.96          | 33.20          |
| **Ours**        | 6.8  | 1.4  | 75.87          | 80.09          | 75.21          | 66.06          | 75.42          | 44.40          | 63.40          | 43.10          | 47.99          | 43.18          |


# 推理部署

推理所需的模型权重、源代码和配置已在 Hugging Face 上发布。 下载链接可以在本文档开头的[表格](#开源模型)中找到。 下面，我们以 TransNormerLLM-1B 为例演示各种推理方法。 程序会自动从Hugging Face下载所需的资源。
## Dependency Installation

```shell
pip install -r requirements.txt
```

## Notice
If you encounter errors related to Triton, please set the following environment variables:
```
export use_triton=False
```

## Python 推理代码

### 基础模型推理演示

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("OpenNLPLab/TransNormerLLM-1B", trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("OpenNLPLab/TransNormerLLM-1B", device_map="auto", trust_remote_code=True)
```

> 在上面的代码片段中，模型加载指定`device_map='auto'`，它将使用所有可用的GPU。 如果需要指定要使用的设备，可以通过类似于“export CUDA_VISIBLE_DEVICES=0,1”（使用0和1显卡）的方式进行控制。

# 微调模型

## 依赖安装

```shell
git clone https://github.com/OpenNLPLab/TransNormerLLM.git
cd TransNormerLLM/fine-tune
pip install -r requirements.txt
```
- 要使用LoRA等轻量级微调方法，您必须另外安装[peft](https://github.com/huggingface/peft)。

## 训练

下面，我们提供了使用 ZeRO-3 在单台机器上微调 TransNormerLLM-7B-Base 的示例。

训练数据：`alpaca_data.json`。 此示例数据取自 [alpaca_data.json](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json)，包含 52,002 个条目的选择，并已重新格式化。 主要目的是演示如何SFT我们的模型，不保证有效性。

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

# 社区生态

**📢📢📢我们将不断更新这里社区和生态系统对 TransNormerLLM 的支持😀😀😀**
- [nanoTransnormer](https://github.com/Doraemonzzz/nanoTransNormer)

# 许可声明

## 声明


我们特此声明，我们的团队没有开发过任何基于 TransNormerLLM 模型的应用程序，也没有在 iOS、Android、Web 或任何其他平台上开发过。 我们强烈呼吁所有用户不要利用TransNormerLLM模型进行任何危害国家/社会安全或违法的活动。 此外，我们要求用户不要将 TransNormerLLM 模型用于未经过适当安全审查和备案的互联网服务。 我们希望所有用户都能遵守这一原则，确保技术的发展在规范、合法的环境中进行。

我们已尽力确保模型训练过程中使用的数据的合规性。 然而，尽管我们付出了巨大的努力，由于模型和数据的复杂性，仍然可能会出现一些不可预见的问题。 因此，如果因使用TransNormerLLM开源模型而出现任何问题，包括但不限于数据安全问题、舆情风险，或模型被误导、滥用、传播或不当利用带来的任何风险和问题， 我们将不承担任何责任。

## 协议

TransNormerLLM 模型的社区使用需要遵守 [Apache 2.0](https://github.com/OpenNLPLab/TransNormerLLM/blob/main/LICENSE) 和 [TransNormerLLM 模型社区许可证](https://huggingface.co/OpenNLPLab/TransNormerLLM-7B-Base/resolve/main/TransNormerLLM%202%E6%A8%A1%E5%9E%8B%E7%A4%BE%E5%8C%BA%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE.pdf)。  TransNormerLLM 模型支持商业用途。 如果您计划将TransNormerLLM模型或其衍生品用于商业目的，请确保您的实体满足以下条件：

   1. 您或您关联公司的服务或产品的日活跃用户（DAU）低于100万。
   2. 您或您的关联公司都不是软件服务提供商或云服务提供商。
   3. 未经 TransNormerLLM 许可，您或您的关联公司不可能将给予您的商业许可授予或重新授权给其他第三方。

满足上述条件后，您需要通过以下联系邮箱提交TransNormerLLM模型社区许可协议所需的申请材料：opennlplab@gmail.com。 一旦获得批准，TransNormerLLM 将特此授予您非排他性、全球性、不可转让、不可再许可、可撤销的商业版权许可。

## 致谢
我们的项目基于如下开源项目进行开发:
- [Baichuan](https://github.com/baichuan-inc/Baichuan-7B)用于tokenizer部分。
- [metaseq](https://github.com/facebookresearch/metaseq)用于训练部分。
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)用于测评部分。


## 引用

如果您想引用我们的工作，请使用以下参考文献：
```
@article{qin2023scaling,
  title={Scaling transnormer to 175 billion parameters},
  author={Qin, Zhen and Li, Dong and Sun, Weigao and Sun, Weixuan and Shen, Xuyang and Han, Xiaodong and Wei, Yunshen and Lv, Baohong and Yuan, Fei and Luo, Xiao and others},
  journal={arXiv preprint arXiv:2307.14995},
  year={2023}
}
```
