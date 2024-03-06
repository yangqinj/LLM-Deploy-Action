# ChatGLM模型

ChatGLM系列模型由清华大学开发的、支持中英文双语问答的对话LLM。目前开源了6b级别的模型，包括[ChatGLLM-6B](https://github.com/THUDM/ChatGLM-6B)、[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)和[ChatGLM3-6B](https://github.com/THUDM/ChatGLM3)三个模型。ChatGLM4-6B并没有开源，但是可以在[智谱清言](https://www.chatglm.cn/)和[API平台](https://open.bigmodel.cn/)体验。



[TOC]

## General Language Model

ChatGLM使用[General Language Model （GLM）](https://github.com/THUDM/GLM)作为backbone，GLM架构结合了BERT的双向注意力和GPT的单项注意力，以期望一个模型既可以完成文本生成类任务（NLG），也可以完成文本理解、文本总结类任务（NLU），因此GLM提出了**自回归式完形填空预训练任务**：



![image-20240208092533234](.\images\image-20240208092533234.png)

+ **自回归式完形填空任务**：从原始的文本中随机采样不同长度的文本得到若干个span，将span的位置使用[M]替换，从而将文本分成Part A和Part B两部分：被MASK掉的文本和随机采样得到的span。对于模型来说，输入是Part A，输出是Part B
+ **2D位置编码**：Part A和Part B两个部分分别编码，其中[M]位置只有一个编码，而不是每一个token一个编码。这种编码的好处是模型不知道被mask掉的span长度，从而生成不定长的文本。
+ **注意力机制**：Part A中使用双向注意力，Part B中使用单向注意力，这样也符合人类做完形填空时的方法，先看完整个句子，再在空格处从左往右填入文本。
+ **两种mask策略**：使用[MASK]替换掉句子中被随机采样的短文本，使用[gMASK]替换掉句子中末尾随机长度的若干长文本，这样就实现了既可以学习双向的上下文信息，也可以完成文本生成类任务。



## GLM-130B

### 模型结构修改

基于GLM架构，清华首先训练了GLM-130B模型。为了平稳有效训练大规模模型和在推理上取得更好的性能，GLM-130B在模型架构和模型训练上做了如下几点调整：

+ **旋转位置编码RoPE**：为了得到更好的外推性，在推理时对于长文本输入仍能保持较好的性能，使用了旋转位置编码（作者对比了RoPE和ALiBi的PPL值，发现ALiBi的提升较小）
+ **DeepNorm**：可以得到更好的训练性能性，梯度更加稳定，避免了Pre-LN中主干值随着模型变深变得异常大的问题（作者设置\alpha=(2N)^\fract{1}/{2}，\beta=(2N)^\fract{-1}/{2}，N表示模型层数)
+ **GeGLU**：使用GeLU作为激活函数的GLU变体



### 数据和预训练任务

+ **自监督填空任务**：1）使用了1.2T英文Pile数据集、1T中文wudao数据集和250G网上爬取的中文数据（包括论坛、维基百科和QA问答数据）2）数据集中30%的数据做[MASK]掩码，70%做[gMask]掩码。前者占每个训练样本的15%，后者采样随机长度进行替换
+ **多任务指令预训练**：为了让模型学习prompt形式的数据，在下游任务中取得更好的结果，引入了多任务指令预训练任务（Multi-Task Instruction Pre-Training, MIP），包含了文本生成、QA、情感分析、句子补充、摘要等任务数据集。



## 训练策略

+ **混合精度**： 在前向传播和后向传播中使用fp16，对于优化器状态和权重使用了fp32。 （结合了DeepNorm增加稳定性；考虑到BF16只能在A100上使用并且会增加~15%的显存消耗，所以没有使用BF16）

+ **嵌入层梯度缩放**：作者在实验中发现，在训练初期嵌入层的梯度数量级比其他层要大很多，则导致loss出现很大的波动，从而使得训练崩溃，所以使用了嵌入层梯度缩放：

  > embedding = embedding * \alpha + embedding.detach() * (1 - \alpha), \alpa=0.1

+ **硬件和并行训练策略**：在96台DGX-A100（8*40G）（总共768张A100GPU）上训练了60天；采用了数据并行、张量模型并行和流水线并行的3D并行策略，流水线并行采用了PipeDream-Flush策略；batch size设置为4224，4路张量模型并行和8路流水线并行；训练过程中硬件FLOPS利用率（HFU）为43.3%，模型FLOPS利用率（MFU）为32.5%



### 推理部署

+ **FasterTransformer(C++实现)：**为了提高推理速度，研究者们花了2个月时间使用FasterTransformer实现了GLM-130B。相比于PyTorch原生实现的BLOOM-176B推理速度提升了7~8.4倍
+ **W8A16**：为了避免全部INT8量化带来了激活值异常问题，研究者使用了W8A16量化方式（权重使用INT8量化，激活值使用FP16）
+ **INT4量化**：为了进一步降低显存消耗，GLM-130B最终采用了INT4量化方法，可以在消费级显卡上推理（ 4 × RTX 3090 Ti (24G) or 8 × RTX 2080 Ti (11G)）



# ChatGLM-6B系列模型

清华开源了GLM的6B和13B两个版本，其中使用较多的还是ChatGLM-6B版本，因此我们具体来看看ChatGLM-6B、ChatGLM2-6B和ChatGLM3-6B三者的区别。



| 模型        | 特点                                                         | 训练数据     | 上下文长度                                                   | 推理加速                                                     | 连接                                         |
| ----------- | ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| ChatGLM-6B  | 中英双语                                                     | ~1T tokens   | 2K                                                           | INT4 量化需 6GB 显存;INT8需8GB；FP16需要13GB                 | [LINK](https://github.com/THUDM/ChatGLM-6B)  |
| ChatGLM2-6B |                                                              | 1.4T tokens  | 使用FlashAttention，将基座模型提升到32K；对话阶段使用8K训练；32K对应的模型为[ChatGLM-6B-32K](https://huggingface.co/THUDM/chatglm2-6b-32k) | 使用MultiQueryAttention提高推理速度和降低显存，推理速度提升42%；INT4量化下，同样6GB显存支持上下文从1K提升到8K | [LINK](https://github.com/THUDM/ChatGLM2-6B) |
| ChatGLM3-6B | 为了统一多个任务的输入，使用了全新的[prompt格式](https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md)；原生支持函数调用、代码执行和Agent等场景 | 更多训练数据 |                                                              |                                                              | [LINK](https://github.com/THUDM/ChatGLM3)    |
| ChatGLM4-6B | 没有开源，但是可以提供了[API平台](https://open.bigmodel.cn/) |              |                                                              |                                                              |                                              |



## 参考资料

### ChatGLM

+ [ChatGLM之GLM-130B开源模型](https://zhuanlan.zhihu.com/p/639772040)

### 旋转位置编码

+ [图解RoPE旋转位置编码及其特性](https://zhuanlan.zhihu.com/p/667864459)
+ [十分钟读懂旋转编码（RoPE）](https://www.zhihu.com/tardis/zm/art/647109286?source_id=1003)
+ [大模型基础组件 - Positaion Encoding](https://blog.nghuyong.top/2023/09/02/NLP/llm-position-embedding/)
+ [关于Transformer中的位置编码-ALiBi](https://zhuanlan.zhihu.com/p/642846676)
+ [Llama也中招，混合精度下位置编码竟有大坑，百川智能给出修复方案](https://www.jiqizhixin.com/articles/2023-08-22-2)
+ [RoPE旋转位置编码深度解析：理论推导、代码实现、长度外推](https://zhuanlan.zhihu.com/p/645263524)
+ [Transformer升级之路：10、RoPE是一种β进制编码](https://spaces.ac.cn/archives/9675)

### DeepNorm

+ [NLP任务中-layer-norm比BatchNorm好在哪里](https://github.com/DA-southampton/NLP_ability/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/Transformer/NLP%E4%BB%BB%E5%8A%A1%E4%B8%AD-layer-norm%E6%AF%94BatchNorm%E5%A5%BD%E5%9C%A8%E5%93%AA%E9%87%8C.md)
+ [如何评价微软亚研院提出的把 Transformer 提升到了 1000 层的 DeepNet？](https://www.zhihu.com/question/519668254/answer/2371885202)

### GeGLU

+ [大模型基础｜激活函数｜从ReLU 到SwiGLU](https://zhuanlan.zhihu.com/p/650237644)
+ [GELU激活函数详解](https://zhuanlan.zhihu.com/p/302394523)
+ [腾大模型|结构组件-2——ReLU、GeLU、SwiGLU、GeGLU](https://zhuanlan.zhihu.com/p/621058772)

### 模型FLOPS利用率和硬件FLOPS利用率

+ [语言模型的训练时间：从估算到 FLOPs 推导](https://zhuanlan.zhihu.com/p/646905171)

### 混合精度训练

+ [LLM大模型之精度问题（FP16，FP32，BF16）详解与实践](https://zhuanlan.zhihu.com/p/657886517)
+ [【PyTorch】唯快不破：基于Apex的混合精度加速](https://zhuanlan.zhihu.com/p/79887894)
+ [浅谈混合精度训练](https://zhuanlan.zhihu.com/p/103685761)

### 并行训练策略

+ [图解大模型训练](https://zhuanlan.zhihu.com/p/613196255)
