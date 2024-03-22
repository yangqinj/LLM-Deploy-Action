# ChatGLM-6B模型推理和相关知识点

从ChatGLM-6B模型切入大模型推理加速优化，先看看最简单的用transformers如何加载和使用ChatGLM-6B进行推理，然后再通过阅读源码了解ChatGLM-6B的模型结构和相关设置，接着使用不同的推理框架进行优化，最后学习相关的知识点。

[TOC]

## transformers推理

首先，需要从[huggingface](https://huggingface.co/THUDM/chatglm-6b)上下载对应的模型文件（除了.bin模型文件之外，其他的.py文件和.json配置文件也需要下载）。然后用[ChatGLM-6B提供的示例代码](https://github.com/THUDM/ChatGLM-6B/tree/main#%E4%BB%A3%E7%A0%81%E8%B0%83%E7%94%A8)从本地加载和运行模型即可（建议尚可，但是没有突出女性这个特点 ）：

```python
In [15]: response, history = model.chat(tokenizer, "我是一名女性程序员，请给我一些职业规划建议", history=[])
In [16]: print(response)
作为一名女性程序员，你有很多职业选择和机会。以下是一些职业规划建议：

1. 不断提高自己的技能：作为一名女性程序员，你可以通过不断学习新的技术和工具来提高技能，让自己在行业中保持竞争力。

2. 寻找机会参与开源项目：开源项目是锻炼技能、建立声誉和扩展人脉的好方法。你可以寻找一些适合你技能水平的开源项目，并参与其中。

3. 建立个人品牌：建立个人品牌可以帮助你在行业中树立声誉，并获得更多的机会。你可以创建一些个人网站、博客或者社交媒体账户，展示你的技能和经验。

4. 寻找导师或者 mentor：寻找一位经验丰富的导师或者 mentor 可以帮助你更快地成长和发展职业生涯。他们可以提供宝贵的建议和指导，让你更好地应对职业生涯中的挑战。

5. 尝试担任项目经理或者 CEO：如果你有一定的项目管理和领导经验，可以尝试担任项目经理或者 CEO 等职位，这些职位可以让你在项目管理和领导方面有更多的经验和机会。

6. 寻找机会参加行业活动和会议：参加行业活动和会议可以帮助你与行业内的专业人士建立联系，并了解最新的行业趋势和机会。

7. 寻找机会扩展职业圈子：尝试与行业内的其他人建立联系，参加社交活动和会议，扩展职业圈子，并了解更多的机会和职业发展策略。

最后，记住要坚持不懈地追求自己的职业目标，并不断适应和调整自己的职业规划，以适应职业生涯中的挑战和变化。

```



再来看一下tokenizer输出的`input_ids`结果：

```python
In [31]: result = tokenizer(["你好"], return_tensors="pt")

In [32]: input_ids = result["input_ids"][0]

In [33]: input_ids
Out[33]: tensor([     5,  74874, 130001, 130004])

```



根据配置文件，可以看到输出的最后两个token分别对应'[gMASK]'和'[BOS]'。在学习GLM基础模型结构的时候有提到GLM使用[gMASK]替换句子模型的文本，用来训练文本生成任务，这里的结果符合论文说明。在[gMASK]后接一个[BOS]表示文本序列开始。





## 模型结构和源码

通过打印模型可以看到**模型结构**：

+ **模型的结构**比较简单：一层词嵌入+28层GLM组件+一层lm head。每一个GLM在Attention和MLP中使用Pre-LN，在SelfAttention中使用了旋转位置编码，在MLP中使用了GLU激活函数。这些点都跟我们在前一章《ChatGLM模型》中说的一致
+ **维度设置**：模型使用的词表共有130528个tokens，模型的维度是4096

```
In [18]: model
Out[18]:
ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (word_embeddings): Embedding(130528, 4096)
    (layers): ModuleList(
      (0-27): 28 x GLMBlock(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attention): SelfAttention(
          (rotary_emb): RotaryEmbedding()
          (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
          (dense): Linear(in_features=4096, out_features=4096, bias=True)
        )
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (mlp): GLU(
          (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
          (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
        )
      )
    )
    (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=130528, bias=False)
)

```



接下来看看**模型配置**：

+ **序列最大长度**（max_sequence_length）：2048，是一个现在常见的长度
+ **注意力头个数**（num_attention_heads）：32
+ **位置编码**（position_encoding_2d）： 使用了2D模型位置编码（这里跟GLM-130B的不同，GLLM-130B使用了RoPE）
+ **模型精度**（torch_dtype）：使用了float16
+ **缓存**（use_cache）：使用了KV Cache

```
In [19]: model.config
Out[19]:
ChatGLMConfig {
  "_name_or_path": "<path_to_models>/models/THUDM/chatglm-6b",
  "architectures": [
    "ChatGLMModel"
  ],
  "auto_map": {
    "AutoConfig": "configuration_chatglm.ChatGLMConfig",
    "AutoModel": "modeling_chatglm.ChatGLMForConditionalGeneration",
    "AutoModelForSeq2SeqLM": "modeling_chatglm.ChatGLMForConditionalGeneration"
  },
  "bos_token_id": 130004,
  "eos_token_id": 130005,
  "gmask_token_id": 130001,
  "hidden_size": 4096,
  "inner_hidden_size": 16384,
  "layernorm_epsilon": 1e-05,
  "mask_token_id": 130000,
  "max_sequence_length": 2048,
  "model_type": "chatglm",
  "num_attention_heads": 32,
  "num_layers": 28,
  "pad_token_id": 3,
  "position_encoding_2d": true,
  "pre_seq_len": null,
  "prefix_projection": false,
  "quantization_bit": 0,
  "torch_dtype": "float16",
  "transformers_version": "4.27.1",
  "use_cache": true,
  "vocab_size": 130528
}

```



最后来看看**模型源码实现**：

>  PS1: 根据配置，模型的源码实现在文件`modeling_chatglm.py`中，这是用户自定义的模型结构，，所以在加载模型的时候需要传入参数`trust_remote_code=True`。
>
>  PS2: 只记录关键的实现点，具体的请自己看



+ **prompt格式**：根据组装的prompt来看，当只有单轮对话时，prompt没有特殊格式，就是用户输入的问题；当为多轮对话时，对话历史的格式为：`[Round 0]\n问：这是第一个问题\n答：这是第一个问题的答案\n`

  ```python
  if not history:
      prompt = query
      else:
          prompt = ""
          for i, (old_query, response) in enumerate(history):
              prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
              prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
  ```

+ **tokenizer**：根据GLM-130B论文的说明来看，其基于icetk包实现了分词器，从代码中也可以看出这一点。这个分词器前2w个tokens是用于图片的，但是ChatGLM是一个文本模型，所以其实这2W个token在训练的时候是没有使用的，但是微调和推理的时候还是需要加载这些tokens，并且在计算最后的概率时需要多计算2W个logts，非常的浪费显存。因此有[大神](https://github.com/THUDM/ChatGLM-6B/issues/145)裁掉了这2W个tokens，模型文件减小了大概0.3GB，网友反馈推理速度和轮次确实有提升。

  ```python
  # tokenization_chatglm.py: 176
  vocab_files_names = {"vocab_file": "ice_text.model"}
  ```

  

+ **默认生成参数**：使用了概率为70%的Top-p采样，温度设置为0.95

  ```python
      def chat(self, tokenizer, query: str, history: List[Tuple[str, str]] = None, max_length: int = 2048, num_beams=1, do_sample=True, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
  ```

+ **生成模型输入**：

  1. **tokenizer**: ` tensor([     5,  74874, 130001, 130004]) # ['▁', '你好', '[gMASK]', '<sop>']`；根据配置文件可以知道`<sop>`其实就是bos token (tokenization_chatglm.py: build_inputs_with_special_tokens)

  2. **生成attention_mask**: GLM中给定上下文的tokens相互之间是可以看到的，但是看不到被MASK掉的文本，而被MASK掉的文本可以看到上下文以及当前位置前的生成文本。(modeling_chatglm.py: get_mask)

     ```python
     tensor([[[[False, False, False,  True],
               [False, False, False,  True],
               [False, False, False,  True],
               [False, False, False, False]]]])
     ```

  3. **生成position_ids**: 根据配置文件可以看到在ChatGLM-6B中使用了二维位置编码`"position_encoding_2d": true`

     ```python
     tensor([[[0, 1, 2, 2],
              [0, 0, 0, 1]]])
     ```

  4. **旋转位置编码和自注意力模块**：ChatGLM-6B使用的是不可学习的旋转位置编码

     ```python
     self.rotary_emb = RotaryEmbedding(
                 self.hidden_size // (self.num_attention_heads * 2)
                 if position_encoding_2d # 如果使用了2D位置编码，维度还需要再除以2
                 else self.hidden_size // self.num_attention_heads,
                 base=10000,
                 precision=torch.half, # 半精度
                 learnable=False, # 不学习
             )
     ```

     **需要注意的是：**源码里旋转位置编码部分的代码实现是有问题的，我们来仔细看一下这部分的实现。假设输入序列的长度为3，向量维度是4，batch size为1，现在来计算位置下标为1的token对应的query的位置编码：（旋转位置编码的原理可以参考这篇文章：[十分钟读懂旋转编码（RoPE）](https://zhuanlan.zhihu.com/p/647109286)）

     + 正确的编码 vs 错误的编码

       ```python
       # right
       tensor([[[[-0.8415,  0.5403,  1.9699,  3.0198]]]])
       
       # wrong
       tensor([[[[-1.6829,  0.9700,  1.0806,  3.0098]]]])
       ```

       

     - 首选需要计算`m * theta`，在源码中的实现为：

       ```python
       # batch_size = 1, seq_len = 3, dim = 4, base=10000
       
       inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim)) # ()
       t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
       # 得到的结果对应： m_theta_0, m_theta_1
       freqs = torch.einsum('i,j->ij', t, self.inv_freq)
       ```

     - 接下来，代码拼接freqs并计算正选和余弦值：

       ```python
       # 得到的结果对应： m_theta_0, m_theta_1, m_theta_0, m_theta_1
       emb = torch.cat((freqs, freqs), dim=-1)
       
       cos_cached = emb.cos()[:, None, :]
       sin_cached = emb.sin()[:, None, :]
       cos, sin = cos_cached[:seq_len, ...], sin_cached[:seq_len, ...]
       ```

     - 然后获取目标位置(position_id =1)的正选和余弦值：

       ```python
       # cos_m_theta for token in give position
       position_id = torch.LongTensor([[1]]) # the second position
       cos_q = F.embedding(position_id, cos.squeeze(1)).unsqueeze(2) # [sq, b, 1, hn]
       cos_q, cos_q.shape  # pick up the second index
       sin_q = F.embedding(position_id, sin.squeeze(1)).unsqueeze(2)
       ```

     - 然后计算最终的位置编码：

       ```python
       def rotate_half(x):
           x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
           return torch.cat((-x2, x1), dim=x1.ndim - 1)  # dim=-1 triggers a bug in earlier torch versions
       
       q = torch.arange(dim).view(1, 1, 1, dim) # seq_len, batch_size, heads, dim
       # q_0, q_1, q_2, q_3 变成了 -q_2, -q_4, q_0, q_1
       rh_q = rotate_half(q)
       qm = (q * cos_q) + (rh_q * sin_q)
       ```

     - 到这里就可以看到问题出在q和mtheta的下标没有对上。从数学角度来理解，源代码中实现的是：

       ```python
       [
           q_0 * cos_mtheta_0 - q_2 * sin_mtheta_0,
           q_1 * cos_mtheta_1 - q_3 * sin_mtheta_1,
           q_2 * cos_mtheta_0 + q_0 * sin_mtheta_0,
           q_3 * cos_mtheta_1 + q_1 * sin_mtheta_1
       ]
       ```

       但是正确的计算方式应该是：

       ```python
       [
           q_0 * cos_mtheta_0 - q_1 * sin_mtheta_0,
           q_1 * cos_mtheta_0 + q_0 * sin_mtheta_0,
           q_2 * cos_mtheta_1 - q_3 * sin_mtheta_1,
           q_3 * cos_mtheta_1 + q_2 * sin_mtheta_1
       ]
       ```

     - 这个问题在ChatGLM2-6B中已经被修复，具体的实现可以去看一下[源码实现](https://huggingface.co/THUDM/chatglm2-6b/blob/main/modeling_chatglm.py#L161)，测试的结果可以参考一下[test_rotary_embedding.ipynb]()

     由于在ChatGLM-6B中使用的2D位置编码，所以在计算位置编码时将维度除以了2，再分别计算q1和q2，然后进行拼接得到q。这里也可以看一下[源码](https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L457)

  2. **INT4/8量化**

     - 现在下载的模型使用FP16精度的权重，但是ChatGLM-6B实现了对权重的INT4/INT8量化，包括每一层的：Wquery, Wkey, Wvalue, Wdense_z, Wh_to_4h, W4h_to_h；

     - **INT8**：模型实现的是[per-tensor粒度的最大绝对值量化方法](https://huggingface.co/THUDM/chatglm-6b/blob/main/quantization.py#L134)

       ```pyt
       self.weight_scale = (weight_tensor.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
       self.weight = torch.round(weight_tensor / self.weight_scale[:, None]).to(torch.int8)
       ```

     - **INT4**：使用了[cpm_kernels](https://github.com/OpenBMB/cpm_kernels/tree/master)中的int4WeightCompression方法，不过在源代码里面并没有找到int4WeightCompression这个函数的实现，所以不太清楚具体的怎么做的。

     

## 参考

+ [Top-k & Top-p, Temperature](https://zhuanlan.zhihu.com/p/613428710)
+ [为什么int8的取值范围是-128 - 127](https://blog.csdn.net/ordmeng/article/details/99620804)
+ [LLM 量化技术小结](https://zhuanlan.zhihu.com/p/651874446)
+ [详解 QLoRA 原理 （附源码剖析）](https://zhuanlan.zhihu.com/p/638927564)
+ [GPTQ: 模型量化，穷鬼救星](https://github.com/IST-DASLab/gptq/blob/main/llama.py)
+ 


## 参考

+ [Top-k & Top-p, Temperature](https://zhuanlan.zhihu.com/p/613428710)
