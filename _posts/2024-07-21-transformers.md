---
layout: distill
title: Transformers
description: Notes on the Transformer architecture.
tags: study nlp
giscus_comments: false
date: 2024-07-21
featured: true

authors:
  - name: João Leite
    url: "https://jaleite.com"
    affiliations:
      name: University of Sheffield

bibliography: 2024-07-21-transformers.bib

toc:
  - name: Introduction
  - name: Motivation
  - name: Components
    subsections:
      - name: Multi-Head Self-Attention
      - name: Word Embeddings and Positional Encodings  
      - name: Feed Forward Network
      - name: Batch Normalisation
      - name: Residual Connections
  - name: Implementation
  - name: Variations
    subsections:
      - name: Encoder-Decoder
      - name: Encoder-Only
      - name: Decoder-Only

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

# Introduction

The transformer is a sequence-to-sequence neural architecture that revolutionized natural language processing by efficiently learning contextual representations through parallelized attention mechanisms. Originally introduced for machine translation in the seminal paper "Attention is All you Need" by Vaswani et al. (2017) <d-cite key="vaswani2017attention"></d-cite>, transformers have since become the foundation for most modern NLP systems.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/transformer-block.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 1:</b> Overview of the Transformer Architecture 
    (<a href="https://pub.aimind.so/summary-of-transformer-achitecture-c2cef6dcaca6">Source</a>).
</div>


Transformers process sequential data (e.g., language), and excel at tasks where understanding the relationships between elements in the sequence is important. For example:
- **Machine Translation:** Automatically converts text from one language to another.
    > **Input:** English sentence  
    > **Output:** Same sentence in French

- **Text Summarization:** Produces a concise summary from a longer piece of text.
    > **Input:** Long article  
    > **Output:** Short summary

- **Text Generation:** Generates new text based on a given prompt or partial sentence.
    > **Input:** Prompt or partial sentence  
    > **Output:** Continuation or full sentence

- **Question Answering:** Finds or generates answers to questions based on a provided passage.
    > **Input:** Passage and a question  
    > **Output:** Answer extracted or generated from the passage

- **Text Classification:** Assigns predefined categories or labels to a given text based on its content.
    > **Input:** Product review  
    > **Output:** Sentiment label (e.g., positive, negative)


# Motivation
The motivation behind the transformer architecture stems from the limitations of earlier sequence-to-sequence models based on Recurrent Neural Networks (RNNs).

RNNs encode contextual information from previous time steps into "state vectors" that are updated at each step. This approach leads to the state vector being significantly altered with each new input, causing the model to quickly lose information about earlier context in long sequences.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/rnn.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 2:</b> Recurrent Neural Network (RNN): At each time step $t$, the RNN receives an input token $x_t$ along with a state vector $h_{t-1}$, and produces an output token $o_t$ as well as an updated state vector $h_t$. This recurrent process allows the model to maintain contextual information across the sequence. Source: Raj, Dinesh Samuel Sathia et al. <d-cite key="raj-2019-analysis"></d-cite>
</div>

Other variations of RNNs attempt to mitigate this issue. For example, Gated Recurrent Units (GRUs) <d-cite key="chung-etal-2014-empirical"></d-cite> use gating mechanisms to control the flow of information by deciding when and how much to update the state vector at each time step. This enables the model to retain relevant context over longer sequences by allowing it to ignore less important states.

<div class="row mt-3" style="max-width: 50%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/gru.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 3:</b> Gated Recurrent Units (GRUs): The update gate $z$ determines how much of the previous hidden state $h_{t-1}$ is carried forward versus how much is updated with the new candidate hidden state $\tilde{h}_t$. The reset gate $r$ controls how much of the previous hidden state is ignored when computing the candidate hidden state. Source: Cho, Kyunghyun, et al. <d-cite key="cho-etal-2014-learning"></d-cite>
</div>
Although these mechanisms help, the fundamental architectural issues associated with sequential processing still persist:

- **Information loss over long contexts:** For long sequences, the state vector gradually loses information from earlier time steps, which are overwritten by information from later time steps.
- **Compute inefficiency due to non-parallelizable operations:** To generate a token $o_t$ at time step $t$, all previous inputs $x_i$ for $i < t$ must be processed sequentially. This lack of parallelism makes computation prohibitively inefficient when training on extremely large datasets.

# Core Idea
The core component powering the transformer architecture is the **Self-Attention (SA)** mechanism. Instead of relying on a state vector that is updated step by step, self-attention allows every token in a sequence to directly "attend" to all other tokens at once. This means that, for any token $x_t$, the model can consider the context of all tokens $x_i$ (for all $i$) simultaneously, rather than sequentially. This overcomes the information loss over long contexts that occurs in RNNs.


<div class="row mt-3" style="max-width: 50%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/token-attention.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 4:</b> The token "it" attends to all other tokens in the sequence simultaneously. Attenion scores between the token "it" and the tokens "The", "monkey", "that", "banana", "it" are high. Source: Huiqiang Xie, Zhijin Qin, Geoffrey Li (2021). <d-cite key="xie_deep_learning_semantic"></d-cite>
</div>

Additionally, because attention is computed in parallel across the entire sequence using highly efficient matrix operations, modern hardware such as GPUs and TPUs can be leveraged to greatly accelerate both training and inference, enabling training larger models on bigger datasets.

However, this design shift in the architecture from RNNs to self-attention, does not come only with benefits. The self-attention mechanism has important drawbacks in comparison to RNNs:
- <b>Quadratic computational complexity</b>: For a sequence of length $N$, this requires computing an $N{\times}N$ attention matrix. This leads to a computational and memory complexity of $O(N^2)$, i.e., doubling the input length quadruples the necessary computation. In contrast, RNNs have a linear complexity of $O(N)$, i.e., doubling the input length also doubles the necessary computation. This quadratic scaling makes processing very long sequences expensive.
- <b>Fixed-size sequence length</b>: A direct consequence of the quadratic complexity is that Self-Attention must operate on a fixed-length context window. All input sequences are truncated or padded to a predefined maximum length (e.g., 512 tokens). This prevents the model from processing sequences of arbitrary length, a task that RNNs can handle naturally due to their step-by-step processing.

The Self-Attention mechanism is the core of the Transformer architecture, but other components are essential for it to function in practice. Next, we will discuss each component in greater detail.

# Components

## Word Embeddings and Positional Encodings

Unstructured inputs such as text or images must be converted into numerical representations before they can be processed by machine learning models. These numerical representations should encode information about the input in such a way that, for example, vectors representing two synonymous words are close together in the embedding space. There are several methods to learn such vectors. For example, Mikolov et al. (2013) <d-cite key="mikolov-2013-efficient"></d-cite> introduced the continuous bag-of-words and continuous skip-gram models, which learn dense, low-dimensional vectors that capture the semantic meaning of words.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/word-embeddings.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 5:</b> Word Embeddings (<a href="https://arize.com/blog-course/embeddings-meaning-examples-and-how-to-compute/">Source</a>).
</div>

With transformers, input tokens are mapped into dense vectors using an **embedding layer**. The embedding matrix is initialized randomly with shape (<i>vocabulary_size</i>, <i>embedding_dimension</i>) and is learned jointly with the rest of the model during training. Each input token retrieves its corresponding embedding from this table, providing a continuous vector representation that captures semantic properties of the token.

Along with word embeddings, the transformer architecture introduces a second type of embedding called **positional encodings**. This is necessary because the self-attention mechanism is invariant to the order of tokens in the input sequence. Since self-attention processes all tokens in parallel, it treats the input as a set rather than a sequence. Without positional information, the model cannot distinguish between the first, second, or any other position in the sequence, and thus cannot capture the sequential structure of language. 

To address this, we introduce positional encodings that explicitly encode the order of tokens. There are two main approaches to computing positional encodings: using sinusoidal functions or learning the positional encodings jointly with the model. Each method has its own advantages and drawbacks, and the choice often depends on the specific application and requirements.

- **Sinusoidal positional encodings** (as introduced in the original Transformer paper) use fixed sine and cosine functions of different frequencies to encode each position. This approach allows the model to extrapolate to sequence lengths not seen during training and provides a deterministic, non-learnable way to represent position.
- **Learned positional encodings** treat the positional embeddings as parameters that are learned during training, similar to word embeddings. This approach allows the model to adapt the positional information to the specific dataset and task, potentially improving performance, but may not generalize as well to longer sequences.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/positional-embeddings.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 9:</b> Sinusoidal positional encodings (<a href="https://towardsdatascience.com/understanding-positional-encoding-in-transformers-dc6bafc021ab">Source</a>).
</div>

The final input to the transformer block is the element-wise sum of the word embeddings and the positional encodings, allowing the model to capture both semantic meaning and positional information.

Note that the input to the embedding layer is a sequence of token ids, and the output is a matrix of size (<i>context_window</i>, <i>embedding_dimension</i>), where each row represents an input token, and each column represents a feature.

### Implementation
The implementation below considers the approach of jointly learning the positional encodings alongside the rest of the model components.

```python
class EmbeddingEncoder(nn.Module):
    def __init__(self, dim_embedding, vocab_size, context_size):
        super().__init__()
        self.embedding_table = nn.Embedding(
            vocab_size,
            dim_embedding
        )
        self.positional_embedding_table = nn.Embedding(
            context_size,
            dim_embedding
        )
        self.context_size = context_size

    def forward(self, x):
        x_emb = self.embedding_table(x)
        pos_emb = self.positional_embedding_table(
            torch.arange(
                self.block_size,
                device=device
              )
            )
        
        return x_emb + pos_emb
```

## Self-Attention (SA)

As discussed previously, the Self-Attention mechanism enables each token in the input sequence to attend to every other token, allowing the model to capture long-range dependencies (see **Figure 4**).

The goal of self-attention is to learn the relative importance between all pairs of tokens in the sequence. To achieve this, the model computes "attention scores" in the form of a square matrix of size $N{\times}N$, where $N$ is the sequence length, and each entry in the matrix represents how much attention token $i$ should pay to token $j$.

<div class="row mt-3" style="max-width: 50%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/attention-matrix.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 6</b>: Attention scores for an input sequence "Hello I love you". (<a href="https://theaisummer.com/attention/">Source</a>)
</div>

With this goal in mind, let's take a step back to understand how to frame this problem in such a way that the model can learn how to produce the attention scores. We can represent each token in the sequence using three distinct feature vectors: queries ($Q$), keys ($K$), and values ($V$). Intuitively, the kind of information that we want to encode into these vectors are:

- **Query:** What features am I searching for in other tokens? (What am I looking for?)
- **Key:** What features do I present to other tokens, so they can determine if I am relevant to them? (How do I describe myself to others?)
- **Value:** What information do I carry that should be shared if another token attends to me? (What do I represent?)

In practice, we learn these features in the following manner: we randomly initialize three weight matrices: Key ($W_K$), Query ($W_Q$), and Value ($W_V$) matrices. Each of these matrices will multiply the input sentence embedding matrix to obtain the keys, queries, and values $Q$, $K$, and $V$, respectively.

<div class="row mt-3" style="max-width: 80%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/attention-query-key-value.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 7:</b> Keys, Queries, and Values: Three randomly initialized matrices multiply the input embedding to obtain the Keys, Queries, and Values for that particular input sequence (<a href="https://epichka.com/blog/2023/qkv-transformer/">Source</a>).
</div>

Now to obtain the attention scores, we apply the following equation:

$$
\begin{equation}\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V\end{equation} 
$$

Let's break the equation down step-by-step. 

### Dot Product ($QK^T$)
Let's ignore the term $\sqrt{d_k}$ for now. Remember the intuitive meaning of $Q$ (What am I looking for?) and $K$ (How do I describe myself to others?). When we perform a dot-product between $Q$ and $K^T$, we are essentially computing a similarity score between the two features. If ${Q_i}{\cdot}{K_j}^T$ is high, it means that token $j$ has the kind of features that token $i$ is looking for. Let's illustrate this with an example.

Imagine we have three tokens from the sentence "The cats are chasing the mouse": **chasing**, **cat**, and **mouse**. For the self-attention mechanism, each token generates Query, Key, and Value vectors. Here, we'll focus on how **chasing** computes attention scores with respect to **cat** and **mouse**.

Let's assume the following Query and Key vectors have been generated:

* $Q_{chasing} = [-2.0, \ 3.0, \ 2.5, \ -1.0, \ 1.5, \ -2.0]$
* $K_{cat} = [-1.8, \ 2.8, \ 3.0, \ 0.2, \ 2.5, \ -1.5]$
* $K_{mouse} = [-1.5, \ -2.0, \ 2.8, \ -0.5, \ -2.0, \ 3.0]$

---

#### 1. Score: `chasing` → `cat`

We compute the dot product of the query from `chasing` and the key from `cat`.

$Q_{chasing} \cdot K_{cat} = [-2.0, 3.0, 2.5, -1.0, 1.5, -2.0] \cdot [-1.8, 2.8, 3.0, 0.2, 2.5, -1.5]$

The step-by-step multiplication of corresponding elements looks like this:

* $(-2.0 \times -1.8) = 3.6$ (Match ✅)
* $(3.0 \times 2.8) = 8.4$ (Strong Match ✅)
* $(2.5 \times 3.0) = 7.5$ (Strong Match ✅)
* $(-1.0 \times 0.2) = -0.2$ (Minor Mismatch ❌)
* $(1.5 \times 2.5) = 3.75$ (Match ✅)
* $(-2.0 \times -1.5) = 3.0$ (Match ✅)

The final similarity score is the sum of these products:
$$3.6 + 8.4 + 7.5 - 0.2 + 3.75 + 3.0 = \textbf{26.05}$$

---

#### 2. Score: `chasing` → `mouse`

Next, we compute the dot product of the query from `chasing` and the key from `mouse`.

$Q_{chasing} \cdot K_{mouse} = [-2.0, 3.0, 2.5, -1.0, 1.5, -2.0] \cdot [-1.5, -2.0, 2.8, -0.5, -2.0, 3.0]$

The step-by-step multiplication reveals significant disagreement:

* $(-2.0 \times -1.5) = 3.0$ (Match ✅)
* $(3.0 \times -2.0) = -6.0$ (Strong Mismatch ❌)
* $(2.5 \times 2.8) = 7.0$ (Strong Match ✅)
* $(-1.0 \times -0.5) = 0.5$ (Minor Match ✅)
* $(1.5 \times -2.0) = -3.0$ (Mismatch ❌)
* $(-2.0 \times 3.0) = -6.0$ (Strong Mismatch ❌)

The final similarity score reflects this poor alignment:
$$3.0 - 6.0 + 7.0 + 0.5 - 3.0 - 6.0 = \textbf{-4.5}$$

The high positive score for **cat** and the negative score for **mouse** indicate that "cat" is far more relevant to "chasing" in this context. Let's continue on to the next part of the equation.


### Softmax

The Softmax function takes the unbounded $(-\inf, +\inf)$ scores obtained from the dot product, and turns them into a probability distribution ($[0, 1]$, summing to $1$). This gives us the actual attention scores (see **Figure 6**).

<div class="row mt-3" style="max-width: 80%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/attention-softmax.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 8:</b> Softmax (<a href="https://www.mdpi.com/2076-3417/13/23/12784">Source</a>).
</div>

Note that the softmax functions does exactly what its name suggests: a soft version of a max function. With a max function, we would get a $1$ for the largest element in the sequence, and $0$ for everything else. With softmax, we want the highest values to accumulate most of the probability distribution, with smaller values being close to $0$.

### The Scaling Factor ($\sqrt{d_k}$)

The reason to introduce the scaling factor $\sqrt{d_k}$ is because of how the dot product and the softmax function behave. Dot products can grow large, especially with high-dimensional vectors. Large inputs can push the Softmax function into regions where its gradient is almost zero, which makes training unstable. Scaling mitigates this issue by ensuring the input to the softmax function is not extremely large.

### Output Contextual Vectors (Weighted Sum with $V$)

So far, we've calculated who each token should pay attention to. The attention weights tell us the importance of other tokens. Now, we need to use these weights to combine the substance of those tokens. While the Query and Key vectors are used to establish relationships, the Value vector $V$ contains the actual information of a token that we want to aggregate.

The final step is to multiply our attention weights by their corresponding Value vectors and sum them up. This is a **weighted average**.

$$Z_i = \sum_{j} \text{Attention}(i, j) \cdot V_j$$

Note that $Z_i$, the final output vector for token $i$, is composed of all the $V_j$ vectors for every token $j$ in the sequence, weighted by how much they are important to token $i$.

### (Optional) Self-Attention as a Soft Hash Table
If the attention mechanism is already clear by this point, this section can be skipped.


Another good example to build intuition for how the Self-Attention mechanism works is to compare it with the concept of a lookup table / hash table / dictionary.

In a python dictionary, a query will either match entirely with an existing key (and then return it's value), or it will not match at all with any of the keys (and then return an error or a predefined value):

```python
d = {
    "dog": "bark",
    "wolf": "howl",
    "cat": "meow"
}

# this key exists, the query will match and return "bark"
query1 = "dog" 
value1 = lookup[query1]

print(value1)
### Output: "bark"

# this query doesnt match with any existing key, and we will get an error
query2 =  "bird"
value2 = lookup[query2]

print(value2)
### Output: Exception
```

In the attention mechanism, we are performing a similar operation, but we implement a "soft" match of keys and queries. In fact, all queries will match will all keys, but in different intensities. The softmax computation introduces this notion of intensity, which will weigh the value that is returned.

```python
d = {
    "dog": "bark":,
    "wolf": "howl",
    "cat": "meow"
}

# query doesn't exist in the dictionary
query = "husky"

# but we can compute its similarity with the existing keys
# assume that the 'similarity' and 'softmax' functions exist.
similarities = [similarity(key, query) for key in d.keys()]
attention_scores = [softmax(s/sqrt(d_k)) for s in similarities]
contextual_vector = [
    attn_score * value for attn_score, value in zip(attention_scores, d.values())
]

# example:
# similarities = [4.0, 3.0, -1.0]
# attention_scores = [0.70, 0.26, 0.04]  
# contextual_vector = [0.70 * "bark", 0.26 * "howl", 0.04 * "meow"]

# this now represents the token that is associated with the query "husky"
```

### Masked Self-Attention
Another important consideration is that the self-attention mechanism is slightly different in the encoder and in the decoder:
- In the encoder, the SA allows tokens to attend to each other bi-directionally, meaning a token $x_i$ will attend both to previous and to following tokens.
- In the decoder, the SA **does not** allow tokens to attend to future tokens. A token $x_i$ is only allowed to attend to tokens $x_j$ with $j \leq i$.
- The two are calculated in the exact same way. However, for the decoder, we apply a mask to remove the influence of tokens from future time steps (i.e., set the values to $-\infty$).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/attention-mask.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 6:</b> Attention Matrix Masking (<a href="https://krypticmouse.hashnode.dev/attention-is-all-you-need">Source</a>).
</div>



Finally, the last missing component for this mechanism is the composition of multiple SA units into a single **Multi-Head Self-Attention (MHA)** unit.
- Instead of having a single SA unit process the entire input vector, we define multiple SA units, and each unit will compute a piece of the vector.
- For example, if we have a context size of $32$ and embeddings of dimensions $64$, a single self-attention head would output a matrix of dimension ($32 \times 64$).
- Instead, we could define a MHA with $4$ heads, therefore each head will compute a matrix ($32 \times 16$). We then concatenate these matrixes over the embedding dimension, reconstructing the original dimension of $64$.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/attention-mha.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 7:</b> Multi-Head Self-Attention (MHA): Input sequence "I am a student" is processed through $4$ SA units with <i>embedding_dim</i> equals $2$. The attention values $a_0, a_1, a_2, a_3$ are concatenated to form the output with <i>embedding_dim</i> equals $8$ (<a href="https://velog.io/@joongwon00/딥러닝을-이용한-단백질-구조-예측-1.ProteinStructurePrediction">Source</a>).
</div>

**Motivation:** a single attention mechanism might struggle to capture all the intricate patterns in complex sequences due to its limited capacity. Multi-head attention mitigates this by distributing the learning task across multiple heads, reducing the risk of bottlenecks where certain crucial information might be missed or underrepresented. In other words, each head will learn a different latent space, and encode its specialized features into a segment of the final vector. Hopefully each of these segments will capture specialized characteristics of the data, as opposed to having a single vector space encoding everything as in the single SA unit.

### Implementation

```python
class HeadAttention(nn.Module):
    def __init__(
        self,
        block_size,
        dim_embedding,
        dim_head
      ):
        super().__init__()

        self.dim_head = dim_head
        self.dim_embedding = dim_embedding
        
        # It is conventional to remove the bias for these
        self.key = nn.Linear(
            dim_embedding,
            dim_head,
            bias=False
          )
        self.query = nn.Linear(
            dim_embedding,
            dim_head,
            bias=False
          )
        self.value = nn.Linear(
            dim_embedding,
            dim_head,
            bias=False
          )
        
        # a register_buffer saves this variable in the model
        # params, but they are not considered trainable
        # parameters. can access the variable with self.mask
        self.register_buffer(
            "mask", torch.tril(
                torch.ones(block_size, block_size)
          ))
        
    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        prod = q @ k.transpose(-2, -1) * self.dim_head**-0.5
        prod = prod.masked_fill(self.mask == 0, float("-inf"))
        prod = F.softmax(prod, dim=-1)
        
        out = prod @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        block_size,
        num_heads,
        dim_head,
        dim_embedding
      ):
        super().__init__()

        self.head_list = nn.ModuleList(
            [
                HeadAttention(
                    block_size,
                    dim_embedding,
                    dim_head,
                  ) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(dim_embedding, dim_embedding)

    def forward(self, x):
        out = torch.cat(
            [head(x) for head in self.head_list],
            dim=-1
          )
        out = self.proj(out)
        
        return out
```

## Feed Forward / Multi-layer Perceptron (MLP)

Up to now, all the operations performed are linear transformations: matrix multiplications, softmax, addition. To add non-linearity, we simply put a feed forward network with a non-linear activation after the MHA layer.

This is usually a very shallow network with only $1$ hidden layer with a ReLU / GeLU / etc. activation function. It is also common to do an up projection and a down projection in the hidden layer, meaning the input gets stretched to a higher dimensional space when entering the hidden unit, then gets down projected to the original size after leaving the hidden unit.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/mlp.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 10:</b> Multi-Layer Perceptron: Input gets up projected from $3$ to $4$ dimensions, then down projected back from $4$ to $3$ dimensions. Source: Garg, Siddhant & Ramakrishnan, Goutham. (2020) <d-cite key="garg-ramakrishnan-2020-advances"></d-cite>.
</div>

### Implementation

```python
class MLP(nn.Module):
    def __init__(
        self,
        dim_head,
        dim_embedding,
      ):
        super().__init__()
        # increase the dimension by 4
        # then bring it back to the original size
        self.net = nn.Sequential(
            nn.Linear(dim_head, 4*dim_embedding),
            nn.ReLU(),
            nn.Linear(4*dim_embedding, dim_embedding)
        )

    def forward(self, x):
        out = self.net(x)

        return out
```

## Batch Normalisation

Batch normalisation is a procedure that helps stabilise and accelerate the training process by reducing internal covariate shift, which refers to the change in the distribution of layer inputs during training. It consists in normalising the outputs of the layers so that their distribution have a mean of $0$ and a variance of $1$, then applying two learned parameters to scale and shift the distribution. This reduces the chance that gradients explode or vanish (grow or shrink exponentially). This is a key engineering component in ensuring that large and deep neural networks can converge in a stable fashion.

To normalise an input batch, we scale it in this fashion:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

where $\mu_B$ is the mean of the mini-batch, $\sigma_B^2$ is the variance of the mini-batch, and $\epsilon$ is a small constant added for numerical stability.

Next, we define two learnable parameters gamma and beta to allow the model to reconstruct the original (non-scaled) distribution, if needed:

$$
out = \gamma \hat{x}_i + \beta
$$

If the learned $\gamma$ is equal to $1$ and the learned $\beta$ is equal to $0$, this layer is simply normalising the input to mean $0$ and variance $1$. However, the model may learn other parameters that allows it to scale and shift the distribution as it pleases.

### Implementation

```python
class BatchNorm(nn.Module):
    def __init__(
        self,
        embedding_dim,
        eps=1e-5,
        momentum=0.1
      ):
        super(BatchNorm, self).__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))
        
        # Initialize running mean and variance
        self.register_buffer(
            'running_mean',
            torch.zeros(embedding_dim)
          )
        self.register_buffer(
            'running_var',
            torch.ones(embedding_dim)
          )

    def forward(self, x):
        # mean and var are only updated in training
        if self.training:
            batch_mean = x.mean(keepdim=True)
            batch_var = x.var(unbiased=False, keepdim=True)
            
            # Update running statistics
            self.running_mean = (
                (1 - self.momentum) * 
                self.running_mean + 
                self.momentum * 
                batch_mean
              )
            self.running_var = (
                (1 - self.momentum) *
                self.running_var + 
                self.momentum * 
                batch_var
             )
        # stored mean and variance are used in inference
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        # Normalise to mean 0 variance 1
        x_hat = (
            (x - batch_mean) /
            torch.sqrt(batch_var + self.eps)
          )
        
        # Scale and shift the distribution
        out = self.gamma * x_hat + self.beta
        
        return out
```

## Residual Connections

Residual connections were introduced by He et al. (2015) <d-cite key="he-2016-deep"></d-cite>. It is a technique to address the problem of vanishing gradients and to make it easier to train very deep neural networks.

As neural networks become deeper, gradients can become very small during backpropagation, making it difficult to update the weights of earlier layers effectively. This can lead to very slow convergence or even stopping learning altogether.

Residual connections address these issues by allowing the gradient to flow directly through the network, making it easier to train deep networks. They work by adding the input of a layer to its output, effectively allowing the network to learn a residual mapping (identity function).

<div class="row mt-3" style="max-width: 80%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/residual-connection.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 11:</b> Residual Connection in a feed forward neural network. The input $X$ branches in two directions: the first direction goes into the MLP and is processed normally. The other direction skips the MLP completely, (i.e., the input is not changed at all). Both branches are aggregated afterwards. Source: He, Kaiming, et al. (2016) <d-cite key="he-2016-deep"></d-cite>.
</div>

For very deep neural nets, at some point the data transformations being applied to the input are not beneficial anymore, and performance starts decreasing. To fix this, residual connections allow some layers to simply learn "nothing", i.e. learn to pass the input as the output without modifying it whatsoever (i.e., learn the identity function). With this approach, scaling the network should never hurt performance, as it can simply stop learning new functions if needed.

### Implementation

```python
class GenericLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ...
        )

    def forward(self, x):
        out = x + self.net(x)

        return out
```

# Implementation

Check this [google collab](https://colab.research.google.com/drive/1FONo7C6lrhkPSaq5TFLeQ14lWMMisW0e?usp=sharing) to test the architecture with a practical example borrowed from [Andrej Karpathy's lecture series](https://youtu.be/kCc8FmEb1nY).

```python
# Source: adapted from https://youtu.be/kCc8FmEb1nY
class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    """A decoder-only transformer block."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """Bigram language model using transformers."""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
```

# Variations

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/llm-evolutionary-tree.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 12:</b> Evolutionary Tree of Language Models (<a href="https://github.com/Mooler0410/LLMsPracticalGuide">Source</a>).
</div>

## Encoder-Decoder (suitable for seq2seq tasks)
- Suitable for sequence-to-sequence tasks such as machine translation, text summarisation, image captioning.
- Encoder-Decoder models have the advantage of **bidirectionality** when calculating the attention scores. In other words, we don't apply the masking step described previously, allowing tokens from time step t attend to **ALL** tokens (from past and future time steps) instead of only the past tokens.
- In general terms, the model is allowed to process the entire sequence first, and then it will generate the output tokens, as opposed to looking at previous tokens and predicting the current token.
- However, these advantages come with the downside of higher computational cost and complexity. Moreover, research found that other variations are more suitable for tasks where this bidirectionality is not needed.
- Examples of Encoder-Decoder models: BART, T5, T0
- One important implementation detail about encoder-decoder architectures that was not mentioned previously, is that the second MHA layer in the decoder takes the Key and the Value outputs from the encoder as input, while the Query comes from the Masked MHA from the encoder (refer to the first figure).

## Encoder-only (suitable for input understanding tasks e.g. classification)
- Encoder-only architectures are suitable for classification tasks, since we don't want to map the input sequence to another sequence, but rather map the input sequence to a pre-defined set of labels. Therefore, we only need to encode, i.e., learn how to represent the sequence in latent space, and then append a classification head on top to discriminate between the classes. Common tasks using encoder-only models include: text classification, named entity recognition (NER), question answering (with extractive approaches), document embeddings.
- Encoder-only models also have bidirectionality, and are simpler and more efficient than encoder-decoder models. However, they are not suitable for generative tasks, as they only encode the sequence, but can't decode it.
- Examples of encoder-only models: BERT, RoBERTa, DeBERTa

## Decoder-only (suitable for generative tasks)
- Decoder-only architectures are suitable for generative tasks, since these settings are naturally autoregressive (dependant on previous context, not on future context).
- Generative tasks using decoder-only models include: language modelling, text generation, dialogue systems, text autocompletion.
- Examples of decoder-only models: XLNeT, GPT-3, ChatGPT, LLaMa.