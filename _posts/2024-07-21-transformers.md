---
layout: distill
title: Transformers
description: Notes on the Transformer architecture.
tags: study nlp
giscus_comments: true
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

# Motivation

The motivation behind the transformer architecture lies in the limitations found in the previous state-of-the-art architectures for seq2seq tasks that used Recurrent Neural Networks (RNNs).

The Long Short-Term Memory (LSTM) architecture encoded the context from previous time steps in a single "state vector" that is continuously updated as information flows through each processing step. However, this results in the state vector suffering significant changes from one step to the next, and thus losing previous context quickly, especially from far-away previous steps.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/rnn.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <b>Figure 2:</b> Recurrent Neural Network: each time step $t$ processes an input $x_t$ and outputs an output $o_t$. A state vector $V$ is carried from previous time steps (initialised as zero). Source: Raj, Dinesh Samuel Sathia et al. <d-cite key="raj-2019-analysis"></d-cite>
</div>


To account for this, Gated Recurrent Units (GRU) were introduced by Chung, Junyoung et al. (2014) <d-cite key="chung-etal-2014-empirical"></d-cite>. The key idea behind GRU is the use of gate mechanisms that open or close the information flow (i.e., control when and how much to update the context vector in a given time step). This allows the context to flow for longer sequences, as some unimportant states can be ignored.

<div class="row mt-3" style="max-width: 50%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/gru.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 3:</b> The update gate $z$ selects whether the hidden state $h_t$ is to be updated with a new hidden state $\tilde{h}_t$. The reset gate $r$ decides whether the previous hidden state $h_{t-1}$ is ignored. Source: Cho, Kyunghyun, et al. <d-cite key="cho-etal-2014-learning"></d-cite>
</div>

There are two major shortcomings in these RNN approaches:
- **Information is lost over long-contexts**: for long sequences, the state vector loses the context from previous time steps. A single vector does not provide enough bandwidth to carry all relevant information over the long context.
- **Compute inefficiency due to non-paralallelisable operations**: To generate a token $o_t$ at time step $t$, we need to process all previous inputs $x_i$ at time step $i < t$ sequentially, which is computationally inefficient.

The transformer architecture addresses these shortcomings by:
- **Using Self-Attention:** Instead of having a state vector be updated at each time step, the transformer uses the self-attention (SA) mechanism to allow all tokens to capture each other's context. Also, this token attention is not computed sequentially. A token $x_t$ attends to all tokens $x_i$ with $i < t$ simultaneously. On itself, this is would be a limitation, since we lose notion of word ordering. On LSTMs, because the vector is updated sequentially, we preserve the order of information flow (from first word to last). With SA, we lose the notion of word ordering, because there is no sequencing in the computation. To fix this, we introduce positional embeddings, which will be discussed later.
  - SA allows a token $x_t$ to attend to previous tokens with the same strength as any other token, close or far from itself. Thus, it fixes the information loss over long-contexts.
  - SA is not sequential, thus we can compute the attention scores for all pairs of tokens in the sequence in a very efficient and parallelisable way with matrix multiplication.

However there is one important shortcoming introduced in the transformer architecture with respect to previous RNN approaches:
- Now that we don't have recursion, we must define a fixed context length (i.e., the maximum number of elements in the sequence).
- If an input sequence is smaller than this length, we fill the sequence with special tokens such as \<PAD\> tokens.
- If an input sequence is larger than this length, we have to discard some tokens. Usually we truncate the sequence up to <i>max_seq_len</i> tokens, and discard the remaining tokens.
- All transformer-based approaches have this pre-defined maximum sequence length. For most models similar to BERT, the maximum length is $512$ tokens. For GPT-2 it is $1024$, GPT-3 is $2048$, LLaMa3 is around $8000$ tokens.

# Components

## Multi-Head Self-Attention (MHA)

The Multi-Head Self-Attention mechanism allows words in the sequence to "pay attention" (attend) to each other, and through the learning procedure, eventually they will be able of identifying which words from previous time-steps are relevant to generate the element for the current time step.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/attention-example.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 4:</b> Self-Attention: The red word is the word being generated at the current time step. Blue words are words with high attention score at the current time step. Note that words from future time steps are not considered. For example at time step $0$, only the first word "The" is considered. For time step $1$, only the words "The" and "FBI" are considered. Source: Allam, Tarek & McEwen, Jason. (2021) <d-cite key="allam-mcewen-2023-paying"></d-cite>
</div>

To learn this component, we define three weight matrices: Query ($Q$), Key ($K$), and Value ($V$). These weight matrices will be multipled to obtain the attention matrix in the following way:

$$
\begin{equation}\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V\end{equation} 
$$

We apply a dot product between the queries and the keys, and then divide the result by the square root of the number of dimensions of the vectors. This is done to make the dot-product maintain variance close to $1$. Then, we apply softmax to make this distribution sum to $1$, and finally apply a dot-product with the value matrix. Basically the softmax computation acts as a weight to scale the influence of the value matrix.

<div class="row mt-3" style="max-width: 80%; height: auto; margin: 0 auto;">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/attention-mechanism.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 5:</b> Attention Mechanism: The input X is multiplied with matrices $\theta$, $\phi$, and $g$ to obtain the resulting matrices $Q$, $K$, and $V$. $Q$ and $K$ are multiplied to obtain the attention score matrix, that is finally multiplied with the $V$ matrix (<a href="https://blogs.oracle.com/ai-and-datascience/post/multi-head-self-attention-in-nlp">Source</a>).
</div>

A good intuition for how this works is to compare the concept with a lookup table / hash table / dictionary.

In a lookup table, we have the following structure:

```python
d = {
    "dog": "bark",
    "wolf": "howl",
    "cat": "meow"
}

# this key exists, the query will match
# and return "value1"
query1 = "dog" 
value1 = lookup[query1]

# this key doesnt exist, the query won't 
# match and return and error
query3 =  "bird"
value3 = lookup[query3]
```

Note how a query will either match entirely with a single key (and then return it's value), or it will not match at all with any of the keys (and then return an error or a predefined value).

In the attention mechanism, we are performing a similar operation, but we implement a "soft" match of keys and queries. In fact, all queries will match will all keys, but in different intensities. The softmax computation introduces this notion of intensity, which will weigh the value that is going to be returned.

```python
d = {
    "dog": "bark":,
    "wolf": "howl",
    "cat": "meow"
}

# query doesn't exist in the dictionary
# but still we will match it with the keys by their similarity
query = "husky"

# the query has high similarity with dog, less with wolf, almost none with cat
attention = 0.7 * lookup["dog"] + 0.29 * lookup["wolf"] + 0.01 * lookup["cat"]
```

The output of the scaled dot-product is a square matrix with dimensions <i>context_length</i> by <i>context_length</i>, representing the attention score of token $i$ for token $j$. Note that the attention for token $i$ to token $j$ is not the same for token $j$ to token $i$ (i.e., the matrix is not symmetric). We then multiply this matrix with the value matrix to scale it according to the attention scores. $V$ has dimension (<i>context_size</i>, <i>embedding_dim</i>), thus the dot product between the attention scores and $V$ yields a matrix (<i>context_size</i>, <i>embedding_dim</i>).

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

## Word Embeddings and Positional Encodings

Unstructured inputs (e.g., text, images) must be converted into a numerical representation that will be processed. Mikolov et al. (2013) <d-cite key="mikolov-2013-efficient"></d-cite> introduced the concept of word embeddings, which are latent vectors that capture the semantics of words.

In transformers, we also define an embedding matrix that will be learned jointly with the other components. The embedding matrix is initialised randomly, and is of dimension (<i>vocabulary_size</i>, <i>embedding_dimension</i>). Each input token will retrieve an embedding from the table. This embedding will then represent this token in input space.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/word-embeddings.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 8:</b> Word Embeddings (<a href="https://arize.com/blog-course/embeddings-meaning-examples-and-how-to-compute/">Source</a>).
</div>

Along the word embeddings, the transformer architecture defines a second type of embedding called **positional encodings**. Previously in this document we discussed how the SA mechanism doesn't introduce any notion of space. The model doesn't know what is the first and the second token and so on. Therefore, we must somehow encode this information.

There are many approaches to encode position into the word embeddings. Some approaches don't involve learning new weights. For example, we can use sine and cosine functions to encode relative positions as a combination of previous positions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/blogposts/positional-embeddings.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
<b>Figure 9:</b> Positional Encoding with sine and cosine functions (<a href="https://towardsdatascience.com/understanding-positional-encoding-in-transformers-dc6bafc021ab">Source</a>).
</div>

The benefit of this approach is that:
1. We don't have to learn the embeddings, they are calculated analytically
2. We can encode any arbitrary position up to infinity

Other approaches involve learning the embeddings jointly with the transformer block. We define an embedding matrix of size (<i>context_size</i>, <i>embedding_dimention</i>). For each token position, we will learn an embedding. This generally works well, at the cost of having a fixed number of positions (up to <i>context_size</i>), and adding more trainable parameters.

The input to the transformer block is the element-wise sum of both the word embedding and the positional encodings.

### Implementation

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

    def forward(self, x):
        x_emb = self.embedding_table(x)
        pos_emb = self.positional_embedding_table(
            torch.arange(
                self.block_size,
                device=device
              )
            )

        x_emb = x_emb + pos_emb
        
        return x_emb
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