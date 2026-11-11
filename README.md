# TensorTonic Solutions

Welcome to my TensorTonic solutions repository!

Here you'll find my solutions to various machine learning and deep learning problems from [TensorTonic](https://tensortonic.com).

## What is TensorTonic?

TensorTonic is a platform where you can implement core algorithms of Machine Learning from scratch.

This repository contains my personal solutions to these problems, automatically synchronized from the platform.

<!-- tensortonic:start -->
# Matteo Podda's TensorTonic Solutions

Verified machine learning implementations completed on [TensorTonic](https://www.tensortonic.com).

<p align="center">
  <img src="https://www.tensortonic.com/api/badge/podda_matt.svg" alt="TensorTonic Verified Solutions" width="100%" />
</p>

| Problem | Description | Link |
|---|---|---|
| Implement Cosine Similarity | Compute cosine similarity between NumPy vectors with explicit handling for zero-norm inputs. | https://www.tensortonic.com/problems/la-cosine-similarity |
| Implement Dot Product | Compute the algebraic dot product and geometric angle relationship for two equal-length NumPy vectors. | https://www.tensortonic.com/problems/la-dot-product |
| Implement Euclidean Distance | Compute Euclidean distance between equal-length NumPy vectors from the square root of summed squared differences. | https://www.tensortonic.com/problems/la-euclidean-distance |
| Mean Squared Error (MSE) | Compute mean squared error between predictions and targets by averaging their squared element-wise differences. | https://www.tensortonic.com/problems/mean-squared-error |
| Activation Functions | Implement four common activation functions from scratch using basic PyTorch tensor operations (no torch.nn module). | https://www.tensortonic.com/problems/pytorch-activation-function-from-scratch |
| Attention Mechanism from Scratch | Implement the scaled dot-product attention mechanism, a core building block of the Transformer architecture. | https://www.tensortonic.com/problems/pytorch-attention-from-scratch |
| Balanced DataLoader | Build a PyTorch DataLoader that balances class sampling with per-example weights derived from label frequencies. | https://www.tensortonic.com/problems/pytorch-balanced-dataloader |
| Basic Autograd | Use PyTorch autograd to evaluate a scalar function and return its derivative at every supplied input value. | https://www.tensortonic.com/problems/pytorch-basic-autograd |
| Batch Normalization | Normalize each feature across the batch, then scale and shift using learnable parameters. | https://www.tensortonic.com/problems/pytorch-batch-normalization |
| Simple Neural Network | Implement a class SimpleNet subclassing nn.Module with two linear layers and ReLU between them. | https://www.tensortonic.com/problems/pytorch-build-simple-nn-from-scratch |
| Custom Dataset Class | Implement a PyTorch Dataset over row records with indexed feature tensors and labels. | https://www.tensortonic.com/problems/pytorch-custom-dataclass |
| Custom Linear Layer | Implement a custom linear layer that computes the affine transformation without using any built-in linear layer. | https://www.tensortonic.com/problems/pytorch-custom-linear-layer |
| Custom SGD with Momentum | Implement momentum SGD by subclassing the PyTorch optimizer interface and maintaining per-parameter velocity. | https://www.tensortonic.com/problems/pytorch-custom-optimizer |
| Dropout from Scratch | Implement PyTorch inverted dropout from a supplied mask during training while returning inputs unchanged in evaluation mode. | https://www.tensortonic.com/problems/pytorch-dropout-from-scratch |
| Early Stopping | Train a PyTorch model with validation monitoring and stop after the configured number of unimproved epochs. | https://www.tensortonic.com/problems/pytorch-early-stopping |
| Gradient Accumulation | Simulate gradient accumulation over multiple micro-batches, and return the final weights and last averaged gradient. | https://www.tensortonic.com/problems/pytorch-gradient-accumulation |
| Loss Functions | Implement three common loss functions from scratch using PyTorch tensor operations: mean squared error, cross-entropy, and Huber loss. | https://www.tensortonic.com/problems/pytorch-loss-functions |
| Manual Weight Update | Perform a PyTorch training step with manual parameter updates after backpropagation, without an optimizer object. | https://www.tensortonic.com/problems/pytorch-manual-weight-update |
| Masked Causal Attention | Implement scaled dot-product attention with a causal mask that prevents each position from attending to future positions. | https://www.tensortonic.com/problems/pytorch-masked-causal-attention |
| Mini Training Loop | Run one complete PyTorch training epoch over a DataLoader and return the mean batch loss. | https://www.tensortonic.com/problems/pytorch-mini-training |
| Multi-Head Attention | Implement PyTorch multi-head attention with head splitting, scaled softmax attention, concatenation, and output projection. | https://www.tensortonic.com/problems/pytorch-multi-head-attention |
| Optimizer Scheduler | Train with a PyTorch optimizer and StepLR schedule, recording the learning rate applied at each epoch. | https://www.tensortonic.com/problems/pytorch-optimizer-scheduler |
| Residual Block | Implement a PyTorch residual block with two padded convolutions, batch normalization, ReLU, and an identity shortcut. | https://www.tensortonic.com/problems/pytorch-residual-block |
| RNN Cell from Scratch | Implement a vanilla PyTorch RNN cell that combines current inputs and previous hidden states with a tanh update. | https://www.tensortonic.com/problems/pytorch-rnn-cell-from-scratch |
| Softmax from Scratch | Implement numerically stable batched softmax in PyTorch by shifting logits before exponentiation and normalization. | https://www.tensortonic.com/problems/pytorch-softmax-from-scratch |
| Tensor Operations | Perform common element-wise and matrix tensor operations: add, multiply, matmul, power, and max. | https://www.tensortonic.com/problems/pytorch-tensor-arithmetic |
| Tensor Factory | Create PyTorch tensors with zeros, ones, or a constant fill value using the requested shape and dtype. | https://www.tensortonic.com/problems/pytorch-tensor-creation |
| Tensor Shape Manipulation | Reshape tensors using three common PyTorch operations: flatten to collapse into 1D, squeeze to remove size-1 dimensions. | https://www.tensortonic.com/problems/pytorch-tensor-reshape |
| Transform Pipeline | Implement a callable class that converts a raw image tensor into a normalized, channel-first tensor ready for a neural network. | https://www.tensortonic.com/problems/pytorch-transforms-pipeline |
| Weight Initialization | Implement a function that initializes a weight tensor using one of four standard initialization methods. | https://www.tensortonic.com/problems/pytorch-weight-initialization |
| Scaled Dot-Product Attention | Implement scaled dot-product attention in PyTorch using query-key scores, softmax weights, and value aggregation. | https://www.tensortonic.com/research/transformer/transformers-attention |
| Embedding Layer | Create PyTorch token embeddings and scale each lookup by the square root of the Transformer model dimension. | https://www.tensortonic.com/research/transformer/transformers-embedding |
| Feed-Forward Network | Implement the Transformer's position-wise feed-forward network with two linear projections and a ReLU activation. | https://www.tensortonic.com/research/transformer/transformers-feed-forward |
| Multi-Head Attention | Build NumPy multi-head attention with learned projections, per-head scaled attention, concatenation, and output projection. | https://www.tensortonic.com/research/transformer/transformers-multi-head-attention |
| Positional Encoding | Implement sinusoidal Transformer positional encodings in NumPy with alternating sine and cosine dimensions. | https://www.tensortonic.com/research/transformer/transformers-positional-encoding |
| Tokenization | Build a word-level Transformer tokenizer with fixed special-token IDs, sorted vocabulary entries, encoding, and decoding. | https://www.tensortonic.com/research/transformer/transformers-tokenization |

View my verified ML profile: [TensorTonic profile](https://www.tensortonic.com/profile/podda_matt)
<!-- tensortonic:end -->
