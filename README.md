# CTorch

A minimal PyTorch-like deep learning framework written from scratch in C++,
focused on:

- understanding autograd

- dynamic computation graphs

- code quality

## Motivation

Modern deep learning frameworks abstract away critical system-level details
such as automatic differentiation, computation graphs, and tensor operations.

CTorch was built to deeply understand how frameworks like PyTorch work under
the hood by implementing a minimal yet functional version in C++.

## Non-goals

CTorch is NOT intended to be:

- A production-ready framework
- GPU-accelerated
- Optimized for performance
- A PyTorch replacement

The focus is correctness, clarity, and learning.

## Core Features

- Tensor abstraction with shape tracking
- Dynamic computation graph
- Reverse-mode automatic differentiation
- Basic tensor operations (add, mul, matmul)
- Neural network layers (Linear, ReLU)
- Optimizers (SGD)
