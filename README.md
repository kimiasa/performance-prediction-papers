# Relevant Papers on Using ML for Performance Prediction
## Table of Contents

### Multimodal or Graph-Based Deep Learning Models
1. [PERFOGRAPH: A Numerical Aware Program Graph
Representation for Performance Optimization and
Program Analysis](https://proceedings.neurips.cc/paper_files/paper/2023/file/b41907dd4df5c60f86216b73fe0c7465-Paper-Conference.pdf)

2. [Performance Optimization using Multimodal Modeling and
Heterogeneous GNN](https://dl.acm.org/doi/pdf/10.1145/3588195.3592984)

3. [GRANITE: A Graph Neural Network Model for
Basic Block Throughput Estimation](https://arxiv.org/pdf/2210.03894) 

### LSTM-Based Models
1. [Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation
using Deep Neural Networks](https://proceedings.mlr.press/v97/mendis19a/mendis19a.pdf)

### Decision Tree and KNN Models
1. [A Hybrid Machine Learning Method for Cross-Platform Performance Prediction of Parallel Applications](https://dl.acm.org/doi/pdf/10.1145/3673038.3673059)

### LLM
1. [HPC-GPT: Integrating Large Language Model for
High-Performance Computing](https://dl.acm.org/doi/10.1145/3624062.3624172)

2. [Data Race Detection Using Large Language Models](https://dl.acm.org/doi/10.1145/3624062.3624088)

# Multimodal or Graph-Based Deep Learning Models

## PERFOGRAPH: A Numerical Aware Program Graph Representation for Performance Optimization and Program Analysis
- **Authors**: TehraniJamsaz et al.
- **Year**: 2023
- **Summary**: This work proposes Perfograph, a program graph representation based on LLVM-IR for performance optimization and program analysis leveraging graph neural networks (GNNs). Perfograph includes tracking reused local identifiers and memory locations, embedding numerical constants using a decimal-based encoding, and decomposing array and vector types into discrete nodes for more nuanced representation. It is evaluated across six downstream tasks including device mapping, parallelism discovery, parallel pattern detection, NUMA and prefetchers configuration prediction, thread coarsening factor (TCF) prediction, and algorithm classification.

## Performance Optimization using Multimodal Modeling and Heterogeneous GNN
- **Authors**:  Dutta et al.
- **Year**: 2023
- **Summary**: This work proposes multimodal graph neural network and autoencoder (MGA) tuner, a multimodal deep learning based approach that utilizes Heterogeneous Graph Neural Networks along with Denoising Autoencoders to model Intermediate Representation (IR)-based code representations as distinct modalities as part of static analysis augmented with performance counter or dynamic features. It is tested for two major tasks: optimizing OpenMP loop parameters (thread count, scheduling policy, and chunk size) and determining the best device for heterogeneous mapping of OpenCL kernels.

## GRANITE: A Graph Neural Network Model for Basic Block Throughput Estimation
- **Authors**: Sykora ett al.
- **Year**: 2023
- **Summary**: This paper introduces Granite, a machine learning model that estimates the throughput of basic blocks across different microarchitectures. GRANITE uses a graph representation of basic blocks that captures both structural and data dependencies between instructions. It combines GNNs to predict basic block throughput and uses multi-layer feedforward decoder networks for scalability. It achives 1.7% less error rate compared to sota basic block performance estimator for the x86-64 target and improves training and inference throughput by approximately 3.0x. 


# LSTM-Based Models

## Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation using Deep Neural Networks
- **Authors**: Mendis et al.
- **Year**: 2019
- **Summary**: This paper introduces Ithemal, a deep learning tool for predicting the throughput of a set of instructions on x86-64 architectures. Ithemal uses a hierarchical LSTM approach to learn throughput predictions from opcodes and operands of instructions in a basic block. The authors show that Ithemal outperforms sota tools like LLVM's llvm-mca and Intel's IACA, achieving less than half their error rate while maintaining comparable speed. Additionally, Ithemal can be adapted to various processor microarchitectures with minimal effort, addressing the challenges of building and maintaining analytical models for complex CISC machines.
  
# Decision Tree and KNN Models

## A Hybrid Machine Learning Method for Cross-Platform Performance Prediction of Parallel Applications
- **Authors**: Mahdavi, Kaveh
- **Year**: 2024
- **Summary**: This paper introduces a machine learning approach for predicting parallel application performance across diverse computing platforms. The method uses performance ratios between a reference and target platforms, requiring only brief partial executions on the reference system by mapping CPU bursts to labeled data. It utilizes The Ensemble Cluster Classify Regress (ECCR) method, which is a three-stage approach using LightGBM models to capture nuanced performance characteristics, and a k-Nearest Neighbor Classifier to select the most appropriate regression model for predictions. Experiments across various platforms and applications demonstrate cross-validation accuracy exceeding 98% and execution time prediction accuracy for unseen applications exceeding 94%.

# LLM

## HPC-GPT: Integrating Large Language Model for High-Performance Computing
- **Authors**: Ding et al.
- **Year**: 2023
- **Summary**: This paper introduces HPC-GPT, a LLaMA-based model fine-tuned for high-performance computing (HPC) tasks. Aiming to address the limitations of general Large Language Models (LLMs) in HPC, HPC-GPT is trained on QA pairs specific to HPC. The model is tested on two key tasks: managing AI models and datasets for HPC, and detecting data races. Results show that HPC-GPT performs comparably to existing methods, highlighting its potential to bridge the gap between LLMs and HPC-specific applications, making these models more effective in complex computing scenarios.

## Data Race Detection Using Large Language Models
- **Authors**: Chen et al. 
- **Year**: 2023
- **Summary**: This paper explores using Large Language Models (LLMs) for detecting data races in OpenMP programs, offering an alternative to manual tool development. It proposes a dataset named DRB-ML, derived from DataRaceBench, to evaluate and fine-tune LLMs. The results show while LLMs show potential in identifying data races, they currently do not perform as well as traditional tools in providing detailed information about the involved variables.








