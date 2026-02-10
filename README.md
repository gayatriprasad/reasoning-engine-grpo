# Reasoning Engine GRPO

This repository contains the code and documentation for fine-tuning a Large Language Model (LLM) to improve its mathematical and logical reasoning capabilities. The project leverages **Group Relative Policy Optimization (GRPO)**, a reinforcement learning technique, to train a base model (Alibaba's Qwen3) on the `open-R1-math` dataset.

The entire process is designed to run on local hardware, accelerated by the `Unsloth` library for memory and speed optimizations.

## Project Goal

The primary goal is not just to fine-tune a model, but to build a "reasoning engine" by explicitly rewarding the model for generating logically sound, step-by-step thought processes. This moves beyond simple answer accuracy and focuses on creating a more transparent and reliable model.

## Core Technologies

*   **Model:** Qwen3 (4B parameter version)
*   **Dataset:** `HuggingFaceH4/open-R1-math`
*   **Fine-Tuning Framework:** `Unsloth` for high-performance LoRA training.
*   **Reinforcement Learning:** `trl` library for implementing GRPO.
*   **Experiment Tracking:** `wandb` (Weights & Biases) for logging metrics and comparing runs.

## Why GRPO? A Strategic Choice for Reasoning

Aligning an LLM's behavior is a key challenge. While Supervised Fine-Tuning (SFT) teaches a model to mimic a dataset, Reinforcement Learning (RL) teaches it to optimize for a specific goal. For this project, our goal is to improve reasoning. We chose GRPO after considering several popular RL algorithms.

| Algorithm | How it Works | Pros | Cons | Why Not For This Project? |
| :--- | :--- | :--- | :--- | :--- |
| **PPO (Proximal Policy Optimization)** | An "actor-critic" method. It uses a separate, trained **reward model** to score the LLM's (actor's) output and a **value function** (critic) to stabilize training. | Powerful and general-purpose. The standard for many complex RL tasks. | **High Complexity:** Requires training and maintaining a separate reward model, which is a significant engineering and computational overhead. **Instability:** Can be difficult to tune and prone to "reward hacking" where the model finds loopholes in the reward function. | The overhead of training a separate reward model is too high for our goal. We have a deterministic way to score math reasoning (the answer is either right or wrong), so a learned reward model is unnecessary complexity. |
| **DPO (Direct Preference Optimization)** | A simpler, offline method. It learns from a dataset of **preference pairs** (`chosen` vs. `rejected` responses). It directly optimizes the policy to increase the likelihood of the `chosen` responses. | **Simple & Stable:** Does not require a separate reward model or complex hyperparameters. Very effective for general alignment (e.g., "be more helpful"). | **Binary Signal:** Only learns from a simple "this is better than that" signal. It cannot distinguish between a slightly wrong answer and a completely nonsensical one. | Reasoning is not a binary problem. An answer that is off by 1 is much better than an answer that is wildly incorrect. DPO's simple win/loss signal doesn't capture this nuance, which is critical for teaching mathematical reasoning. |
| **GRPO (Group Relative Policy Optimization)** | An on-policy method where the model generates a **group of responses** for a single prompt. A **deterministic reward function** (which we write) scores each response. The algorithm then updates the model to favor the characteristics of the higher-scoring responses over the lower-scoring ones. | **Nuanced Rewards:** Allows for a granular, continuous reward signal (e.g., proximity to the correct answer). **No Reward Model:** Avoids the complexity of PPO by using a simple, code-based reward function. **Efficient:** Learns from multiple responses per prompt, making it more sample-efficient than DPO. | Requires generating multiple responses during training, which adds some computational cost over DPO. | **This is the ideal choice for our project.** It allows us to define precisely what "good reasoning" means through our reward function (e.g., correct final answer, correct format) and provides a much richer, more nuanced signal than DPO without the massive overhead of PPO. |

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/[your-username]/[your-repo-name].git
cd [your-repo-name]
```

### 2. Set Up the Environment

This project uses a Python virtual environment to manage dependencies.
```bash
# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### 3. Install Dependencies

The core dependencies are managed via `pip` and the `requirements.txt` file. The `Unsloth` library provides optimized kernels for Apple Silicon.
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install Unsloth with M-series kernels and other dependencies
pip install "unsloth[m5-mps-new-kernels]"

# Install utilities
pip install -r requirements.txt
```

### 4. Weights & Biases Login

This project uses `wandb` for experiment tracking. You will need a free account.
```bash
# Log in to your wandb account
wandb login
```

## Usage

The project is structured into Jupyter notebooks:

1.  `01_SFT_pre-tuning.ipynb`: Performs Supervised Fine-Tuning to teach the model the desired output format.
2.  `02_GRPO_reasoning_tuning.ipynb`: Implements the main GRPO training loop to refine the model's reasoning abilities.
3.  `03_Inference_and_Evaluation.ipynb`: Contains code to run inference with the final model and evaluate its performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
