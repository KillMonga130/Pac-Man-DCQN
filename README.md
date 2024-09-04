# Deep Convolutional Q-Learning for Pac-Man

This project demonstrates the use of Deep Convolutional Q-Networks (DCQN) to play Pac-Man. The AI model was trained using reinforcement learning techniques, with the goal of maximizing the score while navigating the game environment.

## Project Overview

- **Environment**: MsPacmanDeterministic-v0 (from Gymnasium Atari)
- **Algorithm**: Deep Q-Network (DQN) with Convolutional Neural Networks (CNN)
- **Training Episodes**: 2000
- **Model Performance**: Solved the environment in 369 episodes with an average score of 500.10.

## How It Works

1. **Neural Network Architecture**: 
   - The network consists of four convolutional layers followed by fully connected layers that output action values.
   
2. **Reinforcement Learning**: 
   - The AI learns by interacting with the game environment and using Q-Learning to adjust its strategy over time.

3. **Hyperparameters**:
   - Learning Rate: `5e-4`
   - Minibatch Size: `64`
   - Discount Factor: `0.99`
   - Epsilon Decay: `0.995`

## Setup

To run the code, ensure you have the following dependencies installed:

```bash
pip install gymnasium "gymnasium[atari, accept-rom-license]"
pip install torch torchvision numpy
