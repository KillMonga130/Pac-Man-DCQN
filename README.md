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
   ```python
   class Network(nn.Module):
    def __init__(self, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(10 * 10 * 128, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

   
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

```
## Future Improvements
Experiment with different neural network architectures for improved performance.
Explore other Atari environments with the same learning algorithm.
Add more sophisticated exploration techniques (e.g., Double DQN, Dueling DQN).
