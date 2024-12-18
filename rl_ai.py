import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class RLAgent:
    def __init__(self, state_size, action_size, player_id, model_file=None):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id
        self.memory = deque(maxlen=10000)
        self.gamma = 0.935 # discount rate org .95
        self.epsilon = 0.01 if model_file else 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # Up, Down, Left, Right
        self.episode_rewards = []
        self.training_step = 0

        if model_file:
            self.load_model(model_file)
            self.model.eval()

    def get_state(self, game_board, player, opponent):
        state = np.zeros((7, 7, 3))  # 3 channels: empty, player, opponent
        for i in range(-3, 4):
            for j in range(-3, 4):
                x, y = player.x + i, player.y + j
                if 0 <= x < game_board.width and 0 <= y < game_board.height:
                    if game_board.grid[y][x] == 0:
                        state[i + 3][j + 3][0] = 1  # Empty
                    elif game_board.grid[y][x] == player.player_id:
                        state[i + 3][j + 3][1] = 1  # Player
                    else:
                        state[i + 3][j + 3][2] = 1  # Opponent
                else:
                    state[i + 3][j + 3][2] = 1  # Treat walls as opponent
        return state.flatten()

    def get_valid_directions(self, current_direction):
        invalid_direction = [-current_direction[0], -current_direction[1]]
        return [d for d in self.directions if d != invalid_direction]

    def get_direction(self, game_board, player, opponent):
        state = self.get_state(game_board, player, opponent)
        valid_directions = self.get_valid_directions(player.direction)

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_directions)

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        valid_q_values = [q_values[0][self.directions.index(d)].item() for d in valid_directions]
        best_valid_action = valid_directions[np.argmax(valid_q_values)]

        return best_valid_action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor)
            target_f[0][self.directions.index(action)] = target

            loss = nn.MSELoss()(self.model(state_tensor), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, game_board, player, opponent):
        state = self.get_state(game_board, player, opponent)
        done = False
        total_reward = 0

        while not done:
            action = self.get_direction(game_board, player, opponent)
            old_direction = player.direction[:]
            player.change_direction(action)

            if player.direction == [-old_direction[0], -old_direction[1]]:
                player.direction = old_direction  # Revert to the old direction

            collision = player.move(game_board)

            next_state = self.get_state(game_board, player, opponent)
            reward = 1  # Reward for surviving one more step

            if collision or game_board.is_collision(player.x, player.y):
                reward = -10  # Penalty for collision
                done = True
            elif game_board.is_collision(opponent.x, opponent.y):
                reward = 10  # Reward for opponent's collision
                done = True

            total_reward += reward
            self.remember(state, action, reward, next_state, done)
            state = next_state

            if not done:
                game_board.grid[player.y][player.x] = player.player_id

            self.training_step += 1
            if self.training_step % 4 == 0:
                self.replay(32)

        self.episode_rewards.append(total_reward)
        return total_reward

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()