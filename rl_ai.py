import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy


class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(256, 128)):
        super(DQN, self).__init__()
        hidden1, hidden2 = hidden_sizes
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class RLAgent:
    """Double DQN agent with a target network and replay buffer."""

    def __init__(self, state_size, action_size, player_id, model_file=None,
                 buffer_size=20000, batch_size=64, gamma=0.99,
                 lr=1e-4, epsilon_start=1.0, epsilon_final=0.05, epsilon_decay=0.9995,
                 target_update=1000, hidden_sizes=(256, 128)):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.hidden_sizes = hidden_sizes

        self._build_networks(hidden_sizes)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start if model_file is None else epsilon_final
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.learn_step = 0

        self.directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # Up, Down, Left, Right

        if model_file:
            self.load_model(model_file)

    def get_state(self, game_board, player, opponent):
        state = np.zeros((7, 7, 3), dtype=np.float32)  # 3 channels: empty, player, opponent
        for i in range(-3, 4):
            for j in range(-3, 4):
                x, y = player.x + i, player.y + j
                if 0 <= x < game_board.width and 0 <= y < game_board.height:
                    if game_board.grid[y][x] == 0:
                        state[i + 3][j + 3][0] = 1.0  # Empty
                    elif game_board.grid[y][x] == player.player_id:
                        state[i + 3][j + 3][1] = 1.0  # Player
                    else:
                        state[i + 3][j + 3][2] = 1.0  # Opponent
                else:
                    state[i + 3][j + 3][2] = 1.0  # Treat walls as opponent
        return state.flatten()

    def valid_action_indices(self, current_direction):
        invalid = [-current_direction[0], -current_direction[1]]
        return [i for i, d in enumerate(self.directions) if d != invalid]

    def choose_action(self, game_board, player, opponent):
        """Return (action_idx, direction_list) chosen by epsilon-greedy policy."""
        state = self.get_state(game_board, player, opponent)
        valid_idxs = self.valid_action_indices(player.direction)

        # Exploration
        if random.random() < self.epsilon:
            a_idx = random.choice(valid_idxs)
            return a_idx, self.directions[a_idx]

        self.online.eval()
        with torch.no_grad():
            s_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_vals = self.online(s_t)[0].cpu().numpy()
        # pick best among valid
        best = max(valid_idxs, key=lambda idx: q_vals[idx])
        return best, self.directions[best]

    def get_direction(self, game_board, player, opponent):
        """Compatibility wrapper for existing Player.move() usage."""
        _, direction = self.choose_action(game_board, player, opponent)
        return direction

    def remember(self, state, action_idx, reward, next_state, done):
        self.replay_buffer.push(state, action_idx, reward, next_state, done)

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Double DQN: online network selects action, target network evaluates
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_actions = torch.argmax(self.online(next_states), dim=1)
        next_q_target = self.target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

        expected_q = rewards + (1.0 - dones) * (self.gamma * next_q_target)

        loss = nn.MSELoss()(q_values, expected_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

    def save_model(self, filename):
        torch.save(self.online.state_dict(), filename)

    def load_model(self, filename):
        try:
            state_dict = torch.load(filename, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(filename, map_location=self.device)

        checkpoint_hidden_sizes = self._hidden_sizes_from_state_dict(state_dict)
        if checkpoint_hidden_sizes != self.hidden_sizes:
            self._build_networks(checkpoint_hidden_sizes)

        self.online.load_state_dict(state_dict)
        self.target.load_state_dict(self.online.state_dict())
        self.online.eval()

    def _build_networks(self, hidden_sizes):
        self.hidden_sizes = hidden_sizes
        self.online = DQN(self.state_size, self.action_size, hidden_sizes).to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.optimizer = optim.Adam(self.online.parameters(), lr=self.lr)

    def _hidden_sizes_from_state_dict(self, state_dict):
        try:
            hidden1, input_size = state_dict["fc1.weight"].shape
            hidden2, fc2_input = state_dict["fc2.weight"].shape
            output_size, fc3_input = state_dict["fc3.weight"].shape
        except KeyError as exc:
            raise ValueError(f"Checkpoint is missing expected DQN layer: {exc}") from exc

        if input_size != self.state_size or output_size != self.action_size:
            raise ValueError(
                "Checkpoint shape does not match this agent: "
                f"expected input/output {(self.state_size, self.action_size)}, "
                f"got {(input_size, output_size)}"
            )
        if fc2_input != hidden1 or fc3_input != hidden2:
            raise ValueError("Checkpoint contains inconsistent DQN layer shapes.")

        return (hidden1, hidden2)
