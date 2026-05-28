import pygame
from game_board import GameBoard
from player import Player
from rl_ai import RLAgent
import matplotlib.pyplot as plt


def plot_rewards(rewards1, rewards2):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards1, label='Player 1')
    plt.plot(rewards2, label='Player 2')
    plt.title('Episode Rewards over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('rewards_plot.png')
    plt.close()


def train():
    game_board = GameBoard(40, 30)
    rl_agent1 = RLAgent(state_size=7 * 7 * 3, action_size=4, player_id=1)
    rl_agent2 = RLAgent(state_size=7 * 7 * 3, action_size=4, player_id=2)
    player1 = Player(10, 15, (255, 0, 0), 1, rl_agent1)
    player2 = Player(30, 15, (0, 0, 255), 2, rl_agent2)

    # Set opponents
    player1.set_opponent(player2)
    player2.set_opponent(player1)

    num_episodes = 10000 #10000 initially
    rewards1 = []
    rewards2 = []

    for episode in range(num_episodes):
        game_board = GameBoard(40, 30)
        player1.reset(10, 15)
        player2.reset(30, 15)

        # mark initial positions
        game_board.grid[player1.y][player1.x] = player1.player_id
        game_board.grid[player2.y][player2.x] = player2.player_id

        done = False
        total_reward1 = 0
        total_reward2 = 0
        max_steps = 2000
        step = 0

        while not done and step < max_steps:
            step += 1
            # Choose actions for both players (self-play)
            s1 = rl_agent1.get_state(game_board, player1, player2)
            s2 = rl_agent2.get_state(game_board, player2, player1)

            a1_idx, a1_dir = rl_agent1.choose_action(game_board, player1, player2)
            a2_idx, a2_dir = rl_agent2.choose_action(game_board, player2, player1)

            # Move both using chosen actions (avoid querying controller twice)
            collision1 = player1.move(game_board, a1_dir)
            collision2 = player2.move(game_board, a2_dir)

            # Compute next states
            next_s1 = rl_agent1.get_state(game_board, player1, player2)
            next_s2 = rl_agent2.get_state(game_board, player2, player1)

            # Determine rewards
            if collision1 or game_board.is_collision(player1.x, player1.y):
                reward1, reward2 = -50, 20
                done = True
            elif collision2 or game_board.is_collision(player2.x, player2.y):
                reward1, reward2 = 20, -50
                done = True
            else:
                reward1, reward2 = 5, 5

            total_reward1 += reward1
            total_reward2 += reward2

            # Update board if still alive
            if not done:
                game_board.grid[player1.y][player1.x] = player1.player_id
                game_board.grid[player2.y][player2.x] = player2.player_id

            # Store transitions (use action indices)
            rl_agent1.remember(s1, a1_idx, reward1, next_s1, done)
            rl_agent2.remember(s2, a2_idx, reward2, next_s2, done)

            # Learn
            rl_agent1.learn()
            rl_agent2.learn()

        rewards1.append(total_reward1)
        rewards2.append(total_reward2)

        if episode % 100 == 0:
            print(f"Episode: {episode}, Player 1 Reward: {total_reward1}, Player 2 Reward: {total_reward2}")
            plot_rewards(rewards1, rewards2)

    rl_agent1.save_model("tron_model_player1.pth")
    rl_agent2.save_model("tron_model_player2.pth")
    plot_rewards(rewards1, rewards2)


if __name__ == "__main__":
    train()