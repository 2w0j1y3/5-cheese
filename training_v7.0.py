import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import random
import time
import os
import pickle
from datetime import datetime
import glob
import lz4.frame
import logging
from pathlib import Path
import sys
import gc
import torch
import torch.nn as nn
import torch.optim as optim

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 设置随机种子
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

BOARD_SIZE = 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MOVES = BOARD_SIZE * BOARD_SIZE

BASE_DIR = Path("")
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
FIGURES_DIR = BASE_DIR / "figures"
POINTS_DIR = BASE_DIR / "points"

for dir_path in [BASE_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, POINTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

def setup_logging():
    """设置日志系统"""
    log_filename = LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    return logging.getLogger(__name__)

logger = setup_logging()

class ProgressBar:
    def __init__(self, total, prefix='', length=90):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.start_time = time.time()
        self.last_update = 0
        self.last_width = 0

    def update(self, current, gap=0, draw_rate=0, defense=0):
        now = time.time()
        if now - self.last_update < 0.5 and current != self.total:
            return
        self.last_update = now
        elapsed = time.time() - self.start_time
        if current > 0:
            eta = elapsed / current * (self.total - current)
        else:
            eta = 0
        percent = current / self.total * 100
        filled = int(self.length * current // self.total)
        bar = '|'+'█' * filled + '─' * (self.length - filled)+'|'
        progress_str = (f'\r{self.prefix} {bar} '
                       f'{percent:3.0f}% '
                       f'[{int(elapsed):d}:{int(eta):d}] '
                       f'差距:{gap:4.1f}% '
                       f'平局:{draw_rate:4.1f}% '
                       f'防守:{defense:5d}')
        if len(progress_str) < self.last_width:
            sys.stdout.write(progress_str + ' ' * (self.last_width - len(progress_str)))
        else:
            sys.stdout.write(progress_str)
        sys.stdout.flush()
        self.last_width = len(progress_str)
        if current == self.total:
            print()

class Gomoku:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.board = None
        self.current_player = 1
        self.last_move = None
        self.move_history = []
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.last_move = None
        self.move_history = []
        return self.get_state()

    def get_state(self):
        return (self.board.copy(), self.current_player)

    def get_valid_moves(self):
        return [(i, j) for i in range(self.board_size)
                for j in range(self.board_size) if self.board[i, j] == 0]

    def get_nearby_pieces(self, row, col, distance=2):
        count = 0
        for i in range(max(0, row-distance), min(self.board_size, row+distance+1)):
            for j in range(max(0, col-distance), min(self.board_size, col+distance+1)):
                if self.board[i, j] != 0:
                    count += 1
        return count

    def make_move(self, position):
        i, j = position
        if self.board[i, j] != 0:
            return False, self.get_state(), -10, True
        self.board[i, j] = self.current_player
        self.last_move = position
        self.move_history.append((position, self.current_player))
        if self.check_win(i, j):
            return True, self.get_state(), 1, True
        if len(self.get_valid_moves()) == 0:
            return True, self.get_state(), 0, True
        self.current_player = 2 if self.current_player == 1 else 1
        return True, self.get_state(), 0, False

    def check_win(self, row, col):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        player = self.board[row, col]
        for dr, dc in directions:
            count = 1
            for step in range(1, 5):
                r, c = row + dr*step, col + dc*step
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                    break
                if self.board[r, c] == player:
                    count += 1
                else:
                    break
            for step in range(1, 5):
                r, c = row - dr*step, col - dc*step
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                    break
                if self.board[r, c] == player:
                    count += 1
                else:
                    break
            if count >= 5:
                return True
        return False

    def is_opponent_threat(self, row, col, opponent):
        original = self.board[row, col]
        self.board[row, col] = opponent
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        threats = {'five': 0, 'live_four': 0, 'rush_four': 0, 'live_three': 0}
        for dr, dc in directions:
            left_count = 0
            left_blocked = False
            for step in range(1, 5):
                r = row - dr*step
                c = col - dc*step
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                    left_blocked = True
                    break
                if self.board[r, c] == opponent:
                    left_count += 1
                elif self.board[r, c] == 0:
                    break
                else:
                    left_blocked = True
                    break
            right_count = 0
            right_blocked = False
            for step in range(1, 5):
                r = row + dr*step
                c = col + dc*step
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                    right_blocked = True
                    break
                if self.board[r, c] == opponent:
                    right_count += 1
                elif self.board[r, c] == 0:
                    break
                else:
                    right_blocked = True
                    break
            total = 1 + left_count + right_count
            if total >= 5:
                threats['five'] += 1
            elif total == 4:
                left_free = not left_blocked
                right_free = not right_blocked
                if left_free and right_free:
                    threats['live_four'] += 1
                elif left_free or right_free:
                    threats['rush_four'] += 1
            elif total == 3:
                left_free = not left_blocked
                right_free = not right_blocked
                if left_free and right_free:
                    threats['live_three'] += 1
        self.board[row, col] = original
        if threats['five'] > 0 or threats['live_four'] > 0 or threats['live_three'] >= 2 or \
           (threats['live_three'] >= 1 and threats['rush_four'] >= 1) or threats['rush_four'] >= 2:
            return True
        return False

    def evaluate_position(self, row, col, player):
        attack_score = 0
        defend_score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        opponent = 2 if player == 1 else 1
        for dr, dc in directions:
            attack_score += self._evaluate_direction(row, col, player, dr, dc, is_opponent=False)
            defend_score += self._evaluate_direction(row, col, opponent, dr, dc, is_opponent=True) * 2.0
        center = self.board_size // 2
        center_dist = abs(row - center) + abs(col - center)
        center_bonus = (self.board_size - center_dist) * 2
        attack_score += center_bonus
        defend_score += center_bonus * 0.5
        nearby = self.get_nearby_pieces(row, col, 2)
        cluster_bonus = nearby * 10
        attack_score += cluster_bonus
        defend_score += cluster_bonus * 1.5
        return attack_score, defend_score

    def _evaluate_direction(self, row, col, player, dr, dc, is_opponent=False):
        count = 1
        left_block = False
        right_block = False
        left_space = 0
        right_space = 0
        for step in range(1, 5):
            r, c = row + dr*step, col + dc*step
            if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                right_block = True
                break
            if self.board[r, c] == player:
                count += 1
            elif self.board[r, c] == 0:
                right_space = self._count_empty(r, c, dr, dc)
                break
            else:
                right_block = True
                break
        for step in range(1, 5):
            r, c = row - dr*step, col - dc*step
            if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                left_block = True
                break
            if self.board[r, c] == player:
                count += 1
            elif self.board[r, c] == 0:
                left_space = self._count_empty(r, c, -dr, -dc)
                break
            else:
                left_block = True
                break
        base_score = 0
        if count >= 5:
            base_score += 1000000
        elif count == 4:
            if not (left_block and right_block):
                base_score += 500000 if is_opponent else 200000
            else:
                base_score += 100000 if is_opponent else 20000
        elif count == 3:
            if not (left_block and right_block):
                base_score += 200000 if is_opponent else 80000
            else:
                base_score += 30000 if is_opponent else 8000
        elif count == 2:
            if not (left_block and right_block):
                base_score += 2000 if is_opponent else 1000
            else:
                base_score += 400 if is_opponent else 200
        elif count == 1:
            base_score += 20 if is_opponent else 10
        if not left_block:
            base_score += left_space * (100 if is_opponent else 50)
        if not right_block:
            base_score += right_space * (100 if is_opponent else 50)
        return base_score

    def _count_empty(self, row, col, dr, dc, max_steps=3):
        count = 0
        for step in range(1, max_steps+1):
            r, c = row + dr*step, col + dc*step
            if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                break
            if self.board[r, c] == 0:
                count += 1
            else:
                break
        return count

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, board_size=BOARD_SIZE):
        super(DQN, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.fc2 = nn.Linear(512, board_size * board_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DDQNAI:
    def __init__(self, player, learning_rate=0.001, discount_factor=0.95,
                 exploration_rate=0.5, exploration_decay=0.995,
                 buffer_size=100000, batch_size=64, target_update_freq=100,
                 attack_weight=0.4, defend_weight=0.5,
                 attack_heuristic=0.001, defend_heuristic=0.0005,
                 attack_reward_coef=0.001, defend_reward_coef=0.0005):
        self.player = player
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exploration_decay
        self.lr = learning_rate
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.attack_weight = attack_weight
        self.defend_weight = defend_weight
        self.attack_heuristic = attack_heuristic
        self.defend_heuristic = defend_heuristic
        self.attack_reward_coef = attack_reward_coef
        self.defend_reward_coef = defend_reward_coef

        self.policy_net = DQN(BOARD_SIZE).to(DEVICE)
        self.target_net = DQN(BOARD_SIZE).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        self.memory = ReplayBuffer(buffer_size)

        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.name = f"AI-{player}"
        self.total_games = 0
        self.defense_count = 0
        self.critical_defense_count = 0

    def _state_to_tensor(self, state):
        board, current_player = state
        opponent = 2 if current_player == 1 else 1
        ch_self = (board == current_player).astype(np.float32)
        ch_opponent = (board == opponent).astype(np.float32)
        ch_turn = np.full((BOARD_SIZE, BOARD_SIZE), 1.0 if current_player == 1 else 0.0, dtype=np.float32)
        tensor = np.stack([ch_self, ch_opponent, ch_turn], axis=0)
        return torch.from_numpy(tensor).unsqueeze(0).to(DEVICE)

    def get_action(self, state, valid_moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)

        temp_env = Gomoku(BOARD_SIZE)
        board_array, _ = state
        temp_env.board = board_array.copy()
        opponent = 2 if self.player == 1 else 1
        threat_moves = self._detect_all_threats(temp_env, opponent)
        if threat_moves:
            forced_moves = [m for m in valid_moves if m in threat_moves]
            if forced_moves:
                valid_moves = forced_moves

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        mask = np.full(BOARD_SIZE * BOARD_SIZE, -np.inf)
        for i, j in valid_moves:
            idx = i * BOARD_SIZE + j
            mask[idx] = q_values[idx]
        best_idx = np.argmax(mask)
        best_move = (best_idx // BOARD_SIZE, best_idx % BOARD_SIZE)
        return best_move

    def _detect_all_threats(self, env, opponent):
        threats = set()
        board = env.board
        for i in range(env.board_size):
            for j in range(env.board_size):
                if board[i, j] == 0 and env.is_opponent_threat(i, j, opponent):
                    threats.add((i, j))
        return threats

    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self._train_step()
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.total_games += 1

    def _train_step(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        state_batch = torch.cat([self._state_to_tensor(s) for s in states], dim=0)
        next_state_batch = torch.cat([self._state_to_tensor(ns) for ns in next_states], dim=0)
        action_batch = torch.tensor([a[0]*BOARD_SIZE + a[1] for a in actions], device=DEVICE).unsqueeze(1)
        reward_batch = torch.tensor(rewards, device=DEVICE, dtype=torch.float32)
        done_batch = torch.tensor(dones, device=DEVICE, dtype=torch.float32)

        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze()
            target_q = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.functional.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filename, total_training_games=None):
        if total_training_games is None:
            total_training_games = self.total_games
        save_data = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_games': self.total_games,
            'total_training_games': total_training_games,
            'epsilon': self.epsilon,
            'defense_count': self.defense_count,
            'critical_defense_count': self.critical_defense_count,
            'lr': self.lr,
            'attack_weight': self.attack_weight,
            'defend_weight': self.defend_weight,
            'attack_heuristic': self.attack_heuristic,
            'defend_heuristic': self.defend_heuristic,
            'attack_reward_coef': self.attack_reward_coef,
            'defend_reward_coef': self.defend_reward_coef,
        }
        data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = lz4.frame.compress(data)
        with open(filename, 'wb') as f:
            f.write(compressed)
        size_mb = len(compressed) / (1024 * 1024)
        print(f"模型已保存: {Path(filename).name} ({size_mb:.2f} MB)")

    def load(self, filename):
        with open(filename, 'rb') as f:
            compressed = f.read()
        data = lz4.frame.decompress(compressed)
        save_data = pickle.loads(data)
        self.policy_net.load_state_dict(save_data['policy_net_state_dict'])
        self.target_net.load_state_dict(save_data['target_net_state_dict'])
        self.optimizer.load_state_dict(save_data['optimizer_state_dict'])
        self.wins = save_data.get('wins', 0)
        self.losses = save_data.get('losses', 0)
        self.draws = save_data.get('draws', 0)
        self.total_games = save_data.get('total_games', 0)
        self.total_training_games = save_data.get('total_training_games', self.total_games)
        self.epsilon = save_data.get('epsilon', self.epsilon)
        self.defense_count = save_data.get('defense_count', 0)
        self.critical_defense_count = save_data.get('critical_defense_count', 0)
        self.lr = save_data.get('lr', self.lr)
        self.attack_weight = save_data.get('attack_weight', self.attack_weight)
        self.defend_weight = save_data.get('defend_weight', self.defend_weight)
        self.attack_heuristic = save_data.get('attack_heuristic', 0.001)
        self.defend_heuristic = save_data.get('defend_heuristic', 0.0005)
        self.attack_reward_coef = save_data.get('attack_reward_coef', 0.001)
        self.defend_reward_coef = save_data.get('defend_reward_coef', 0.0005)
        print(f"模型已加载: {Path(filename).name}")

class AITrainer:
    def __init__(self, ai1, ai2, board_size=BOARD_SIZE):
        self.ai1 = ai1
        self.ai2 = ai2
        self.board_size = board_size
        self.start_episode = 0
        self.no_progress_counter = 0
        self.last_gap = None

        self.stats = {
            'ai1_wins': [], 'ai2_wins': [], 'draws': [],
            'ai1_epsilon': [], 'ai2_epsilon': [],
            'ai1_lr': [], 'ai2_lr': [],
            'ai1_attack_weight': [], 'ai1_defend_weight': [],
            'ai2_attack_weight': [], 'ai2_defend_weight': [],
            'defense_count': [], 'critical_defense': [],
            'gap': []
        }
        self.game_details = []
        self.recent_results = []
        self.balance_count = 0
        self.last_balance_check = 0
        self.global_aggressiveness = 1.0
        self.high_draw_counter = 0

    def play_game(self, first_player=None, training=True):
        env = Gomoku(self.board_size)
        if first_player is None:
            first_player = random.choice([1, 2])
        if first_player == 2:
            env.current_player = 2
        state = (env.board.copy(), env.current_player)
        done = False
        move_count = 0
        defense_this_game = 0
        critical_defense_this_game = 0
        while not done and move_count < MAX_MOVES:
            move_count += 1
            current_ai = self.ai1 if env.current_player == 1 else self.ai2
            valid_moves = env.get_valid_moves()
            opponent = 2 if env.current_player == 1 else 1
            urgent_moves = self._detect_urgent_defense(env, opponent, current_ai)
            if urgent_moves:
                defense_this_game += 1
                if len(urgent_moves) > 0:
                    critical_defense_this_game += 1
            action = current_ai.get_action(state, valid_moves, training=training)
            success, next_state, reward, done = env.make_move(action)
            if not success:
                current_ai.update(state, action, -10, next_state, done)
                break
            current_ai.update(state, action, reward, next_state, done)
            state = (env.board.copy(), env.current_player)
        if done:
            if reward == 1:
                winner = current_ai
                loser = self.ai2 if current_ai == self.ai1 else self.ai1
                winner.wins += 1
                loser.losses += 1
                result = 'ai1_win' if winner == self.ai1 else 'ai2_win'
            elif reward == 0:
                self.ai1.draws += 1
                self.ai2.draws += 1
                result = 'draw'
            else:
                winner = self.ai2 if current_ai == self.ai1 else self.ai1
                loser = current_ai
                winner.wins += 1
                loser.losses += 1
                result = 'ai2_win' if winner == self.ai2 else 'ai1_win'
            if current_ai == self.ai1:
                self.ai1.defense_count += defense_this_game
                self.ai1.critical_defense_count += critical_defense_this_game
            else:
                self.ai2.defense_count += defense_this_game
                self.ai2.critical_defense_count += critical_defense_this_game
            self.recent_results.append(result)
            if len(self.recent_results) > 100:
                self.recent_results.pop(0)
        return result, first_player

    def _detect_urgent_defense(self, env, opponent, ai):
        threats = ai._detect_all_threats(env, opponent)
        return list(threats)

    def train(self, episodes=10000, save_every=5000, log_every=500):
        print("\n" + "=" * 60)
        print(f"开始训练: {episodes} 局")
        print(f"已有训练: {self.start_episode} 局")
        print(f"总计将达: {self.start_episode + episodes} 局")
        print(f"保存间隔: 每 {save_every} 局")
        print(f"日志记录: 每 {log_every} 局 (详见logs文件夹)")
        print("=" * 60)
        print()
        start_time = time.time()
        progress = ProgressBar(episodes, prefix='训练')
        balance_interval = 70
        for episode in range(1, episodes + 1):
            first_player = 1 if episode % 2 == 0 else 2
            result, _ = self.play_game(first_player=first_player, training=True)
            if episode % 10 == 0:
                self.stats['ai1_wins'].append(self.ai1.wins)
                self.stats['ai2_wins'].append(self.ai2.wins)
                self.stats['draws'].append(self.ai1.draws)
                self.stats['ai1_epsilon'].append(self.ai1.epsilon)
                self.stats['ai2_epsilon'].append(self.ai2.epsilon)
                self.stats['ai1_lr'].append(self.ai1.lr)
                self.stats['ai2_lr'].append(self.ai2.lr)
                self.stats['defense_count'].append(self.ai1.defense_count + self.ai2.defense_count)
                self.stats['critical_defense'].append(self.ai1.critical_defense_count + self.ai2.critical_defense_count)
                self.stats['ai1_attack_weight'].append(self.ai1.attack_weight)
                self.stats['ai1_defend_weight'].append(self.ai1.defend_weight)
                self.stats['ai2_attack_weight'].append(self.ai2.attack_weight)
                self.stats['ai2_defend_weight'].append(self.ai2.defend_weight)
                total = self.ai1.wins + self.ai2.wins + self.ai1.draws
                if total > 0:
                    rate1 = self.ai1.wins / total * 100
                    rate2 = self.ai2.wins / total * 100
                    self.stats['gap'].append(abs(rate1 - rate2))
            if episode % balance_interval == 0:
                self._balance_ai()
            if episode % log_every == 0:
                elapsed = time.time() - start_time
                speed = episode / elapsed if elapsed > 0 else 0
                total_games = self.ai1.wins + self.ai1.losses + self.ai1.draws
                if len(self.recent_results) >= 100:
                    recent_wins1 = self.recent_results.count('ai1_win')
                    recent_wins2 = self.recent_results.count('ai2_win')
                    recent_draws = self.recent_results.count('draw')
                    recent_rate1 = recent_wins1 / len(self.recent_results) * 100
                    recent_rate2 = recent_wins2 / len(self.recent_results) * 100
                    recent_draw_rate = recent_draws / len(self.recent_results) * 100
                    gap = abs(recent_rate1 - recent_rate2)
                else:
                    recent_rate1 = self.ai1.wins / total_games * 100 if total_games > 0 else 0
                    recent_rate2 = self.ai2.wins / total_games * 100 if total_games > 0 else 0
                    recent_draw_rate = self.ai1.draws / total_games * 100 if total_games > 0 else 0
                    gap = abs(recent_rate1 - recent_rate2)
                total_defense = self.ai1.defense_count + self.ai2.defense_count
                avg_defense = total_defense / episode if episode > 0 else 0
                logger.info(f"\n{'='*50}")
                logger.info(f"第 {episode}/{episodes} 局 ({episode/episodes*100:.1f}%)")
                logger.info(f"速度: {speed:.1f}局/秒, 已用: {elapsed/60:.1f}分钟")
                logger.info(f"总战绩: AI1 {self.ai1.wins}胜, AI2 {self.ai2.wins}胜, 平局 {self.ai1.draws}")
                logger.info(f"近期100局: AI1 {recent_rate1:.1f}%, AI2 {recent_rate2:.1f}%, 平局 {recent_draw_rate:.1f}%")
                logger.info(f"胜率差距: {gap:.1f}%")
                logger.info(f"防守统计: 总{total_defense}次, 平均{avg_defense:.2f}次/局")
                logger.info(f"参数: AI1 ε={self.ai1.epsilon:.3f}, lr={self.ai1.lr:.3f}")
                logger.info(f"       AI2 ε={self.ai2.epsilon:.3f}, lr={self.ai2.lr:.3f}")
                logger.info(f"{'='*50}")
            if episode % save_every == 0 and episode != episodes:
                elapsed = time.time() - start_time
                speed = episode / elapsed
                total_games = self.ai1.wins + self.ai1.losses + self.ai1.draws
                rate1 = self.ai1.wins / total_games * 100 if total_games > 0 else 0
                rate2 = self.ai2.wins / total_games * 100 if total_games > 0 else 0
                draw_rate = self.ai1.draws / total_games * 100 if total_games > 0 else 0
                gap = abs(rate1 - rate2)
                total_defense = self.ai1.defense_count + self.ai2.defense_count
                logger.info(f"\n📊 第 {episode} 局检查点 速度:{speed:.1f}局/秒")
                logger.info(f"  战绩: AI1 {self.ai1.wins}胜 AI2 {self.ai2.wins}胜 平局 {self.ai1.draws}")
                logger.info(f"  胜率: AI1 {rate1:.1f}% AI2 {rate2:.1f}% 平局 {draw_rate:.1f}%")
                logger.info(f"  差距:{gap:.1f}% 防守:{total_defense}次")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                total_episodes = self.start_episode + episode
                self.ai1.save(POINTS_DIR / f"AI1_point_{total_episodes}_{timestamp}.pkl", total_training_games=total_episodes)
                self.ai2.save(POINTS_DIR / f"AI2_point_{total_episodes}_{timestamp}.pkl", total_training_games=total_episodes)
                self.plot_stats(episode)
            if len(self.recent_results) >= 100:
                recent_wins1 = self.recent_results.count('ai1_win')
                recent_wins2 = self.recent_results.count('ai2_win')
                recent_draws = self.recent_results.count('draw')
                recent_total = len(self.recent_results)
                recent_gap = abs(recent_wins1 - recent_wins2) / recent_total * 100
                recent_draw_rate = recent_draws / recent_total * 100
            else:
                total_games = self.ai1.wins + self.ai1.losses + self.ai1.draws
                if total_games > 0:
                    recent_gap = abs(self.ai1.wins - self.ai2.wins) / total_games * 100
                    recent_draw_rate = self.ai1.draws / total_games * 100
                else:
                    recent_gap = 0
                    recent_draw_rate = 0
            total_defense = self.ai1.defense_count + self.ai2.defense_count
            progress.update(episode, recent_gap, recent_draw_rate, total_defense)
            if episode % 10000 == 0:
                gc.collect()
        elapsed_total = time.time() - start_time
        print(f"\n\n🎉 训练完成！总用时: {elapsed_total/60:.1f}分钟")
        print(f"平均速度: {episodes/elapsed_total:.1f}局/秒")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_episodes = self.start_episode + episodes
        self.ai1.save(MODELS_DIR / f"AI1_final_{total_episodes}_{timestamp}.pkl", total_training_games=total_episodes)
        self.ai2.save(MODELS_DIR / f"AI2_final_{total_episodes}_{timestamp}.pkl", total_training_games=total_episodes)
        self.plot_stats(episodes)

    def _balance_ai(self):
        """改进后的平衡调整（目标平局率 25%~40%）
        同时同步更新优化器的学习率
        """
        if len(self.recent_results) < 50:
            return

        wins1 = self.recent_results.count('ai1_win')
        wins2 = self.recent_results.count('ai2_win')
        draws = self.recent_results.count('draw')
        total = len(self.recent_results)
        if total == 0:
            return

        rate1 = wins1 / total * 100
        rate2 = wins2 / total * 100
        draw_rate = draws / total * 100
        gap = abs(rate1 - rate2)

        self.balance_count += 1

        # 检测无进步（基于平局率变化）
        if hasattr(self, 'last_draw_rate') and abs(draw_rate - self.last_draw_rate) < 2:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0
        self.last_draw_rate = draw_rate

        # 高平局计数器（用于全局进攻倾向）
        if draw_rate > 80:
            self.high_draw_counter += 1
        else:
            self.high_draw_counter = 0

        # 全局进攻倾向调整（长期记忆，用于极端情况）
        if self.high_draw_counter >= 5:
            self.global_aggressiveness = min(3.0, self.global_aggressiveness * 1.2)
            logger.info(f"📈 全局进攻倾向提升至 {self.global_aggressiveness:.2f}")
            self.high_draw_counter = 0
        else:
            # 缓慢回归到1.0
            self.global_aggressiveness = max(1.0, self.global_aggressiveness * 0.99)

        # 若平局率持续过高且长时间无改善，强制干预
        if draw_rate > 80 and self.no_progress_counter >= 10:
            logger.info("⚠️ 平局率持续过高，执行强制干预！")
            self._force_intervention()
            # 强制干预后，可能已经修改了学习率，但为了确保优化器同步，我们显式更新两个AI的优化器学习率
            for ai in [self.ai1, self.ai2]:
                ai.optimizer.param_groups[0]['lr'] = ai.lr
            self.no_progress_counter = 0
            return

        # 确定强弱
        if rate1 > rate2:
            strong_ai, weak_ai = self.ai1, self.ai2
            strong_name, weak_name = "AI1(强)", "AI2(弱)"
        else:
            strong_ai, weak_ai = self.ai2, self.ai1
            strong_name, weak_name = "AI2(强)", "AI1(弱)"

        # 保存旧值（用于日志）
        old_strong_eps, old_weak_eps = strong_ai.epsilon, weak_ai.epsilon
        old_strong_lr, old_weak_lr = strong_ai.lr, weak_ai.lr
        old_strong_attack_w = strong_ai.attack_weight
        old_strong_defend_w = strong_ai.defend_weight
        old_weak_attack_w = weak_ai.attack_weight
        old_weak_defend_w = weak_ai.defend_weight
        old_strong_attack_h = strong_ai.attack_heuristic
        old_strong_defend_h = strong_ai.defend_heuristic
        old_weak_attack_h = weak_ai.attack_heuristic
        old_weak_defend_h = weak_ai.defend_heuristic

        # ========== 根据胜率差距调整 ε 和 lr ==========
        if gap > 40:
            weak_ai.epsilon = max(0.005, weak_ai.epsilon * 0.6)
            strong_ai.epsilon = max(0.002, strong_ai.epsilon * 0.8)
            weak_ai.lr = min(0.3, weak_ai.lr * 1.2)
            strong_ai.lr = min(0.28, strong_ai.lr * 1.1)
            level = "差距>40%"
        elif gap > 20:
            weak_ai.epsilon = max(0.01, weak_ai.epsilon * 0.8)
            strong_ai.epsilon = max(0.003, strong_ai.epsilon * 0.9)
            weak_ai.lr = min(0.3, weak_ai.lr * 1.1)
            strong_ai.lr = min(0.25, strong_ai.lr * 1.05)
            level = "差距>20%"
        elif gap < 10:
            weak_ai.epsilon = min(0.4, weak_ai.epsilon * 1.3)
            strong_ai.epsilon = max(0.002, strong_ai.epsilon * 0.9)
            weak_ai.lr = max(0.08, weak_ai.lr * 0.95)
            strong_ai.lr = min(0.25, strong_ai.lr * 1.02)
            level = "差距<10% (拉开)"
        else:
            weak_ai.lr = min(0.28, weak_ai.lr * 1.02)
            strong_ai.lr = min(0.25, strong_ai.lr * 1.01)
            level = "微调"

        # 同步优化器学习率（修改lr后立即更新）
        strong_ai.optimizer.param_groups[0]['lr'] = strong_ai.lr
        weak_ai.optimizer.param_groups[0]['lr'] = weak_ai.lr

        # ========== 根据平局率调整攻防权重和启发式系数 ==========
        MIN_WEIGHT, MAX_WEIGHT = 0.05, 0.95
        MIN_HEUR, MAX_HEUR = 0.0001, 0.01

        # 目标平局率区间：25%~40%
        if draw_rate > 80:
            # 极高平局：强力进攻（结合全局倾向）
            factor = self.global_aggressiveness
            new_attack_w = min(MAX_WEIGHT, strong_ai.attack_weight * factor)
            new_defend_w = max(MIN_WEIGHT, strong_ai.defend_weight / factor)
            strong_ai.attack_weight = new_attack_w
            strong_ai.defend_weight = new_defend_w
            weak_ai.attack_weight = new_attack_w
            weak_ai.defend_weight = new_defend_w

            strong_ai.attack_heuristic = min(MAX_HEUR, strong_ai.attack_heuristic * factor)
            strong_ai.defend_heuristic = max(MIN_HEUR, strong_ai.defend_heuristic / factor)
            weak_ai.attack_heuristic = min(MAX_HEUR, weak_ai.attack_heuristic * factor)
            weak_ai.defend_heuristic = max(MIN_HEUR, weak_ai.defend_heuristic / factor)

            # 强化弱AI学习率，弱化强AI学习率（注意这里也修改了lr，需要同步优化器）
            weak_ai.lr = min(0.3, weak_ai.lr * 1.5)
            strong_ai.lr = max(0.01, strong_ai.lr * 0.8)
            # 同步优化器
            strong_ai.optimizer.param_groups[0]['lr'] = strong_ai.lr
            weak_ai.optimizer.param_groups[0]['lr'] = weak_ai.lr

            draw_action = f"强力进攻(全局倾向 x{self.global_aggressiveness:.2f})"
        elif draw_rate > 60:
            # 较高平局：中等进攻
            factor = 1.2
            strong_ai.attack_weight = min(MAX_WEIGHT, strong_ai.attack_weight * factor)
            strong_ai.defend_weight = max(MIN_WEIGHT, strong_ai.defend_weight / factor)
            weak_ai.attack_weight = min(MAX_WEIGHT, weak_ai.attack_weight * factor)
            weak_ai.defend_weight = max(MIN_WEIGHT, weak_ai.defend_weight / factor)

            strong_ai.attack_heuristic = min(MAX_HEUR, strong_ai.attack_heuristic * factor)
            strong_ai.defend_heuristic = max(MIN_HEUR, strong_ai.defend_heuristic / factor)
            weak_ai.attack_heuristic = min(MAX_HEUR, weak_ai.attack_heuristic * factor)
            weak_ai.defend_heuristic = max(MIN_HEUR, weak_ai.defend_heuristic / factor)

            # 此分支没有修改lr，无需同步优化器
            draw_action = "中等进攻"
        elif draw_rate > 40:
            # 略高于目标区间：温和进攻
            factor = 1.05
            strong_ai.attack_weight = min(MAX_WEIGHT, strong_ai.attack_weight * factor)
            strong_ai.defend_weight = max(MIN_WEIGHT, strong_ai.defend_weight / factor)
            weak_ai.attack_weight = min(MAX_WEIGHT, weak_ai.attack_weight * factor)
            weak_ai.defend_weight = max(MIN_WEIGHT, weak_ai.defend_weight / factor)

            strong_ai.attack_heuristic = min(MAX_HEUR, strong_ai.attack_heuristic * factor)
            strong_ai.defend_heuristic = max(MIN_HEUR, strong_ai.defend_heuristic / factor)
            weak_ai.attack_heuristic = min(MAX_HEUR, weak_ai.attack_heuristic * factor)
            weak_ai.defend_heuristic = max(MIN_HEUR, weak_ai.defend_heuristic / factor)

            draw_action = "温和进攻"
        elif draw_rate < 25:
            # 平局过少：温和加强防守
            factor = 0.95  # 防守加强因子
            strong_ai.attack_weight = max(MIN_WEIGHT, strong_ai.attack_weight * factor)
            strong_ai.defend_weight = min(MAX_WEIGHT, strong_ai.defend_weight / factor)
            weak_ai.attack_weight = max(MIN_WEIGHT, weak_ai.attack_weight * factor)
            weak_ai.defend_weight = min(MAX_WEIGHT, weak_ai.defend_weight / factor)

            strong_ai.attack_heuristic = max(MIN_HEUR, strong_ai.attack_heuristic * factor)
            strong_ai.defend_heuristic = min(MAX_HEUR, strong_ai.defend_heuristic / factor)
            weak_ai.attack_heuristic = max(MIN_HEUR, weak_ai.attack_heuristic * factor)
            weak_ai.defend_heuristic = min(MAX_HEUR, weak_ai.defend_heuristic / factor)

            draw_action = "温和防守"
        else:
            # 25% ≤ 平局率 ≤ 40%，处于理想区间，保持现有参数
            draw_action = "保持"

        # 统一限制参数范围
        for ai in [strong_ai, weak_ai]:
            ai.epsilon = max(0.001, min(0.5, ai.epsilon))
            ai.lr = max(0.01, min(0.3, ai.lr))
            # 优化器学习率必须与ai.lr一致
            ai.optimizer.param_groups[0]['lr'] = ai.lr
            ai.attack_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, ai.attack_weight))
            ai.defend_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, ai.defend_weight))
            ai.attack_heuristic = max(MIN_HEUR, min(MAX_HEUR, ai.attack_heuristic))
            ai.defend_heuristic = max(MIN_HEUR, min(MAX_HEUR, ai.defend_heuristic))

        logger.info(f"\n⚖️ 平衡调整 #{self.balance_count}: {level} 差距={gap:.1f}% 平局率={draw_rate:.1f}% {draw_action}")
        logger.info(f"  {strong_name}: ε {old_strong_eps:.3f}→{strong_ai.epsilon:.3f}, lr {old_strong_lr:.3f}→{strong_ai.lr:.3f}, "
                    f"攻击权 {old_strong_attack_w:.2f}→{strong_ai.attack_weight:.2f}, 防守权 {old_strong_defend_w:.2f}→{strong_ai.defend_weight:.2f}, "
                    f"进攻启发 {old_strong_attack_h:.4f}→{strong_ai.attack_heuristic:.4f}, 防守启发 {old_strong_defend_h:.4f}→{strong_ai.defend_heuristic:.4f}")
        logger.info(f"  {weak_name}: ε {old_weak_eps:.3f}→{weak_ai.epsilon:.3f}, lr {old_weak_lr:.3f}→{weak_ai.lr:.3f}, "
                    f"攻击权 {old_weak_attack_w:.2f}→{weak_ai.attack_weight:.2f}, 防守权 {old_weak_defend_w:.2f}→{weak_ai.defend_weight:.2f}, "
                    f"进攻启发 {old_weak_attack_h:.4f}→{weak_ai.attack_heuristic:.4f}, 防守启发 {old_weak_defend_h:.4f}→{weak_ai.defend_heuristic:.4f}")


    def _force_intervention(self):
        """强制干预：当长时间无进步时，大幅调整参数，但保留全局进攻倾向"""
        logger.info("🔧 执行强制干预策略")

        # 根据当前胜率确定强弱（使用最近100局）
        wins1 = self.recent_results.count('ai1_win')
        wins2 = self.recent_results.count('ai2_win')
        total = len(self.recent_results)
        if total > 0:
            rate1 = wins1 / total * 100
            rate2 = wins2 / total * 100
            if rate1 > rate2:
                strong_ai, weak_ai = self.ai1, self.ai2
                strong_name, weak_name = "AI1(强)", "AI2(弱)"
            else:
                strong_ai, weak_ai = self.ai2, self.ai1
                strong_name, weak_name = "AI2(强)", "AI1(弱)"
        else:
            # 如果还没有对局，随机指定
            strong_ai, weak_ai = self.ai1, self.ai2
            strong_name, weak_name = "AI1", "AI2"

        # 保存旧值
        old_strong_eps = strong_ai.epsilon
        old_weak_eps = weak_ai.epsilon
        old_strong_lr = strong_ai.lr
        old_weak_lr = weak_ai.lr
        old_strong_attack_w = strong_ai.attack_weight
        old_strong_defend_w = strong_ai.defend_weight
        old_weak_attack_w = weak_ai.attack_weight
        old_weak_defend_w = weak_ai.defend_weight

        # 强制调整探索率和学习率（与原来一致）
        strong_ai.epsilon = max(0.002, min(0.05, strong_ai.epsilon * 0.3))
        strong_ai.lr = min(0.25, strong_ai.lr * 1.3)

        weak_ai.epsilon = min(0.4, weak_ai.epsilon * 2.5)
        weak_ai.lr = min(0.3, weak_ai.lr * 1.5)

        # 重置攻击/防守权重时，结合全局进攻倾向
        MIN_WEIGHT, MAX_WEIGHT = 0.05, 0.95
        # 基础默认值
        base_attack = 0.2
        base_defend = 0.7
        # 应用全局倾向：进攻倾向越高，攻击权重乘因子，防守权重除因子
        factor = self.global_aggressiveness
        new_attack = min(MAX_WEIGHT, base_attack * factor)
        new_defend = max(MIN_WEIGHT, base_defend / factor)

        strong_ai.attack_weight = new_attack
        strong_ai.defend_weight = new_defend
        weak_ai.attack_weight = new_attack
        weak_ai.defend_weight = new_defend

        # 同时启发式系数也按同样逻辑调整（可选）
        strong_ai.attack_heuristic = min(0.01, 0.001 * factor)
        strong_ai.defend_heuristic = max(0.0001, 0.0005 / factor)
        weak_ai.attack_heuristic = min(0.01, 0.001 * factor)
        weak_ai.defend_heuristic = max(0.0001, 0.0005 / factor)

        logger.info(f"  强制调整后:")
        logger.info(f"  {strong_name}: ε {old_strong_eps:.3f}→{strong_ai.epsilon:.3f}, lr {old_strong_lr:.3f}→{strong_ai.lr:.3f}")
        logger.info(f"  {weak_name}: ε {old_weak_eps:.3f}→{weak_ai.epsilon:.3f}, lr {old_weak_lr:.3f}→{weak_ai.lr:.3f}")
        logger.info(f"  攻击权重: {old_strong_attack_w:.2f}→{strong_ai.attack_weight:.2f} (因子={factor:.2f})")
        logger.info(f"  防守权重: {old_strong_defend_w:.2f}→{strong_ai.defend_weight:.2f}")

    def plot_stats(self, episode):
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            episodes = range(len(self.stats['ai1_wins']))
            axes[0, 0].plot(episodes, self.stats['ai1_wins'], label='AI1', color='#2c3e50')
            axes[0, 0].plot(episodes, self.stats['ai2_wins'], label='AI2', color='#7f8c8d')
            axes[0, 0].plot(episodes, self.stats['draws'], label='Draws', color='#e74c3c')
            axes[0, 0].set_xlabel('Episodes (x10)')
            axes[0, 0].set_ylabel('Wins')
            axes[0, 0].set_title('Win Statistics')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            total = np.array(self.stats['ai1_wins']) + np.array(self.stats['ai2_wins']) + np.array(self.stats['draws'])
            total_safe = np.where(total == 0, 1, total)
            rate1 = np.array(self.stats['ai1_wins']) / total_safe * 100
            rate2 = np.array(self.stats['ai2_wins']) / total_safe * 100
            axes[0, 1].plot(episodes, rate1, label='AI1 Win Rate', color='#2c3e50')
            axes[0, 1].plot(episodes, rate2, label='AI2 Win Rate', color='#7f8c8d')
            gap = np.array(self.stats['gap'])
            axes[0, 1].plot(episodes, gap, label='Gap', color='red', linestyle='--', alpha=0.7)
            axes[0, 1].axhline(y=50, color='red', linestyle=':', alpha=0.5)
            axes[0, 1].set_xlabel('Episodes (x10)')
            axes[0, 1].set_ylabel('Win Rate / Gap (%)')
            axes[0, 1].set_title('Win Rate Trend with Gap')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 100)
            axes[0, 2].plot(episodes, self.stats['ai1_epsilon'], label='AI1', color='#2c3e50')
            axes[0, 2].plot(episodes, self.stats['ai2_epsilon'], label='AI2', color='#7f8c8d')
            axes[0, 2].set_xlabel('Episodes (x10)')
            axes[0, 2].set_ylabel('Epsilon')
            axes[0, 2].set_title('Exploration Rate')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            axes[1, 0].plot(episodes, self.stats['ai1_lr'], label='AI1', color='#2c3e50')
            axes[1, 0].plot(episodes, self.stats['ai2_lr'], label='AI2', color='#7f8c8d')
            axes[1, 0].set_xlabel('Episodes (x10)')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 1].plot(episodes, self.stats['ai1_attack_weight'], label='AI1', color='#2c3e50')
            axes[1, 1].plot(episodes, self.stats['ai2_attack_weight'], label='AI2', color='#7f8c8d')
            axes[1, 1].set_xlabel('Episodes (x10)')
            axes[1, 1].set_ylabel('Attack Weight')
            axes[1, 1].set_title('Attack Weight')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 2].plot(episodes, self.stats['ai1_defend_weight'], label='AI1', color='#2c3e50')
            axes[1, 2].plot(episodes, self.stats['ai2_defend_weight'], label='AI2', color='#7f8c8d')
            axes[1, 2].set_xlabel('Episodes (x10)')
            axes[1, 2].set_ylabel('Defend Weight')
            axes[1, 2].set_title('Defend Weight')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim(0, 1)
            plt.tight_layout()
            filename = FIGURES_DIR / f"stats_ep{episode}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  统计图已保存: {filename.name}")
        except Exception as e:
            logger.info(f"  绘制统计图失败: {e}")

def list_all_models():
    models = list(MODELS_DIR.glob("*.pkl"))
    if not models:
        print("没有找到任何模型文件")
        return []
    print("\n可用的模型文件：")
    valid_models = []
    for i, model in enumerate(models):
        try:
            with open(model, 'rb') as f:
                compressed = f.read()
                data = lz4.frame.decompress(compressed)
                save_data = pickle.loads(data)
                wins = save_data.get('wins', 0)
                losses = save_data.get('losses', 0)
                draws = save_data.get('draws', 0)
                total = wins + losses + draws
                win_rate = wins/total*100 if total > 0 else 0
                q_size = len(save_data.get('q_table', {}))
                defense = save_data.get('defense_count', 0)
                print(f"{i+1}. {model.name} - {wins}胜 {losses}负 {draws}平 "
                      f"(胜率 {win_rate:.1f}%, 防守{defense}次, Q表{q_size})")
                valid_models.append(str(model))
        except Exception as e:
            print(f"{i+1}. {model.name} - 无法读取: {e}")
            valid_models.append(str(model))
    return valid_models

def main():
    print("\n" + "=" * 60)
    print("五子棋AI训练系统 (Double DQN 版)")
    print("=" * 60)
    models = list_all_models()
    print("\n请选择黑棋 AI (player 1): (输入0表示创建新 DDQN AI)")
    if models:
        for i, m in enumerate(models):
            print(f"{i+1}. {Path(m).name}")
    choice1 = int(input("输入序号: ")) - 1
    if choice1 == -1:
        model1 = None
        print("选择黑棋: 新 DDQN AI")
    else:
        model1 = models[choice1]
        print(f"选择黑棋: {Path(model1).name}")
    print("\n请选择白棋 AI (player 2): (输入0表示创建新 DDQN AI)")
    choice2 = int(input("输入序号: ")) - 1
    if choice2 == -1:
        model2 = None
        print("选择白棋: 新 DDQN AI")
    else:
        model2 = models[choice2]
        print(f"选择白棋: {Path(model2).name}")
    ai_black = DDQNAI(player=1, learning_rate=0.001, exploration_rate=0.4)
    ai_white = DDQNAI(player=2, learning_rate=0.001, exploration_rate=0.4)
    if model1 is not None:
        ai_black.load(model1)
    if model2 is not None:
        ai_white.load(model2)
    start_episode = max(
        getattr(ai_black, 'total_training_games', ai_black.total_games) if model1 is not None else 0,
        getattr(ai_white, 'total_training_games', ai_white.total_games) if model2 is not None else 0
    )
    trainer = AITrainer(ai_black, ai_white)
    trainer.start_episode = start_episode
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    episodes = input("请输入训练局数 (默认10000): ")
    episodes = int(episodes) if episodes.strip() else 10000
    save_every = input("保存间隔 (默认5000): ")
    save_every = int(save_every) if save_every.strip() else 5000
    log_every = input("日志记录间隔 (默认500): ")
    log_every = int(log_every) if log_every.strip() else 500
    print(f"\n训练配置:")
    print(f"  总局数: {episodes}")
    print(f"  已有训练: {start_episode} 局")
    print(f"  保存间隔: 每 {save_every} 局")
    print(f"  日志记录: 每 {log_every} 局")
    print(f"  模型目录: {MODELS_DIR}")
    print(f"  日志目录: {LOGS_DIR}")
    print(f"  图表目录: {FIGURES_DIR}")
    input("\n按Enter开始训练...")
    trainer.train(episodes=episodes, save_every=save_every, log_every=log_every)
    total = ai_black.wins + ai_black.losses + ai_black.draws
    print(f"{ai_black.name}: {ai_black.wins}胜 {ai_black.losses}负 {ai_black.draws}平 "
          f"(胜率 {ai_black.wins/total*100:.1f}%)")
    print(f"{ai_white.name}: {ai_white.wins}胜 {ai_white.losses}负 {ai_white.draws}平 "
          f"(胜率 {ai_white.wins/total*100:.1f}%)")
    total_defense = ai_black.defense_count + ai_white.defense_count
    total_critical = ai_black.critical_defense_count + ai_white.critical_defense_count
    print(f"\n🛡️ 防守统计:")
    print(f"  总防守次数: {total_defense}")
    print(f"  关键防守次数: {total_critical}")
    print(f"  平均每局防守: {total_defense/episodes:.2f}次")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()