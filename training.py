import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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
import shutil

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子
np.random.seed(42)
random.seed(42)

# ==================== 常量配置 ====================
BOARD_SIZE = 9
MAX_MOVES = BOARD_SIZE * BOARD_SIZE

# 目录配置
BASE_DIR = Path("")
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
FIGURES_DIR = BASE_DIR / "figures"
POINTS_DIR = BASE_DIR / "points"

for dir_path in [BASE_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, POINTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


# ==================== 日志配置 ====================
class DefaultTagFilter(logging.Filter):
    """为所有日志记录提供默认的 module_tag，避免 extra 缺失时报错"""
    def filter(self, record):
        if not hasattr(record, 'module_tag'):
            record.module_tag = 'System'   # 默认标签，如果没传 extra 就显示这个
        return True

def setup_logging():
    """设置日志系统：INFO及以上写入主日志，DEBUG单独写入调试日志"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = LOGS_DIR / f"training_{timestamp}.log"
    debug_filename = LOGS_DIR / f"debug_{timestamp}.log"

    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module_tag)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # ---------- 主日志处理器 (INFO及以上) ----------
    info_handler = logging.FileHandler(log_filename, encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    info_handler.addFilter(DefaultTagFilter())   # 确保 module_tag 存在

    # ---------- DEBUG日志处理器 (仅DEBUG级别) ----------
    debug_handler = logging.FileHandler(debug_filename, encoding='utf-8')
    debug_handler.setLevel(logging.DEBUG)        # 接收所有DEBUG及以上
    debug_handler.setFormatter(formatter)
    debug_handler.addFilter(DefaultTagFilter())
    # 关键：只允许 DEBUG 级别的记录通过
    debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)

    # ---------- 配置根记录器 ----------
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)          # 根级别必须为DEBUG，DEBUG消息才能传播
    root_logger.addHandler(info_handler)
    root_logger.addHandler(debug_handler)

    # ---------- 抑制第三方库的DEBUG噪音 ----------
    # 将matplotlib的日志级别提高到WARNING，避免其字体查找等DEBUG信息污染调试日志
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    return logging.getLogger(__name__)

logger = setup_logging()
logger.info("训练系统启动", extra={'module_tag': 'System'})


# ==================== 进度显示 ====================
class ProgressBar:
    """极简进度条 - 只显示进度条、百分比和时间"""

    COLOR_GREEN = '\033[92m'
    COLOR_YELLOW = '\033[93m'
    COLOR_BLUE = '\033[94m'
    COLOR_CYAN = '\033[96m'
    COLOR_RESET = '\033[0m'

    def __init__(self, total, prefix='', length=80, use_color=True):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.use_color = use_color
        self.start_time = time.time()
        self.last_update = 0
        self.last_visible_width = 0

    def update(self, current, gap=0, draw_rate=0, defense=0):
        now = time.time()
        if now - self.last_update < 0.5 and current != self.total:
            return

        self.last_update = now
        elapsed = time.time() - self.start_time

        if current > 0:
            eta = elapsed / current * (self.total - current)
            eta = max(0, eta)
        else:
            eta = 0

        percent = current / self.total * 100
        percent_str = f'{percent:3.0f}%'
        time_str = f'[{int(elapsed):d}:{int(eta):d}]'

        filled = int(self.length * current // self.total) if self.total > 0 else self.length
        bar_inner = '█' * filled + '─' * (self.length - filled)
        bar = '|' + bar_inner + '|'

        visible_str = f'\r{self.prefix} {bar} {percent_str} {time_str}'
        visible_len = len(visible_str)

        if self.use_color:
            colored_bar = ('|'
                           + self.COLOR_GREEN + '█' * filled + self.COLOR_RESET
                           + self.COLOR_YELLOW + '─' * (self.length - filled) + self.COLOR_RESET
                           + '|')
            colored_percent = self.COLOR_BLUE + percent_str + self.COLOR_RESET
            colored_time = self.COLOR_CYAN + time_str + self.COLOR_RESET
            output_str = f'\r{self.prefix} {colored_bar} {colored_percent} {colored_time}'
        else:
            output_str = visible_str

        sys.stdout.write(output_str)
        if visible_len < self.last_visible_width:
            sys.stdout.write(' ' * (self.last_visible_width - visible_len))
        sys.stdout.flush()
        self.last_visible_width = visible_len

        if current == self.total:
            print()

# ==================== 游戏环境 ====================
class Gomoku:
    """五子棋环境"""
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
                r, c = row + dr * step, col + dc * step
                if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                    break
                if self.board[r, c] == player:
                    count += 1
                else:
                    break
            for step in range(1, 5):
                r, c = row - dr * step, col - dc * step
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
        """
        判断对手下在 (row, col) 后是否形成必须防守的威胁。
        使用窗口枚举法检测五连、活四、冲四、活三（仅连续活三）。
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        threats = {'five': 0, 'live_four': 0, 'rush_four': 0, 'live_three': 0}

        for dr, dc in directions:
            cells = []
            r, c = row, col
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                r -= dr
                c -= dc
            r += dr
            c += dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                cells.append((r, c))
                r += dr
                c += dc

            try:
                pos = cells.index((row, col))
            except ValueError:
                continue

            has_five = False
            has_live_four = False
            has_rush_four = False
            has_live_three = False

            for start in range(max(0, pos - 4), min(pos + 1, len(cells) - 4)):
                window = cells[start:start + 5]
                cnt_x = 0
                empty_positions = []
                piece_indices = []
                for idx, (r2, c2) in enumerate(window):
                    val = self.board[r2, c2]
                    if val == opponent:
                        cnt_x += 1
                        piece_indices.append(idx)
                    elif val == 0:
                        empty_positions.append((r2, c2))
                cnt_empty = len(empty_positions)

                left_open = False
                right_open = False
                if start - 1 >= 0:
                    rl, cl = cells[start - 1]
                    if self.board[rl, cl] == 0:
                        left_open = True
                if start + 5 < len(cells):
                    rr, cr = cells[start + 5]
                    if self.board[rr, cr] == 0:
                        right_open = True

                if cnt_x >= 5:
                    has_five = True
                    break
                elif cnt_x == 4 and cnt_empty == 1:
                    if left_open and right_open:
                        has_live_four = True
                        break
                    elif left_open or right_open:
                        has_rush_four = True
                        break
                elif cnt_x == 3 and cnt_empty == 2:
                    if len(piece_indices) == 3:
                        piece_indices.sort()
                        if (piece_indices[1] == piece_indices[0] + 1 and
                            piece_indices[2] == piece_indices[1] + 1):
                            if left_open and right_open:
                                has_live_three = True
                                break

            if has_five:
                threats['five'] += 1
            if has_live_four:
                threats['live_four'] += 1
            if has_rush_four:
                threats['rush_four'] += 1
            if has_live_three:
                threats['live_three'] += 1

        if threats['five'] > 0:
            return True
        if threats['live_four'] > 0:
            return True
        if threats['live_three'] >= 2:
            return True
        if threats['live_three'] >= 1 and threats['rush_four'] >= 1:
            return True
        if threats['rush_four'] >= 2:
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
            r, c = row + dr * step, col + dc * step
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
            r, c = row - dr * step, col - dc * step
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
        for step in range(1, max_steps + 1):
            r, c = row + dr * step, col + dc * step
            if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                break
            if self.board[r, c] == 0:
                count += 1
            else:
                break
        return count

# ==================== Zobrist哈希 ====================
class ZobristHash:
    def __init__(self, board_size=BOARD_SIZE):
        self.board_size = board_size
        self.table = np.random.randint(0, 2**63, (board_size, board_size, 3), dtype=np.uint64)
    
    def compute_hash(self, board):
        h = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                piece = board[i, j]
                if piece != 0:
                    h ^= self.table[i, j, piece]
        return h

# ==================== Q-learning AI ====================
class ZobristQLearningAI:
    def __init__(self, player, learning_rate=0.1, discount_factor=0.95, 
                exploration_rate=0.5, exploration_decay=0.995,
                max_q_size=5000000, prune_ratio=0.7):
        self.player = player
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exploration_decay
        self.lr = learning_rate
        self.attack_weight = 0.4
        self.defend_weight = 0.5
        self.attack_heuristic = 0.001
        self.defend_heuristic = 0.0005
        self.attack_reward_coef = 0.00005
        self.defend_reward_coef = 0.00005
        self.zobrist = ZobristHash()
        
        self.q_table = {}
        self.max_q_size = max_q_size
        self.prune_ratio = prune_ratio
        # 删除原来的 self.prune_keep_size
        
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.name = f"AI-{player}"
        self.total_games = 0
        
        self.defense_count = 0
        self.critical_defense_count = 0
    
    def get_state_hash(self, state):
        board_array, _ = state
        return self.zobrist.compute_hash(board_array)
    
    def get_action(self, state, valid_moves, training=True):
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)

        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}

        temp_env = Gomoku(BOARD_SIZE)
        board_array, _ = state
        temp_env.board = board_array.copy()
        opponent = 2 if self.player == 1 else 1

        threat_moves = self._detect_all_threats(temp_env, opponent)

        if threat_moves:
            forced_moves = [m for m in valid_moves if m in threat_moves]
            if forced_moves:
                valid_moves = forced_moves

        if training and random.random() < 0.2:
            attack_scores = []
            for move in valid_moves:
                attack_val, _ = temp_env.evaluate_position(move[0], move[1], self.player)
                attack_scores.append((attack_val, move))
            attack_scores.sort(reverse=True, key=lambda x: x[0])
            best_attack_moves = [m for s, m in attack_scores if s == attack_scores[0][0]]
            return random.choice(best_attack_moves)

        best_value = -float('inf')
        best_moves = []

        for move in valid_moves:
            q_att, q_def, _ = self.q_table[state_hash].get(move, [0.0, 0.0, 0])
            attack_val, defend_val = temp_env.evaluate_position(move[0], move[1], self.player)
            q_combined = self.attack_weight * q_att + self.defend_weight * q_def
            total_value = q_combined + attack_val * self.attack_heuristic + defend_val * self.defend_heuristic
            if total_value > best_value:
                best_value = total_value
                best_moves = [move]
            elif total_value == best_value:
                best_moves.append(move)

        return random.choice(best_moves) if best_moves else random.choice(valid_moves)

    def _detect_all_threats(self, env, opponent):
        threats = set()
        board = env.board
        for i in range(env.board_size):
            for j in range(env.board_size):
                if board[i, j] == 0:
                    board[i, j] = opponent
                    if env.is_opponent_threat(i, j, opponent):
                        threats.add((i, j))
                    board[i, j] = 0
        return threats

    def update(self, state, action, reward, next_state, done):
        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        if action not in self.q_table[state_hash]:
            self.q_table[state_hash][action] = [0.0, 0.0, 0]
        current = self.q_table[state_hash][action]
        current_q_att, current_q_def, access_cnt = current

        current[2] += 1

        if done:
            max_future_q_att = 0
            max_future_q_def = 0
        else:
            next_state_hash = self.get_state_hash(next_state)
            if next_state_hash in self.q_table and self.q_table[next_state_hash]:
                max_future_q_att = max([v[0] for v in self.q_table[next_state_hash].values()])
                max_future_q_def = max([v[1] for v in self.q_table[next_state_hash].values()])
            else:
                max_future_q_att = 0
                max_future_q_def = 0

        temp_env = Gomoku(BOARD_SIZE)
        board_array, _ = state
        temp_env.board = board_array.copy()
        attack_val, defend_val = temp_env.evaluate_position(action[0], action[1], self.player)

        extra_reward = 0
        if defend_val > attack_val * 1.2:
            extra_reward = defend_val * self.defend_reward_coef
        elif attack_val > defend_val * 1.2:
            extra_reward = attack_val * self.attack_reward_coef

        if reward == 1:
            adjusted_reward = 100.0
        elif reward == 0 and done:
            adjusted_reward = -1.0
        elif reward == -10:
            adjusted_reward = -20.0
        else:
            adjusted_reward = reward

        total_reward = adjusted_reward + extra_reward

        new_q_att = current_q_att + self.lr * (total_reward + self.gamma * max_future_q_att - current_q_att)
        new_q_def = current_q_def + self.lr * (total_reward + self.gamma * max_future_q_def - current_q_def)

        self.q_table[state_hash][action] = [new_q_att, new_q_def, current[2]]

        if current[2] % 1000 == 0:
            logger.debug(
                f"Q-update: state={state_hash}, action={action}, "
                f"q_att={new_q_att:.3f}, q_def={new_q_def:.3f}, access={current[2]}",
                extra={'module_tag': 'QUpdate'}
            )

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.total_games += 1
            if self.total_games % 1000 == 0 and len(self.q_table) > self.max_q_size:
                self._prune_q_table()

    def _prune_q_table(self):
        if len(self.q_table) <= self.max_q_size:
            return
        keep_size = int(self.max_q_size * self.prune_ratio)
        keep_size = max(1, keep_size)

        state_access = {h: sum(v[2] for v in actions.values()) for h, actions in self.q_table.items()}
        sorted_access = sorted(state_access.values(), reverse=True)
        total_access = sum(sorted_access)
        avg_before = total_access / len(sorted_access) if sorted_access else 0
        kept_access = sum(sorted_access[:keep_size])
        avg_after = kept_access / keep_size if keep_size > 0 else 0

        keep_states = {h for h, _ in sorted(state_access.items(), key=lambda x: x[1], reverse=True)[:keep_size]}
        self.q_table = {h: self.q_table[h] for h in keep_states}

        logger.info(
            f"Q-table pruned: {len(sorted_access)} -> {keep_size} "
            f"(ratio {self.prune_ratio:.0%}, max_cap={self.max_q_size}) | "
            f"avg_access: {avg_before:.1f} -> {avg_after:.1f}, "
            f"kept_access={kept_access}/{total_access} ({kept_access/total_access*100:.1f}%)",
            extra={'module_tag': 'Trim'}
        )

    def save(self, filename, total_training_games=None):
        if total_training_games is None:
            total_training_games = self.total_games
        save_data = {
            'q_table': self.q_table,
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
            'max_q_size': self.max_q_size,
            'prune_ratio': self.prune_ratio,
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
        logger.info(f"模型已保存: {Path(filename).name} ({size_mb:.2f} MB, Q表:{len(self.q_table)})",extra={'module_tag': 'Save'})    

    def load(self, filename):
        with open(filename, 'rb') as f:
            compressed = f.read()
        data = lz4.frame.decompress(compressed)
        save_data = pickle.loads(data)

        self.q_table = save_data['q_table']
        for state_hash in list(self.q_table.keys()):
            actions = self.q_table[state_hash]
            for action in list(actions.keys()):
                val = actions[action]
                if isinstance(val, (int, float)):
                    actions[action] = [float(val), float(val), 0]
                elif isinstance(val, list) and len(val) == 2:
                    actions[action] = val + [0]

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
        self.max_q_size = save_data.get('max_q_size', self.max_q_size)
        self.prune_ratio = save_data.get('prune_ratio', 0.7)
        self.attack_heuristic = save_data.get('attack_heuristic', 0.001)
        self.defend_heuristic = save_data.get('defend_heuristic', 0.0005)
        self.attack_reward_coef = save_data.get('attack_reward_coef', 0.00005)
        self.defend_reward_coef = save_data.get('defend_reward_coef', 0.00005)
        print(f"模型已加载: {Path(filename).name}")

# ==================== AI训练器 ====================
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
            'defense_count': [],
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
        game_memory = []
        move_count = 0
        
        defense_this_game = 0
        
        while not done and move_count < MAX_MOVES:
            move_count += 1
            
            current_ai = self.ai1 if env.current_player == 1 else self.ai2
            valid_moves = env.get_valid_moves()
            
            opponent = 2 if env.current_player == 1 else 1
            urgent_moves = self._detect_urgent_defense(env, opponent, current_ai)
            
            if urgent_moves:
                defense_this_game += 1
            
            action = current_ai.get_action(state, valid_moves, training=training)
            
            success, next_state, reward, done = env.make_move(action)
            
            if not success:
                current_ai.update(state, action, -10, next_state, done)
                break
            
            game_memory.append((state, action, reward, next_state, done, current_ai))
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
            
            self._backpropagate(game_memory)
            
            if current_ai == self.ai1:
                self.ai1.defense_count += defense_this_game
            else:
                self.ai2.defense_count += defense_this_game
            
            self.recent_results.append(result)
            if len(self.recent_results) > 100:
                self.recent_results.pop(0)
            
            # ----- 记录每局详情 -----
            logger.debug(
                f"Game: {result} | moves={move_count} | defense={defense_this_game} | "
                f"winner={'AI1' if result=='ai1_win' else ('AI2' if result=='ai2_win' else 'Draw')}",
                extra={'module_tag': 'Game'}
            )
        
        return result, first_player

    def _detect_urgent_defense(self, env, opponent, ai):
        return list(ai._detect_all_threats(env, opponent))
    
    def _backpropagate(self, game_memory):
        for state, action, reward, next_state, done, ai in reversed(game_memory):
            ai.update(state, action, reward, next_state, done)
    
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
        
        balance_interval = 100
        
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
                if len(self.recent_results) >= 100:
                    recent_wins1 = self.recent_results.count('ai1_win')
                    recent_wins2 = self.recent_results.count('ai2_win')
                    gap = abs(recent_wins1 - recent_wins2) / len(self.recent_results) * 100
                else:
                    total_games = self.ai1.wins + self.ai1.losses + self.ai1.draws
                    gap = abs(self.ai1.wins - self.ai2.wins) / total_games * 100 if total_games > 0 else 0
                logger.info(f"进度: {episode}/{episodes} 速度:{speed:.1f}局/秒 胜率差距:{gap:.1f}% "
                            f"ε1={self.ai1.epsilon:.3f} ε2={self.ai2.epsilon:.3f}",extra={'module_tag': 'Train'})
                # ----- 新增统计 -----
                q1_size = len(self.ai1.q_table)
                q2_size = len(self.ai2.q_table)
                total_states = q1_size + q2_size
                if total_states > 0:
                    total_access = sum(sum(v[2] for v in act.values()) for act in self.ai1.q_table.values()) + \
                                sum(sum(v[2] for v in act.values()) for act in self.ai2.q_table.values())
                    avg_access = total_access / total_states
                else:
                    avg_access = 0
                logger.info(
                    f"Q-table stats: total_states={total_states} (AI1={q1_size}, AI2={q2_size}), "
                    f"avg_access={avg_access:.1f}",
                    extra={'module_tag': 'Stats'}
                )
            
            if episode % save_every == 0 and episode != episodes:
                elapsed = time.time() - start_time
                speed = episode / elapsed
                logger.info(f"检查点保存: 第{episode}局 速度:{speed:.1f}局/秒",extra={'module_tag': 'Save'})
                
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

        if hasattr(self, 'last_draw_rate') and abs(draw_rate - self.last_draw_rate) < 2:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0
        self.last_draw_rate = draw_rate

        target_low = 20
        target_high = 40

        if draw_rate > target_high:
            factor = 1.0 + (draw_rate - target_high) / 100
            self.global_aggressiveness = min(3.0, self.global_aggressiveness * factor)
        elif draw_rate < target_low:
            factor = 1.0 - (target_low - draw_rate) / 200
            self.global_aggressiveness = max(0.5, self.global_aggressiveness * factor)
        else:
            self.global_aggressiveness = max(0.8, min(1.2, self.global_aggressiveness * 0.99))

        if draw_rate > 80 and self.no_progress_counter >= 10:
            self._force_intervention()
            self.no_progress_counter = 0
            return

        if rate1 > rate2:
            strong_ai, weak_ai = self.ai1, self.ai2
            strong_name, weak_name = "AI1(强)", "AI2(弱)"
        else:
            strong_ai, weak_ai = self.ai2, self.ai1
            strong_name, weak_name = "AI2(强)", "AI1(弱)"

        # 保存旧值（仅用于日志记录，但为精简只记调整前的重要参数）
        old_strong_eps = strong_ai.epsilon
        old_weak_eps = weak_ai.epsilon
        old_strong_lr = strong_ai.lr
        old_weak_lr = weak_ai.lr

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

        MIN_WEIGHT, MAX_WEIGHT = 0.05, 0.95
        MIN_HEUR, MAX_HEUR = 0.0001, 0.01

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

        for ai in [strong_ai, weak_ai]:
            ai.epsilon = max(0.001, min(0.5, ai.epsilon))
            ai.lr = max(0.01, min(0.3, ai.lr))
            ai.attack_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, ai.attack_weight))
            ai.defend_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, ai.defend_weight))
            ai.attack_heuristic = max(MIN_HEUR, min(MAX_HEUR, ai.attack_heuristic))
            ai.defend_heuristic = max(MIN_HEUR, min(MAX_HEUR, ai.defend_heuristic))

        logger.info(
            f"Adjust #{self.balance_count}: gap={gap:.1f}% draw={draw_rate:.1f}% factor={self.global_aggressiveness:.2f} | "
            f"strong ε {old_strong_eps:.3f}→{strong_ai.epsilon:.3f} lr {old_strong_lr:.3f}→{strong_ai.lr:.3f} | "
            f"weak ε {old_weak_eps:.3f}→{weak_ai.epsilon:.3f} lr {old_weak_lr:.3f}→{weak_ai.lr:.3f}",
            extra={'module_tag': 'Balance'}
        )

    def _force_intervention(self):
        logger.warning("执行强制干预策略",extra={'module_tag': 'Force'})

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
            strong_ai, weak_ai = self.ai1, self.ai2
            strong_name, weak_name = "AI1", "AI2"

        old_strong_eps = strong_ai.epsilon
        old_weak_eps = weak_ai.epsilon
        old_strong_lr = strong_ai.lr
        old_weak_lr = weak_ai.lr

        strong_ai.epsilon = max(0.002, min(0.05, strong_ai.epsilon * 0.3))
        strong_ai.lr = min(0.25, strong_ai.lr * 1.3)

        weak_ai.epsilon = min(0.4, weak_ai.epsilon * 2.5)
        weak_ai.lr = min(0.3, weak_ai.lr * 1.5)

        factor = self.global_aggressiveness
        MIN_WEIGHT, MAX_WEIGHT = 0.05, 0.95
        base_attack = 0.2
        base_defend = 0.7
        new_attack = min(MAX_WEIGHT, base_attack * factor)
        new_defend = max(MIN_WEIGHT, base_defend / factor)

        strong_ai.attack_weight = new_attack
        strong_ai.defend_weight = new_defend
        weak_ai.attack_weight = new_attack
        weak_ai.defend_weight = new_defend

        strong_ai.attack_heuristic = min(0.01, 0.001 * factor)
        strong_ai.defend_heuristic = max(0.0001, 0.0005 / factor)
        weak_ai.attack_heuristic = min(0.01, 0.001 * factor)
        weak_ai.defend_heuristic = max(0.0001, 0.0005 / factor)

        logger.warning(f"强制调整: {strong_name} ε {old_strong_eps:.3f}→{strong_ai.epsilon:.3f}, "
                    f"{weak_name} ε {old_weak_eps:.3f}→{weak_ai.epsilon:.3f}",extra={'module_tag': 'Force'})
    
    def plot_stats(self, episode):
        if not self.stats['ai1_wins']:
            return

        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            n_points = len(self.stats['ai1_wins'])
            x_total = np.arange(1, n_points + 1) * 10 + self.start_episode

            # ---------- Subplot 1: Cumulative Reward Comparison ----------
            ai1_wins = np.array(self.stats['ai1_wins'])
            ai2_wins = np.array(self.stats['ai2_wins'])
            draws = np.array(self.stats['draws'])
            total_games = ai1_wins + ai2_wins + draws

            cum_reward1 = 2 * ai1_wins + draws - total_games   # +1 win, -1 loss, 0 draw
            cum_reward2 = 2 * ai2_wins + draws - total_games

            window = min(100, n_points)
            if window > 0:
                ma1 = np.convolve(cum_reward1, np.ones(window)/window, mode='valid')
                ma2 = np.convolve(cum_reward2, np.ones(window)/window, mode='valid')
                x_ma = x_total[window-1:]
            else:
                ma1, ma2 = cum_reward1, cum_reward2
                x_ma = x_total

            axes[0, 0].plot(x_total, cum_reward1, label='AI1 Cumulative Reward', color='#2c3e50', alpha=0.5, linewidth=1.5)
            axes[0, 0].plot(x_total, cum_reward2, label='AI2 Cumulative Reward', color='#7f8c8d', alpha=0.5, linewidth=1.5)
            axes[0, 0].plot(x_ma, ma1, label=f'AI1 {window}-episode MA', color='#2c3e50', linewidth=2.5)
            axes[0, 0].plot(x_ma, ma2, label=f'AI2 {window}-episode MA', color='#7f8c8d', linewidth=2.5)
            axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            axes[0, 0].set_xlabel('Total Episodes')
            axes[0, 0].set_ylabel('Cumulative Reward')
            axes[0, 0].set_title('AI1 vs AI2 Cumulative Reward (+1 win, -1 loss, 0 draw)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # ---------- Subplot 2: Win Rates and Gap ----------
            total_safe = np.where(total_games == 0, 1, total_games)
            rate1 = ai1_wins / total_safe * 100
            rate2 = ai2_wins / total_safe * 100
            gap = np.array(self.stats['gap'])

            axes[0, 1].plot(x_total, rate1, label='AI1 Win Rate', color='#2c3e50')
            axes[0, 1].plot(x_total, rate2, label='AI2 Win Rate', color='#7f8c8d')
            axes[0, 1].plot(x_total, gap, label='Win Rate Gap', color='red', linestyle='--', alpha=0.7)
            axes[0, 1].axhline(y=50, color='red', linestyle=':', alpha=0.5)
            axes[0, 1].set_xlabel('Total Episodes')
            axes[0, 1].set_ylabel('Win Rate / Gap (%)')
            axes[0, 1].set_title('Win Rate Trend with Gap')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 100)

            # ---------- Subplot 3: Exploration Rate ----------
            axes[0, 2].plot(x_total, self.stats['ai1_epsilon'], label='AI1', color='#2c3e50')
            axes[0, 2].plot(x_total, self.stats['ai2_epsilon'], label='AI2', color='#7f8c8d')
            axes[0, 2].set_xlabel('Total Episodes')
            axes[0, 2].set_ylabel('Epsilon')
            axes[0, 2].set_title('Exploration Rate')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # ---------- Subplot 4: Learning Rate ----------
            axes[1, 0].plot(x_total, self.stats['ai1_lr'], label='AI1', color='#2c3e50')
            axes[1, 0].plot(x_total, self.stats['ai2_lr'], label='AI2', color='#7f8c8d')
            axes[1, 0].set_xlabel('Total Episodes')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # ---------- Subplot 5: Attack Weight ----------
            axes[1, 1].plot(x_total, self.stats['ai1_attack_weight'], label='AI1', color='#2c3e50')
            axes[1, 1].plot(x_total, self.stats['ai2_attack_weight'], label='AI2', color='#7f8c8d')
            axes[1, 1].set_xlabel('Total Episodes')
            axes[1, 1].set_ylabel('Attack Weight')
            axes[1, 1].set_title('Attack Weight')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 1)

            # ---------- Subplot 6: Defend Weight ----------
            axes[1, 2].plot(x_total, self.stats['ai1_defend_weight'], label='AI1', color='#2c3e50')
            axes[1, 2].plot(x_total, self.stats['ai2_defend_weight'], label='AI2', color='#7f8c8d')
            axes[1, 2].set_xlabel('Total Episodes')
            axes[1, 2].set_ylabel('Defend Weight')
            axes[1, 2].set_title('Defend Weight')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim(0, 1)

            plt.tight_layout()
            total_episodes = self.start_episode + episode
            filename = FIGURES_DIR / f"stats_ep{total_episodes}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Stats figure saved: {filename.name}", extra={'module_tag': 'Save'})
        except Exception as e:
            logger.error(f"Failed to plot stats: {e}", extra={'module_tag': 'Save'})

# ==================== 工具函数 ====================
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
                win_rate = wins / total * 100 if total > 0 else 0
                q_size = len(save_data.get('q_table', {}))
                defense = save_data.get('defense_count', 0)

                print(f"{i+1}. {model.name} - {wins}胜 {losses}负 {draws}平 "
                      f"(胜率 {win_rate:.1f}%, 防守{defense}次, Q表{q_size})")
                valid_models.append(str(model))
        except Exception as e:
            print(f"  {model.name} - 无法读取，已跳过: {e}")
    if not valid_models:
        print("没有可用的模型文件")
    return valid_models

# ==================== 主函数 ====================
def main():
    print("\n" + "=" * 60)
    print("五子棋AI训练系统 v5.3")
    print("=" * 60)
    
    models = list_all_models()
    
    if models:
        print("\n请选择两个AI进行训练：(输入0表示创建新AI)")
        while True:
            try:
                print("请选择黑棋AI (player 1):")
                choice1 = int(input("输入序号: ")) - 1
                if choice1 == -1:
                    model1 = None
                    print("选择黑棋: 新AI")
                    break
                elif 0 <= choice1 < len(models):
                    model1 = models[choice1]
                    print(f"选择黑棋: {Path(model1).name}")
                    break
                else:
                    print(f"无效序号，请输入 0 到 {len(models)} 之间的数字")
            except ValueError:
                print("请输入有效数字")
        while True:
            try:
                print("\n请选择白棋AI (player 2):")
                choice2 = int(input("输入序号: ")) - 1
                if choice2 == -1:
                    model2 = None
                    print("选择白棋: 新AI")
                    break
                elif 0 <= choice2 < len(models):
                    model2 = models[choice2]
                    print(f"选择白棋: {Path(model2).name}")
                    break
                else:
                    print(f"无效序号，请输入 0 到 {len(models)} 之间的数字")
            except ValueError:
                print("请输入有效数字")
    else:
        print("\n创建新AI...")
        model1 = model2 = None

    ai_black = ZobristQLearningAI(player=1, learning_rate=0.1, exploration_rate=0.4)
    ai_white = ZobristQLearningAI(player=2, learning_rate=0.1, exploration_rate=0.4)
    
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
    
    while True:
        try:
            episodes = input("请输入训练局数 (默认10000): ")
            episodes = int(episodes) if episodes.strip() else 10000
            break
        except ValueError:
            print("请输入有效整数")
    while True:
        try:
            save_every = input("保存间隔 (默认5000): ")
            save_every = int(save_every) if save_every.strip() else 5000
            break
        except ValueError:
            print("请输入有效整数")
    while True:
        try:
            log_every = input("日志记录间隔 (默认500): ")
            log_every = int(log_every) if log_every.strip() else 500
            break
        except ValueError:
            print("请输入有效整数")
    
    print(f"\n训练配置:")
    print(f"  总局数: {episodes}")
    print(f"  已有训练: {start_episode} 局")
    print(f"  保存间隔: 每 {save_every} 局")
    print(f"  日志记录: 每 {log_every} 局")
    print(f"  平衡调整: 每100局动态调整")
    print(f"  模型目录: {MODELS_DIR}")
    print(f"  日志目录: {LOGS_DIR}")
    print(f"  图表目录: {FIGURES_DIR}")
    
    input("\n按Enter开始训练...")
    
    trainer.train(
        episodes=episodes,
        save_every=save_every,
        log_every=log_every
    )
    
    print("\n" + "=" * 60)
    print("训练完成！最终统计：")
    print("=" * 60)
    
    total = ai_black.wins + ai_black.losses + ai_black.draws
    print(f"{ai_black.name}: {ai_black.wins}胜 {ai_black.losses}负 {ai_black.draws}平 "
          f"(胜率 {ai_black.wins/total*100:.1f}%)")
    print(f"{ai_white.name}: {ai_white.wins}胜 {ai_white.losses}负 {ai_white.draws}平 "
          f"(胜率 {ai_white.wins/total*100:.1f}%)")
    
    total_defense = ai_black.defense_count + ai_white.defense_count
    total_critical = ai_black.critical_defense_count + ai_white.critical_defense_count
    print(f"\n🛡️ 防守统计:")
    print(f"  总防守次数: {total_defense}")
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