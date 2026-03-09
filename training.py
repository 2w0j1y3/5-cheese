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

# ==================== 进度显示 ====================
class ProgressBar:
    """像pip一样的进度条 - 只显示进度和动态数据"""
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
        bar = '█' * filled + '─' * (self.length - filled)
        
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
    
    # 保留 evaluate_position 和 _evaluate_direction 供外部可能使用（如测试脚本），但 AI 不再调用它们
    def evaluate_position(self, row, col, player):
        """高质量评估函数（保留供外部使用）"""
        score = 0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            attack_score = self._evaluate_direction(row, col, player, dr, dc)
            opponent = 2 if player == 1 else 1
            defend_score = self._evaluate_direction(row, col, opponent, dr, dc) * 1.8
            score += attack_score + defend_score
        center = self.board_size // 2
        center_dist = abs(row - center) + abs(col - center)
        score += (self.board_size - center_dist) * 2
        nearby = self.get_nearby_pieces(row, col, 2)
        score += nearby * 10
        return score
    
    def _evaluate_direction(self, row, col, player, dr, dc):
        """评估单一方向（保留供外部使用）"""
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
        
        if count >= 5:
            return 1000000
        elif count == 4:
            if not (left_block and right_block):
                if player != self.current_player:
                    return 500000
                return 200000
            return 20000
        elif count == 3:
            if not (left_block and right_block):
                if player != self.current_player:
                    return 100000
                return 50000
            return 5000
        elif count == 2:
            if not (left_block and right_block):
                if player != self.current_player:
                    return 2000
                return 1000
            return 200
        elif count == 1:
            return 10
        
        if not left_block:
            score += left_space * 50
        if not right_block:
            score += right_space * 50
        return 0
    
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

# ==================== 简化版 Q-learning AI (不区分进攻/防守) ====================
class ZobristQLearningAI:
    def __init__(self, player, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995):
        self.player = player
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_min = 0.01
        self.epsilon_decay = exploration_decay
        self.lr = learning_rate
        self.zobrist = ZobristHash()
        self.q_table = {}
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.name = f"AI-{player}"
        self.total_games = 0
        
        # 防守统计（保留以兼容测试脚本，但不影响算法）
        self.defense_count = 0
        self.critical_defense_count = 0
    
    def get_state_hash(self, state):
        board_array, _ = state
        return self.zobrist.compute_hash(board_array)
    
    def get_action(self, state, valid_moves, training=True):
        """选择动作：ε-greedy 策略，仅基于 Q 值"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        
        q_values = self.q_table[state_hash]
        
        best_value = -float('inf')
        best_moves = []
        for move in valid_moves:
            q = q_values.get(move, 0.0)
            if q > best_value:
                best_value = q
                best_moves = [move]
            elif q == best_value:
                best_moves.append(move)
        
        return random.choice(best_moves) if best_moves else random.choice(valid_moves)
    
    def update(self, state, action, reward, next_state, done):
        """Q-learning 更新"""
        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {}
        
        current_q = self.q_table[state_hash].get(action, 0.0)
        
        # 计算下一状态的最大 Q 值
        if done:
            max_future_q = 0
        else:
            next_state_hash = self.get_state_hash(next_state)
            if next_state_hash in self.q_table:
                max_future_q = max(self.q_table[next_state_hash].values(), default=0.0)
            else:
                max_future_q = 0.0
        
        # 奖励调整（与原始程序一致）
        if reward == 1:
            adjusted_reward = 10.0
        elif reward == 0 and done:
            adjusted_reward = -0.2
        elif reward == -10:
            adjusted_reward = -10.0
        else:
            adjusted_reward = reward
        
        # Q-learning 更新公式
        new_q = current_q + self.lr * (adjusted_reward + self.gamma * max_future_q - current_q)
        self.q_table[state_hash][action] = new_q
        
        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.total_games += 1
            
            # 定期修剪 Q 表
            if self.total_games % 10000 == 0 and len(self.q_table) > 10000000:
                self._prune_q_table()
    
    def _prune_q_table(self):
        if len(self.q_table) <= 5000000:
            return
        items = list(self.q_table.items())
        random.shuffle(items)
        keep_count = 5000000
        self.q_table = dict(items[:keep_count])
        logger.info(f"Q表已修剪: {len(items)} -> {keep_count}")
    
    def save(self, filename):
        save_data = {
            'q_table': self.q_table,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_games': self.total_games,
            'epsilon': self.epsilon,
            'defense_count': self.defense_count,
            'critical_defense_count': self.critical_defense_count,
            'lr': self.lr,
        }
        data = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = lz4.frame.compress(data)
        with open(filename, 'wb') as f:
            f.write(compressed)
        size_mb = len(compressed) / (1024 * 1024)
        logger.info(f"模型已保存: {Path(filename).name} ({size_mb:.2f} MB, Q表:{len(self.q_table)})")
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            compressed = f.read()
        data = lz4.frame.decompress(compressed)
        save_data = pickle.loads(data)
        
        self.q_table = save_data['q_table']
        self.wins = save_data.get('wins', 0)
        self.losses = save_data.get('losses', 0)
        self.draws = save_data.get('draws', 0)
        self.total_games = save_data.get('total_games', 0)
        self.epsilon = save_data.get('epsilon', self.epsilon)
        self.defense_count = save_data.get('defense_count', 0)
        self.critical_defense_count = save_data.get('critical_defense_count', 0)
        self.lr = save_data.get('lr', self.lr)
        print(f"模型已加载: {Path(filename).name}")

# ==================== AI训练器（完全保留原始训练逻辑）====================
class AITrainer:
    def __init__(self, ai1, ai2, board_size=BOARD_SIZE):
        self.ai1 = ai1
        self.ai2 = ai2
        self.board_size = board_size
        self.start_episode = 0
        
        # 统计数据
        self.stats = {
            'ai1_wins': [], 'ai2_wins': [], 'draws': [],
            'ai1_epsilon': [], 'ai2_epsilon': [],
            'ai1_lr': [], 'ai2_lr': [],
            'defense_count': [], 'critical_defense': [],
            'gap': []
        }
        
        self.game_details = []
        self.recent_results = []
        
        # 平衡控制
        self.balance_count = 0
        self.last_balance_check = 0
    
    def play_game(self, first_player=None, training=True):
        """进行一局游戏"""
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
        critical_defense_this_game = 0
        
        while not done and move_count < MAX_MOVES:
            move_count += 1
            
            current_ai = self.ai1 if env.current_player == 1 else self.ai2
            valid_moves = env.get_valid_moves()
            
            # 检测对手威胁（仅用于统计，不影响决策）
            opponent = 2 if env.current_player == 1 else 1
            urgent_moves = self._detect_urgent_defense(env, opponent)
            
            if urgent_moves:
                defense_this_game += 1
                if len(urgent_moves) > 0:
                    critical_defense_this_game += 1
            
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
            
            # 反向传播
            self._backpropagate(game_memory)
            
            # 记录防守统计
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
    
    def _detect_urgent_defense(self, env, opponent):
        """检测紧急防守位置（仅用于统计）"""
        urgent = []
        for i in range(env.board_size):
            for j in range(env.board_size):
                if env.board[i, j] == 0:
                    threat = env.evaluate_position(i, j, opponent)
                    if threat >= 80000:
                        urgent.append((i, j))
        return urgent
    
    def _backpropagate(self, game_memory):
        future_reward = 0
        for i in range(len(game_memory) - 1, -1, -1):
            state, action, reward, next_state, done, ai = game_memory[i]
            ai.update(state, action, reward + future_reward, next_state, done)
            future_reward = reward * 0.95
    
    def train(self, episodes=100000, save_every=10000, log_every=500):
        """训练主循环（与原始完全一致）"""
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
                self.stats['critical_defense'].append(self.ai1.critical_defense_count + self.ai2.critical_defense_count)
                
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
                self.ai1.save(MODELS_DIR / f"AI1_checkpoint_{total_episodes}_{timestamp}.pkl")
                self.ai2.save(MODELS_DIR / f"AI2_checkpoint_{total_episodes}_{timestamp}.pkl")
                
                self.plot_stats(episode)
            
            # 进度条更新
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
        self.ai1.save(MODELS_DIR / f"AI1_final_{total_episodes}_{timestamp}.pkl")
        self.ai2.save(MODELS_DIR / f"AI2_final_{total_episodes}_{timestamp}.pkl")
        self.plot_stats(episodes)
    
    def _balance_ai(self):
        """平衡两个AI的实力（与原始一致，但启发式权重已移除，此处仅调整ε和lr）"""
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
        logger.info(f"\n⚖️ 调整 #{self.balance_count}: 差距={gap:.1f}% 平局={draw_rate:.1f}%")
        
        if gap < 20 and 30 < draw_rate < 40:
            logger.info(f"✅ 已达到理想状态！停止调整")
            return
        
        if rate1 > rate2:
            strong_ai, weak_ai = self.ai1, self.ai2
            strong_name, weak_name = "AI1(强)", "AI2(弱)"
        else:
            strong_ai, weak_ai = self.ai2, self.ai1
            strong_name, weak_name = "AI2(强)", "AI1(弱)"
        
        old_strong_eps = strong_ai.epsilon
        old_weak_eps = weak_ai.epsilon
        old_strong_lr = strong_ai.lr
        old_weak_lr = weak_ai.lr
        
        if gap > 40:
            strong_ai.epsilon = max(0.005, min(0.01, strong_ai.epsilon * 0.9))
            strong_ai.lr = min(0.25, strong_ai.lr * 1.1)
            weak_ai.epsilon = max(0.06, min(0.09, weak_ai.epsilon * 0.95))
            weak_ai.lr = min(0.30, weak_ai.lr * 1.1)
            level = "差距>40%"
        elif gap > 25:
            strong_ai.lr = min(0.23, strong_ai.lr * 1.05)
            weak_ai.lr = min(0.28, weak_ai.lr * 1.05)
            level = "差距>25%"
        else:
            level = "微调"
        
        # 由于移除了启发式权重，平局率调整仅作用于探索率（可选）
        if draw_rate > 40:
            weak_ai.epsilon = min(0.4, weak_ai.epsilon * 1.1)
            strong_ai.epsilon = max(0.01, strong_ai.epsilon * 0.95)
            draw_action = "鼓励探索"
        elif draw_rate < 20:
            weak_ai.epsilon = max(0.05, weak_ai.epsilon * 0.95)
            strong_ai.epsilon = min(0.2, strong_ai.epsilon * 1.05)
            draw_action = "减少探索"
        else:
            draw_action = "保持"
        
        logger.info(f"\n⚖️ 平衡调整 #{self.balance_count}: {level} 差距={gap:.1f}% 平局率={draw_rate:.1f}% {draw_action}")
        logger.info(f"  {strong_name}: ε {old_strong_eps:.3f}→{strong_ai.epsilon:.3f}, lr {old_strong_lr:.3f}→{strong_ai.lr:.3f}")
        logger.info(f"  {weak_name}: ε {old_weak_eps:.3f}→{weak_ai.epsilon:.3f}, lr {old_weak_lr:.3f}→{weak_ai.lr:.3f}")
    
    def plot_stats(self, episode):
        """绘制统计图（与原始一致）"""
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
            rate1 = np.array(self.stats['ai1_wins']) / total * 100
            rate2 = np.array(self.stats['ai2_wins']) / total * 100
            axes[0, 1].plot(episodes, rate1, label='AI1', color='#2c3e50')
            axes[0, 1].plot(episodes, rate2, label='AI2', color='#7f8c8d')
            axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].set_xlabel('Episodes (x10)')
            axes[0, 1].set_ylabel('Win Rate (%)')
            axes[0, 1].set_title('Win Rate Trend')
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
            
            axes[1, 1].plot(episodes, self.stats['defense_count'], color='#27ae60')
            axes[1, 1].set_xlabel('Episodes (x10)')
            axes[1, 1].set_ylabel('Defense Count')
            axes[1, 1].set_title('Total Defenses')
            axes[1, 1].grid(True, alpha=0.3)
            
            axes[1, 2].plot(episodes, self.stats['gap'], color='#e67e22')
            axes[1, 2].axhline(y=15, color='red', linestyle='--', alpha=0.5, label='平衡目标')
            axes[1, 2].set_xlabel('Episodes (x10)')
            axes[1, 2].set_ylabel('Gap (%)')
            axes[1, 2].set_title('Win Rate Gap')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].set_ylim(0, 100)
            
            plt.tight_layout()
            filename = FIGURES_DIR / f"stats_ep{episode}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"  统计图已保存: {filename.name}")
            
        except Exception as e:
            logger.info(f"  绘制统计图失败: {e}")

# ==================== 工具函数 ====================
def list_all_models():
    """列出所有可用模型"""
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

# ==================== 主函数 ====================
def main():
    print("\n" + "=" * 60)
    print("五子棋AI训练系统 v4.0 (简化版 - 无进攻/防守启发式)")
    print("=" * 60)
    
    models = list_all_models()
    
    if models:
        print("\n请选择两个AI进行训练：")
        print("请选择黑棋AI (player 1):")
        choice1 = int(input("输入序号: ")) - 1
        print("\n请选择白棋AI (player 2):")
        choice2 = int(input("输入序号: ")) - 1
        
        model1 = models[choice1]
        model2 = models[choice2]
        
        print(f"\n选择黑棋: {Path(model1).name}")
        print(f"选择白棋: {Path(model2).name}")
        
        ai_black = ZobristQLearningAI(player=1, learning_rate=0.1, exploration_rate=0.1)
        ai_white = ZobristQLearningAI(player=2, learning_rate=0.1, exploration_rate=0.1)
        
        ai_black.load(model1)
        ai_white.load(model2)

        start_episode = max(ai_black.total_games, ai_white.total_games)
    else:
        print("\n创建新AI...")
        ai_black = ZobristQLearningAI(player=1, learning_rate=0.15, exploration_rate=0.3)
        ai_white = ZobristQLearningAI(player=2, learning_rate=0.15, exploration_rate=0.3)
        start_episode = 0

    trainer = AITrainer(ai_black, ai_white)
    trainer.start_episode = start_episode
    
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    
    episodes = input("请输入训练局数 (默认100000): ")
    episodes = int(episodes) if episodes.strip() else 100000
    
    save_every = input("保存间隔 (默认10000): ")
    save_every = int(save_every) if save_every.strip() else 10000
    
    log_every = input("日志记录间隔 (默认500): ")
    log_every = int(log_every) if log_every.strip() else 500
    
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