"""
精确评估五子棋AI模型（修正版）
指标：决策稳定性、搜索效率、战术能力、Elo等级分
"""

import sys
import os
import random
import time
import pickle
import lz4.frame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict
import torch
import importlib.util
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 动态导入带点号的模块 ====================
def import_module_from_file(filepath: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

v4_module = import_module_from_file(Path("training_v4.2.py"), "v4")
v5_module = import_module_from_file(Path("training_v5.2.py"), "v5")
v6_module = import_module_from_file(Path("training_v6.1.py"), "v6")
v7_module = import_module_from_file(Path("training_v7.0.py"), "v7")

QLearningV4 = v4_module.ZobristQLearningAI
QLearningV5 = v5_module.ZobristQLearningAI
DQNAI = v6_module.DQNAI
if hasattr(v7_module, 'DDQNAI'):
    DDQNAI = v7_module.DDQNAI
else:
    print("警告: training_v7.0.py 中未找到 DDQNAI，使用 DQNAI 代替")
    DDQNAI = DQNAI

# ==================== 常量配置 ====================
BOARD_SIZE = 9
MODELS_DIR = Path("models")
FIGURES_DIR = Path("figures")
TABLES_DIR = Path("tables")
for d in [FIGURES_DIR, TABLES_DIR]:
    d.mkdir(exist_ok=True)

MODEL_MAP = {
    "qlearn.pkl": ("Q-learning v4.2", QLearningV4),
    "dqt.pkl": ("DQT(Q-learning+Heuristic) v5.2", QLearningV5),
    "dqn.pkl": ("DQN v6.1", DQNAI),
    "ddqn.pkl": ("Double DQN v7.0", DDQNAI),
}

# ==================== 游戏环境 ====================
class GomokuEnv:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = 1

    def reset(self, board=None, player=1):
        if board is not None:
            self.board = board.copy()
        else:
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.current_player = player

    def get_valid_moves(self):
        return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if self.board[i, j] == 0]

    def make_move(self, pos):
        i, j = pos
        if self.board[i, j] != 0:
            return False, -10, True
        self.board[i, j] = self.current_player
        if self.check_win(i, j):
            return True, 1, True
        if len(self.get_valid_moves()) == 0:
            return True, 0, True
        self.current_player = 3 - self.current_player
        return True, 0, False

    def check_win(self, row, col):
        player = self.board[row, col]
        dirs = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in dirs:
            cnt = 1
            for step in range(1,5):
                r, c = row+dr*step, col+dc*step
                if not (0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE): break
                if self.board[r,c]==player: cnt+=1
                else: break
            for step in range(1,5):
                r, c = row-dr*step, col-dc*step
                if not (0<=r<BOARD_SIZE and 0<=c<BOARD_SIZE): break
                if self.board[r,c]==player: cnt+=1
                else: break
            if cnt>=5:
                return True
        return False

# ==================== 加载AI（统一接口） ====================
def load_ai(model_path: Path, ai_class):
    ai = ai_class(player=1)
    if hasattr(ai, 'epsilon'):
        ai.epsilon = 0.0  # 确定性决策
    ai.load(str(model_path))
    if hasattr(ai, 'policy_net'):
        ai.policy_net.eval()
    # 统一 get_action 接口
    original_get_action = ai.get_action
    def wrapped_get_action(state, valid_moves, training=False):
        return original_get_action(state, valid_moves, training=training)
    ai.get_action = wrapped_get_action
    return ai

def create_random_state(num_pieces=10):
    """生成随机棋盘（保证合法）"""
    env = GomokuEnv()
    env.reset()
    for _ in range(num_pieces):
        valid = env.get_valid_moves()
        if not valid:
            break
        move = random.choice(valid)
        env.make_move(move)
    return env.board.copy(), env.current_player

# ==================== 1. 决策稳定性 ====================
def evaluate_decision_stability(ai, num_states=20, repeat=200):
    consistencies = []
    entropies = []
    for _ in range(num_states):
        board, turn = create_random_state(random.randint(5, 15))
        state = (board, turn)
        env = GomokuEnv()
        env.reset(board, turn)
        valid = env.get_valid_moves()
        if len(valid) == 0:
            continue
        actions = []
        for _ in range(repeat):
            act = ai.get_action(state, valid, training=False)
            actions.append(act)
        cnt = Counter(actions)
        max_freq = cnt.most_common(1)[0][1]
        consistency = max_freq / repeat * 100
        probs = np.array([c/repeat for c in cnt.values()])
        entropy = -np.sum(probs * np.log2(probs+1e-12))
        consistencies.append(consistency)
        entropies.append(entropy)
    return np.mean(consistencies), np.mean(entropies)

# ==================== 2. 搜索效率 ====================
def evaluate_search_efficiency(ai, num_decisions=2000):
    # 预热
    for _ in range(100):
        board, turn = create_random_state(5)
        env = GomokuEnv()
        env.reset(board, turn)
        valid = env.get_valid_moves()
        if valid:
            ai.get_action((board, turn), valid, training=False)
    times = []
    for _ in range(num_decisions):
        board, turn = create_random_state(random.randint(5, 15))
        env = GomokuEnv()
        env.reset(board, turn)
        valid = env.get_valid_moves()
        if not valid:
            continue
        start = time.perf_counter()
        ai.get_action((board, turn), valid, training=False)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)
    # 使用中位数去除极端值
    avg_time = np.median(times)
    speed = 1000.0 / avg_time if avg_time > 0 else float('inf')
    return avg_time, speed

# ==================== 3. 战术能力（200个题目） ====================
def generate_threat_templates():
    templates = []
    c = BOARD_SIZE // 2
    # 1. 冲四
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for i in range(4):
        board[c, c-2+i] = 1
    templates.append((board, 1, 1))
    # 2. 活三
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for i in range(3):
        board[c, c-1+i] = 1
    templates.append((board, 1, 2))
    # 3. 双活三
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[c, c-1:c+2] = 1
    board[c-1, c] = 1
    board[c+1, c] = 1
    templates.append((board, 1, 1))
    # 4. 冲四活三
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for i in range(4):
        board[c-2, c-2+i] = 1
    board[c, c-1:c+2] = 1
    templates.append((board, 1, 1))
    # 5. 双冲四
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[c-2, c-2:c+2] = 1
    for i in range(4):
        board[c-2+i, c+2] = 1
    templates.append((board, 1, 1))
    return templates

def expand_templates(templates, num_total=200):
    expanded = []
    for board, player, steps in templates:
        # 8种对称变换
        variants = []
        for k in range(4):
            rot = np.rot90(board, k)
            variants.append(rot)
            variants.append(np.fliplr(rot))
        unique = []
        for v in variants:
            if not any(np.array_equal(v, u) for u in unique):
                unique.append(v)
        for v in unique:
            # 随机平移
            max_shift = BOARD_SIZE - v.shape[0]
            if max_shift >= 0:
                sr = random.randint(0, max_shift)
                sc = random.randint(0, max_shift)
                shifted = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
                shifted[sr:sr+v.shape[0], sc:sc+v.shape[1]] = v
                expanded.append((shifted, player, steps))
    while len(expanded) < num_total:
        expanded.append(random.choice(expanded))
    random.shuffle(expanded)
    return expanded[:num_total]

def evaluate_tactical(ai, problems):
    solved = 0
    for board, player, min_steps in tqdm(problems, desc="Tactical"):
        env = GomokuEnv()
        env.reset(board, player)
        win = False
        for _ in range(min_steps + 2):
            valid = env.get_valid_moves()
            if not valid:
                break
            act = ai.get_action((env.board.copy(), env.current_player), valid, training=False)
            _, reward, done = env.make_move(act)
            if done and reward == 1:
                win = True
                break
            if done:
                break
        if win:
            solved += 1
    return solved / len(problems) * 100

# ==================== 4. Elo等级分（修正版） ====================
def play_game(ai1, ai2, first_player=1):
    """返回 1: ai1胜, 2: ai2胜, 0:平局"""
    env = GomokuEnv()
    env.reset()
    if first_player == 2:
        env.current_player = 2
    state = (env.board.copy(), env.current_player)
    done = False
    last_mover = None
    while not done:
        current_ai = ai1 if env.current_player == 1 else ai2
        valid = env.get_valid_moves()
        act = current_ai.get_action(state, valid, training=False)
        _, reward, done = env.make_move(act)
        if not done:
            state = (env.board.copy(), env.current_player)
        if done:
            last_mover = current_ai
    if reward == 1:
        # 最后下棋的一方获胜
        return 1 if last_mover == ai1 else 2
    return 0  # 平局

def compute_elo(ai_list, num_games_per_pair=200):
    n = len(ai_list)
    # 记录每对之间的胜率矩阵
    win_rate = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            wins = 0
            total = num_games_per_pair
            for _ in range(total // 2):
                res = play_game(ai_list[i], ai_list[j], first_player=1)
                if res == 1: wins += 1
                elif res == 2: wins -= 1
                res = play_game(ai_list[i], ai_list[j], first_player=2)
                if res == 1: wins += 1
                elif res == 2: wins -= 1
            # 胜率 (wins/total 范围[-1,1]，映射到[0,1])
            rate = (wins + total) / (2 * total)
            win_rate[i, j] = rate
            win_rate[j, i] = 1 - rate
    # 计算每个AI的平均胜率
    avg_rates = win_rate.mean(axis=1)
    # 使用标准Elo公式，限制范围避免无穷
    elos = []
    for rate in avg_rates:
        if rate <= 0.001:
            elo = 800   # 极低
        elif rate >= 0.999:
            elo = 2200  # 极高
        else:
            elo = 1500 + 400 * np.log10(rate / (1 - rate))
        elos.append(elo)
    return {ai_list[i].name: elos[i] for i in range(n)}

# ==================== 归一化（手动） ====================
def min_max_normalize(df, columns):
    df_norm = df.copy()
    for col in columns:
        minv = df_norm[col].min()
        maxv = df_norm[col].max()
        if maxv - minv < 1e-12:
            df_norm[col] = 0.5
        else:
            df_norm[col] = (df_norm[col] - minv) / (maxv - minv)
    return df_norm

# ==================== 主程序 ====================
def main():
    # 加载所有模型
    ais = []
    names = []
    for fname, (name, ai_class) in MODEL_MAP.items():
        path = MODELS_DIR / fname
        if not path.exists():
            print(f"警告: {path} 不存在，跳过")
            continue
        try:
            ai = load_ai(path, ai_class)
            ai.name = name
            ais.append(ai)
            names.append(name)
            print(f"加载成功: {name}")
        except Exception as e:
            print(f"加载失败 {name}: {e}")
    if len(ais) < 2:
        print("至少需要两个模型才能进行完整评估")
        return

    results = {}

    # 1. 决策稳定性
    print("\n评估决策稳定性...")
    for ai in ais:
        cons, ent = evaluate_decision_stability(ai, num_states=20, repeat=200)
        results[ai.name] = {"一致性系数(%)": cons, "决策熵(bit)": ent}

    # 2. 搜索效率
    print("\n评估搜索效率...")
    for ai in ais:
        avg_time, speed = evaluate_search_efficiency(ai, num_decisions=2000)
        results[ai.name].update({"平均决策时间(ms)": avg_time, "决策速度(步/秒)": speed})

    # 3. 战术能力
    print("\n生成战术测试题目...")
    templates = generate_threat_templates()
    problems = expand_templates(templates, num_total=200)
    print(f"共生成了 {len(problems)} 个战术题目")
    for ai in ais:
        success = evaluate_tactical(ai, problems)
        results[ai.name]["战术成功率(%)"] = success

    # 4. Elo等级分
    print("\n计算Elo等级分（循环赛，每对200局）...")
    elo_dict = compute_elo(ais, num_games_per_pair=200)
    for name, elo in elo_dict.items():
        results[name]["Elo等级分"] = elo

    # 输出表格
    df = pd.DataFrame(results).T
    print("\n========== 评估结果 ==========")
    print(df.round(2))
    csv_path = TABLES_DIR / "evaluation_results.csv"
    df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"结果已保存至 {csv_path}")

    # 绘图：柱状图
    metrics = ["一致性系数(%)", "战术成功率(%)", "Elo等级分", "决策速度(步/秒)"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = [results[name][metric] for name in results.keys()]
        names_list = list(results.keys())
        bars = ax.bar(names_list, values, color=['#2c3e50','#7f8c8d','#e74c3c','#27ae60'])
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}', xy=(bar.get_x()+bar.get_width()/2, height),
                        xytext=(0,3), textcoords="offset points", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "evaluation_bar.png", dpi=150)
    plt.close()
    print(f"柱状图已保存至 {FIGURES_DIR / 'evaluation_bar.png'}")

    # 雷达图
    df_radar = df[["一致性系数(%)", "战术成功率(%)", "Elo等级分", "决策速度(步/秒)"]].copy()
    df_scaled = min_max_normalize(df_radar, df_radar.columns)
    angles = np.linspace(0, 2*np.pi, len(df_scaled.columns), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    for name in df_scaled.index:
        values = df_scaled.loc[name].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=name)
        ax.fill(angles, values, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(df_scaled.columns)
    ax.set_title("综合能力雷达图（归一化后）")
    ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.0))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "radar_chart.png", dpi=150)
    plt.close()
    print(f"雷达图已保存至 {FIGURES_DIR / 'radar_chart.png'}")

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    main()