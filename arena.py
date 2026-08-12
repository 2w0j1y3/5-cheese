import numpy as np
import random
import time
import pickle
import lz4.frame
from pathlib import Path
import logging
import sys
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导入训练程序中的环境与AI类
try:
    from training import Gomoku, ZobristQLearningAI, BOARD_SIZE, MAX_MOVES
except ImportError:
    print("错误：无法导入 training.py，请确保该文件在同一目录下。")
    sys.exit(1)

# ==================== 配置 ====================
MODELS_DIR = Path("models")
POINTS_DIR = Path("points")
SEARCH_DIRS = [MODELS_DIR, POINTS_DIR]

# ==================== ELO 评分系统（增强版） ====================
class EloRating:
    """
    增强版 ELO 等级分，支持动态 K 因子、平局处理、先手校正（可选）
    """
    def __init__(self, initial_rating=1500, k_factor=32, dynamic_k=False, games_per_player=0):
        """
        :param initial_rating: 初始分数
        :param k_factor: 基础 K 因子
        :param dynamic_k: 是否根据对局数动态调整 K（K = k_factor / sqrt(1 + games/100)）
        :param games_per_player: 每个玩家已赛场次（用于动态K，若不启用则忽略）
        """
        self.ratings = {}          # player_id -> rating
        self.initial = initial_rating
        self.k_base = k_factor
        self.dynamic_k = dynamic_k
        self.games_count = defaultdict(int)  # player_id -> 总对局数
        self.history = []          # 记录每场更新 (player_a, player_b, rating_a, rating_b, score_a)

    def add_player(self, player_id):
        if player_id not in self.ratings:
            self.ratings[player_id] = self.initial
            self.games_count[player_id] = 0

    def get_rating(self, player_id):
        return self.ratings.get(player_id, self.initial)

    def expected_score(self, rating_a, rating_b):
        """A 对 B 的期望得分 (0~1)"""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def update(self, player_a, player_b, score_a):
        """
        player_a 对 player_b 的比赛结果，score_a 为 player_a 的得分（胜1，平0.5，负0）
        """
        self.add_player(player_a)
        self.add_player(player_b)
        ra = self.ratings[player_a]
        rb = self.ratings[player_b]
        ea = self.expected_score(ra, rb)
        eb = 1 - ea

        # 计算动态 K 因子
        if self.dynamic_k:
            k_a = self.k_base / (1 + self.games_count[player_a] / 100) ** 0.5
            k_b = self.k_base / (1 + self.games_count[player_b] / 100) ** 0.5
            # 保证最小 K 值
            k_a = max(8, k_a)
            k_b = max(8, k_b)
        else:
            k_a = self.k_base
            k_b = self.k_base

        new_ra = ra + k_a * (score_a - ea)
        new_rb = rb + k_b * ((1 - score_a) - eb)

        self.ratings[player_a] = new_ra
        self.ratings[player_b] = new_rb
        self.games_count[player_a] += 1
        self.games_count[player_b] += 1
        self.history.append((player_a, player_b, ra, rb, score_a, new_ra, new_rb))

    def get_sorted_rankings(self):
        """返回按评级降序排列的 (player_id, rating) 列表"""
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)


# ==================== 竞技场（增强版） ====================
class Arena:
    def __init__(self, model_paths, elo_initial=1500, elo_k=32, dynamic_k=False):
        """
        model_paths: 模型文件路径列表
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.ais = {}              # player_id -> AI实例
        self.elo = EloRating(initial_rating=elo_initial, k_factor=elo_k, dynamic_k=dynamic_k)
        self.player_names = {}     # player_id -> 显示名称
        self.match_results = []    # (p1, p2, score1, score2) 每局结果
        self.game_log = []         # 详细记录 (p1, p2, s1, s2, winner)

    def load_ais(self):
        """加载所有AI模型"""
        for idx, path in enumerate(self.model_paths):
            player_id = f"AI_{idx+1}"
            ai = ZobristQLearningAI(player=1)
            try:
                ai.load(str(path))
                ai.player = 1   # 统一，不影响
                self.ais[player_id] = ai
                self.player_names[player_id] = path.stem
                print(f"已加载: {path.name} -> {player_id}")
            except Exception as e:
                print(f"加载 {path.name} 失败: {e}")

    def play_match(self, p1, p2, num_games=10, swap_first=True):
        """
        对战指定局数，每局独立更新ELO
        """
        ai1 = self.ais[p1]
        ai2 = self.ais[p2]
        for game_num in range(num_games):
            # 决定先手
            first = 1 if (game_num % 2 == 0 or not swap_first) else 2
            env = Gomoku(BOARD_SIZE)
            if first == 2:
                env.current_player = 2
            state = (env.board.copy(), env.current_player)
            done = False
            move_count = 0
            last_ai = None
            while not done and move_count < MAX_MOVES:
                move_count += 1
                if env.current_player == 1:
                    current_ai = ai1
                else:
                    current_ai = ai2
                valid_moves = env.get_valid_moves()
                action = current_ai.get_action(state, valid_moves, training=False)
                success, next_state, reward, done = env.make_move(action)
                if not success:
                    break
                state = (env.board.copy(), env.current_player)
                last_ai = current_ai

            # 结果判定
            if done and reward == 1:
                winner = last_ai
                if winner == ai1:
                    score1, score2 = 1.0, 0.0
                else:
                    score1, score2 = 0.0, 1.0
            elif done and reward == 0:
                score1, score2 = 0.5, 0.5
            else:
                # 超时或异常，平局
                score1, score2 = 0.5, 0.5

            # 更新ELO
            self.elo.add_player(p1)
            self.elo.add_player(p2)
            self.elo.update(p1, p2, score1)

            # 记录结果
            self.match_results.append((p1, p2, score1, score2))
            if score1 > score2:
                winner_id = p1
            elif score1 < score2:
                winner_id = p2
            else:
                winner_id = None
            self.game_log.append((p1, p2, score1, score2, winner_id))

    def run_round_robin(self, games_per_pair=10, swap_first=True):
        """循环赛：所有AI两两对战"""
        if len(self.ais) < 2:
            print("至少需要2个AI进行对战")
            return
        player_ids = list(self.ais.keys())
        n = len(player_ids)
        total_pairs = n * (n-1) // 2
        print(f"开始循环赛，共 {total_pairs} 对组合，每对 {games_per_pair} 局")
        pair_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                p1 = player_ids[i]
                p2 = player_ids[j]
                pair_idx += 1
                print(f"第 {pair_idx}/{total_pairs} 组: {self.player_names[p1]} vs {self.player_names[p2]} ...")
                self.play_match(p1, p2, num_games=games_per_pair, swap_first=swap_first)
                print(f"  完成 {games_per_pair} 局")

    def print_rankings(self):
        """打印当前ELO排名（含统计）"""
        rankings = self.elo.get_sorted_rankings()
        # 计算胜平负
        stats = defaultdict(lambda: [0, 0, 0])  # 胜,平,负
        for p1, p2, s1, s2 in self.match_results:
            if s1 > s2:
                stats[p1][0] += 1
                stats[p2][2] += 1
            elif s1 < s2:
                stats[p1][2] += 1
                stats[p2][0] += 1
            else:
                stats[p1][1] += 1
                stats[p2][1] += 1
        print("\n" + "="*60)
        print("ELO 等级分排名")
        print("="*60)
        print(f"{'排名':<4} {'AI名称':<25} {'ELO分数':<10} {'胜-平-负'}")
        for rank, (pid, rating) in enumerate(rankings, 1):
            name = self.player_names.get(pid, pid)
            w, d, l = stats.get(pid, (0,0,0))
            print(f"{rank:<4} {name:<25} {rating:<10.1f} {w}-{d}-{l}")

    def plot_elo(self, save_path=None, show=True):
        """
        绘制ELO等级分柱状图，按分数降序排列，并标注数值
        """
        if not self.elo.ratings:
            print("无ELO数据，无法绘图")
            return
        rankings = self.elo.get_sorted_rankings()  # 已排序
        names = [self.player_names.get(pid, pid) for pid, _ in rankings]
        scores = [rating for _, rating in rankings]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, scores, color='skyblue', edgecolor='black')
        # 在柱顶显示分数
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{score:.1f}', ha='center', va='bottom', fontsize=10)

        plt.ylabel('ELO 等级分')
        plt.title('AI 模型 ELO 等级分排名')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()

    def save_results(self, filename=None):
        """保存ELO结果到文件"""
        if filename is None:
            filename = f"elo_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("ELO 等级分排名\n")
            f.write("="*60 + "\n")
            rankings = self.elo.get_sorted_rankings()
            stats = defaultdict(lambda: [0,0,0])
            for p1, p2, s1, s2 in self.match_results:
                if s1 > s2:
                    stats[p1][0] += 1
                    stats[p2][2] += 1
                elif s1 < s2:
                    stats[p1][2] += 1
                    stats[p2][0] += 1
                else:
                    stats[p1][1] += 1
                    stats[p2][1] += 1
            f.write(f"{'排名':<4} {'AI名称':<25} {'ELO分数':<10} {'胜-平-负'}\n")
            for rank, (pid, rating) in enumerate(rankings, 1):
                name = self.player_names.get(pid, pid)
                w,d,l = stats.get(pid, (0,0,0))
                f.write(f"{rank:<4} {name:<25} {rating:<10.1f} {w}-{d}-{l}\n")
        print(f"结果已保存到 {filename}")

# ==================== 工具函数 ====================
def find_model_files(dirs=None, pattern="*.pkl"):
    if dirs is None:
        dirs = SEARCH_DIRS
    files = []
    for d in dirs:
        d = Path(d)
        if d.exists():
            files.extend(d.glob(pattern))
    unique = {}
    for f in files:
        if f.name not in unique:
            unique[f.name] = f
    return list(unique.values())

def list_models_interactive():
    files = find_model_files()
    if not files:
        print("未找到任何模型文件（.pkl）")
        return []
    print("\n找到以下模型文件：")
    for i, f in enumerate(files):
        print(f"{i+1}. {f.name} ({f.parent.name})")
    selected = []
    while True:
        choice = input("请输入要选择的模型编号（多个用逗号分隔，如 1,3,5），或输入 'all' 选择全部: ")
        if choice.lower() == 'all':
            selected = files
            break
        try:
            indices = [int(x.strip()) for x in choice.split(',') if x.strip()]
            for idx in indices:
                if 1 <= idx <= len(files):
                    selected.append(files[idx-1])
                else:
                    print(f"编号 {idx} 超出范围，请重新输入")
                    selected = []
                    break
            if selected:
                break
        except ValueError:
            print("输入格式错误，请重新输入")
    return selected

def main():
    print("\n" + "="*60)
    print("AI 对战与 ELO 等级分计算系统 (增强版)")
    print("="*60)
    
    model_files = list_models_interactive()
    if not model_files:
        print("没有选择任何模型，退出")
        return

    print(f"\n选择了 {len(model_files)} 个模型:")
    for f in model_files:
        print(f"  {f.name}")

    # 设置ELO参数
    try:
        games_per_pair = int(input("请输入每对AI对战局数（默认10）: ") or "10")
    except ValueError:
        games_per_pair = 10
    try:
        elo_k = int(input("请输入ELO K因子（默认32）: ") or "32")
    except ValueError:
        elo_k = 32
    dyn_k = input("是否启用动态K因子（根据对局数调整）？(y/n, 默认 n): ").lower() == 'y'
    swap = input("是否交换先手？(y/n, 默认 y): ").lower()
    swap_first = swap != 'n'

    # 创建竞技场
    arena = Arena(model_files, elo_initial=1500, elo_k=elo_k, dynamic_k=dyn_k)
    arena.load_ais()
    arena.run_round_robin(games_per_pair=games_per_pair, swap_first=swap_first)
    arena.print_rankings()

    # 保存结果
    save = input("是否保存ELO结果到文本文件？(y/n, 默认 y): ").lower()
    if save != 'n':
        arena.save_results()

    # 绘制柱状图
    plot = input("是否绘制ELO等级分柱状图？(y/n, 默认 y): ").lower()
    if plot != 'n':
        fig_path = f"elo_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        arena.plot_elo(save_path=fig_path, show=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()