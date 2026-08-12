import sys
import time
import os
from pathlib import Path
from training import Gomoku, ZobristQLearningAI, BOARD_SIZE

# ========== 棋盘显示 ==========
def print_board(env, move_history=None):
    """打印当前棋盘，紧凑对齐格式，并清屏"""
    # 清屏（跨平台）
    os.system('cls' if os.name == 'nt' else 'clear')

    board = env.board
    size = env.board_size
    header = "   " + " ".join(f"{i:1}" for i in range(size))
    print("\n" + header)
    for r in range(size):
        line = f"{r:2}"
        for c in range(size):
            val = board[r, c]
            if val == 1:
                line += " ●"
            elif val == 2:
                line += " ○"
            else:
                line += " ·"
        print(line)
    if move_history and env.last_move:
        last = env.last_move
        color = '黑' if env.board[last[0], last[1]] == 1 else '白'
        print(f"上一步: {last} 由 {color} 下")
    print()

def get_human_move(env, player_name):
    """获取人类玩家输入坐标"""
    while True:
        try:
            inp = input(f"{player_name} 请输入坐标 (行,列) 如 '3,4' 或 '3 4' (q退出): ")
            if inp.lower() == 'q':
                return None
            parts = inp.replace(',', ' ').split()
            if len(parts) != 2:
                print("请输入两个数字，用空格或逗号分隔")
                continue
            r, c = int(parts[0]), int(parts[1])
            if r < 0 or r >= BOARD_SIZE or c < 0 or c >= BOARD_SIZE:
                print(f"坐标应在 0~{BOARD_SIZE-1} 之间")
                continue
            if env.board[r, c] != 0:
                print("该位置已有棋子，请重新输入")
                continue
            return (r, c)
        except ValueError:
            print("输入无效，请输入整数")

def list_models():
    """列出 models 和 points 目录下的 .pkl 文件"""
    dirs = [Path("models"), Path("points")]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.glob("*.pkl"))
    seen = set()
    unique = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return unique

def choose_ai(player_label, cnt):
    """交互式选择AI模型，返回 AI 实例或 None（人工）"""
    models = list_models()
    if not models:
        print(f"未找到任何模型文件，{player_label} 只能使用人工")
        return None
    if cnt == 0:
        print(f"\n可用的模型文件（{player_label}）：")
        for i, m in enumerate(models):
            print(f"  {i+1}. {m.name}")
        print("  h. 人类玩家")
    while True:
        choice = input(f"请选择 {player_label} (输入编号或 'h'): ").strip()
        if choice.lower() == 'h':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                ai = ZobristQLearningAI(player=1)
                try:
                    ai.load(str(models[idx]))
                    return ai
                except Exception as e:
                    print(f"加载模型失败: {e}")
                    continue
            else:
                print(f"请输入 1~{len(models)} 之间的数字")
        except ValueError:
            print("输入无效")

def main():
    print("\n" + "="*50)
    print("五子棋 命令行对战平台")
    print("="*50)
    print("棋盘：● 黑棋，○ 白棋，· 空位")

    black_ai = choose_ai("黑棋 (先手)", 0)
    white_ai = choose_ai("白棋 (后手)", 1)

    if black_ai is None and white_ai is None:
        print("至少一方应为AI，否则无法自动落子")
        return

    env = Gomoku(BOARD_SIZE)
    env.current_player = 1

    mode = "human_vs_ai" if (black_ai is None or white_ai is None) else "ai_vs_ai"
    print(f"\n对战模式: {'人机对战' if mode == 'human_vs_ai' else 'AI自对弈'}")
    print("输入 'q' 可随时退出")
    print("按 Enter 继续...")
    input()

    # 清屏一次，准备开始对局
    os.system('cls' if os.name == 'nt' else 'clear')

    move_history = []
    done = False
    while not done:
        print_board(env, move_history)
        if env.current_player == 1:
            player_name = "黑棋"
            current_ai = black_ai
        else:
            player_name = "白棋"
            current_ai = white_ai

        if current_ai is None:
            action = get_human_move(env, player_name)
            if action is None:
                print("游戏退出")
                return
        else:
            if mode == "ai_vs_ai":
                print(f"{player_name} AI 思考中...")
                time.sleep(0.5)
            valid_moves = env.get_valid_moves()
            if not valid_moves:
                print("棋盘已满，平局")
                break
            action = current_ai.get_action((env.board.copy(), env.current_player), valid_moves, training=False)
            print(f"{player_name} AI 落子 {action}")

        success, next_state, reward, done = env.make_move(action)
        if not success:
            print("非法落子，请重试")
            continue

        move_history.append(action)

        if done:
            print_board(env, move_history)
            if reward == 1:
                last_pos = env.last_move
                last_player = env.board[last_pos[0], last_pos[1]]
                winner = "黑棋" if last_player == 1 else "白棋"
                print(f"🏆 {winner} 获胜！")
            elif reward == 0:
                print("🤝 平局！")
            else:
                print("游戏结束")
            break

    if input("\n是否再来一局？(y/n): ").lower() == 'y':
        main()
    else:
        print("感谢使用！")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()