import numpy as np
import random
import torch
from pathlib import Path
from datetime import datetime
import sys
import gc
import logging

# 从 DQNAI 导入训练所需的组件（复用训练器、环境等）
from DQNAI import AITrainer, ProgressBar, Gomoku, DEVICE, BOARD_SIZE
from DQNAI import MODELS_DIR, LOGS_DIR, FIGURES_DIR, POINTS_DIR, setup_logging, list_all_models
from DDQN import DDQNAI  # 导入 Double DQN AI 类


# 设置日志（复用 DQNAI 中的日志函数）
logger = setup_logging()

def main():
    print("\n" + "=" * 60)
    print("五子棋AI训练系统 (Double DQN 版)")
    print("=" * 60)

    # 列出已有模型（用于继续训练）
    models = list_all_models()

    # 选择黑棋 AI
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

    # 选择白棋 AI
    print("\n请选择白棋 AI (player 2): (输入0表示创建新 DDQN AI)")
    choice2 = int(input("输入序号: ")) - 1
    if choice2 == -1:
        model2 = None
        print("选择白棋: 新 DDQN AI")
    else:
        model2 = models[choice2]
        print(f"选择白棋: {Path(model2).name}")

    # 创建两个 DDQN AI 实例
    ai_black = DDQNAI(player=1, learning_rate=0.001, exploration_rate=0.4)
    ai_white = DDQNAI(player=2, learning_rate=0.001, exploration_rate=0.4)

    # 如果选择了已有模型，则加载
    if model1 is not None:
        ai_black.load(model1)
    if model2 is not None:
        ai_white.load(model2)

    # 计算起始训练局数（取两个 AI 中最大的总训练局数）
    start_episode = max(
        getattr(ai_black, 'total_training_games', ai_black.total_games) if model1 is not None else 0,
        getattr(ai_white, 'total_training_games', ai_white.total_games) if model2 is not None else 0
    )

    # 创建训练器（复用 DQNAI 中的 AITrainer，它与 DDQNAI 兼容）
    trainer = AITrainer(ai_black, ai_white)
    trainer.start_episode = start_episode

    # 输入训练参数
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

    # 开始训练
    trainer.train(
        episodes=episodes,
        save_every=save_every,
        log_every=log_every
    )

    # 输出最终统计
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