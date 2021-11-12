# PPO on minichess

This is a class project for AA228 at Stanford. We aim to train an agent using Proximal Policy Optimization on ray.rllib's distributed computing infrastructure to play Gardner minichess.

## Contents

- [Contents](#contents)
- [Navigation](#navigation)
- [Rules](#rules)
- [References](#references)

## Navigation

    .
    ├── games                   # Implementation of minichess variations' board and game logic
    ├── mcts_old                # Initial experiments with mcts (hence the repo name)
    ├── mcts_for_train.yml      # Up-to-date conda requirements for macos
    ├── sample_games.txt        # Sample games from log output
    ├── train.py                # Main training script

## Rules

Many variations of Gardner minichess exist. We preserve most well-known chess concepts (e.g. castling, en-passant, fifty-move), but for purposes of incremental development of our agent, we may add in these rules one by one. Following the implementation of [meta-minichess](https://github.com/mdhiebert/meta-minichess), we define game termination as when either side's king is captured (including by the opposing king too). 

## References

Meta-learning experiments for the game of minichess and related rule variants. - Michael Hiebert ([code](https://github.com/mdhiebert/meta-minichess))
Proximal Policy Optimization Algorithms - John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov ([paper](https://arxiv.org/pdf/1707.06347.pdf))