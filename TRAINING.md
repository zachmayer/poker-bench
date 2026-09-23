# Simplified Poker Training Guide

## Quick Start

Train a poker AI with the simplified script:

```bash
python train.py --episodes 1000
```

## Key Improvements

### 1. Enhanced Features (18 dimensions)
- **Stack & Position**: Normalized stack sizes, position ratio
- **Pot Odds**: Real-time pot odds calculation
- **Street Indicators**: One-hot encoding for game phase
- **Aggression Metrics**: Betting patterns and pot-to-stack ratios
- **Stack Depth Categories**: Short/medium/deep stack indicators

### 2. Improved Action Space (5 actions)
- Fold
- Check/Call
- Raise Small (33% pot)
- Raise Medium (75% pot)
- Raise Large (150% pot)

### 3. Better Reward Shaping
- Rewards normalized by big blinds
- Discounted future rewards with GAE
- Position-aware reward calculation

### 4. Training Optimizations
- PPO with clipped objectives
- Mixed opponent pool (Random, Calling, Gambling)
- Experience buffer with importance sampling
- Adaptive exploration temperature

## Architecture

```
Input (18 features) →
  Hidden Layers (3x256 with dropout) →
    Actor Head (5 actions) + Critic Head (value)
```

## Training Parameters

- **Learning Rate**: 3e-4 with Adam optimizer
- **Batch Size**: 32 episodes
- **Clip Ratio**: 0.2 for PPO
- **Entropy Coefficient**: 0.01 for exploration
- **GAE Lambda**: 0.95 for advantage estimation

## Expected Performance

- **Initial**: -2 to -5 BB per hand (random play)
- **After 500 episodes**: -0.5 to +0.5 BB per hand
- **After 1000 episodes**: +0.5 to +1.5 BB per hand
- **After 2000 episodes**: +1.0 to +2.5 BB per hand

## Advanced Options

```bash
# Full training with custom parameters
python train.py \
  --episodes 2000 \
  --batch-size 64 \
  --lr 0.001 \
  --hidden-dim 512 \
  --seed 123
```

## Evaluation

The model automatically evaluates every 50 episodes against mixed opponents and reports:
- Average BB won per hand
- Average hands played per game
- Training loss and entropy

## Model Output

Trained model is saved to `poker_model.pt` after training.
