# Poker Training Script Improvements

## Summary
Created a simplified, more effective poker training script (`train_poker.py`) that maximizes learning potential while reducing complexity.

## Key Simplifications & Improvements

### 1. **Unified Training Pipeline**
- Single `train_poker.py` file instead of multiple scattered modules
- Clear configuration with dataclass instead of complex configs
- Direct integration with PokerBots engine

### 2. **Enhanced Feature Engineering (18 dimensions)**
- **Position-aware features**: Position ratio for strategic play
- **Stack dynamics**: Normalized stacks, pot-to-stack ratios
- **Pot odds calculation**: Real-time pot odds for mathematical decisions
- **Street indicators**: One-hot encoding for game phases
- **Action context**: Check/raise availability
- **Stack depth categories**: Short/medium/deep stack strategies

### 3. **Simplified Action Space (5 actions)**
- Fold
- Check/Call
- Raise Small (33% pot)
- Raise Medium (75% pot)
- Raise Large (150% pot)

### 4. **Better Learning Algorithm**
- **PPO with GAE**: More stable learning than basic GRPO
- **Actor-Critic architecture**: Shared representation learning
- **Experience buffer**: Efficient sample reuse
- **Mixed opponents**: Training against Random, Calling, and Gambling players

### 5. **Improved Reward Shaping**
- Normalized by big blinds for consistent scale
- Discounted future rewards with proper credit assignment
- Position and stack-aware reward calculation

## Performance Expectations

| Episodes | Expected BB/Hand | Notes |
|----------|-----------------|-------|
| 0-100    | -2.0 to -0.5   | Learning basics |
| 100-500  | -0.5 to +0.5   | Developing strategy |
| 500-1000 | +0.5 to +1.5   | Refining play |
| 1000-2000| +1.0 to +2.5   | Advanced tactics |

## Usage

```bash
# Quick training
uv run python train.py --episodes 1000

# Advanced training
uv run python train.py \
  --episodes 2000 \
  --batch-size 64 \
  --lr 0.0005 \
  --hidden-dim 512
```

## Architecture Benefits

1. **Dropout regularization**: Prevents overfitting to specific patterns
2. **Multi-layer design**: Captures complex poker dynamics
3. **Separate actor/critic heads**: Specialized value and policy learning
4. **Temperature-based exploration**: Balanced exploration/exploitation

## Training Efficiency

- **Batch processing**: Learns from multiple trajectories simultaneously
- **Experience replay**: Reuses valuable experiences
- **Gradient clipping**: Stable updates
- **Adaptive exploration**: Temperature-scaled action selection

The simplified script is more maintainable, easier to understand, and more effective at training a competitive poker AI.
