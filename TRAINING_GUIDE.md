# 🤺 The RoboDojo Training Journey

This guide explains the evolution of your RL agent from its first "random" steps to becoming a competitive Robocode champion.

---

## 🗺️ Roadmap Overview

Training in RoboDojo is not a single linear process; it is a **Curriculum** that evolves as your bot gets smarter.

| Phase | Milestone                | Typical Iterations | Opponent Style             | Focus                                   |
| :---- | :----------------------- | :----------------- | :------------------------- | :-------------------------------------- |
| **0** | **The Clumsy Discovery** | 1 - 500            | None (`noop_bot`)          | Learning to move and find targets       |
| **1** | **Foundation Skills**    | 500 - 5,000        | `simple_target`, `Crazy`   | Aiming, Shooting, and Basic Dodging     |
| **2** | **Mastering Melee**      | 5,000 - 20,000     | `Walls`, `SpinBot`, `Fire` | Advanced spatial awareness & strategies |
| **3** | **Emergent Self-Play**   | 20,000+            | **Self-Play League**       | Beating its own best past versions      |

---

## 🥚 Phase 0: The Clumsy Discovery (Iter 0-500)

**What's happening:** Your bot starts with a randomly initialized brain. It knows nothing about physics, guns, or coordinates.

*   **Visual Check:** The bot will spin aimlessly, crash into walls, and fire shells at the empty arena.
*   **The Strategy:** We use `noop_bot` (stationary) as an opponent. Eventually, your bot will accidentally hit the target. The PPO algorithm "tags" that action as **GOOD** (via reward) and starts reinforcing it.
*   **Success Metric:** Reward moves from heavily negative (penalties for wall-hits/skipped turns) to near-zero.

---

## 🏹 Phase 1: Foundation Skills (Iter 500 - 5,000)

**What's happening:** The bot now understands how its controls work. It begins to track enemies.

*   **Visual Check:** You'll see the radar "locking on" and the gun tracking the `simple_target`. The bot starts moving in loops to avoid being a sitting duck.
*   **The Transition:** The Curriculum automatically swaps the stationary target for `simple_target` or `Crazy`.
*   **Success Metric:** A steady climb in `episode_reward_mean`. You should see `Win Rate` appearing in your terminal logs.

---

## ⚔️ Phase 2: Mastering Combat (Iter 5,000 - 20,000)

**What's happening:** The bot enters "High School." It is now fighting bots that move fast and shoot back accurately.

*   **Visual Check:** You'll see **Circular Targeting** (predicting where the enemy will be) and **Evasive Maneuvers**. It will stop crashing into walls because the penalty is too high.
*   **The Challenge:** Opponents like `Walls` or `SpinBot` force the agent to learn complex movement patterns.
*   **Success Metric:** Win Rate against the `opponent_pool` reaches > 90%.

---

## 🌌 Phase 3: The Infinite Loop (Self-Play)

**What's happening:** Once your bot can beat all sample bots, it has nothing left to learn from them. Now it must fight **itself**.

*   **Why?** Sample bots have fixed patterns. If your bot only fights `Crazy`, it might become an expert at beating `Crazy` but fail against anything else.
*   **How it works:** Every 500 iterations, the system saves a "Snapshot" of your current bot to the **Policy League**. New matches alternate between sample bots and these elite past versions.
*   **Success Metric:** The **ELO Rating** (visible in `league/elo_mean`). If ELO is rising, your bot is discovering new strategies that its "older self" couldn't handle.

---

## 🛠️ How to Manage the Journey

### 1. Stopping and Resuming
If you need to reboot your machine or change settings, don't worry. Your bot's brain is saved frequently.
```powershell
# Resume where you left off
make train-resume
```

### 2. Monitoring the "Soul" of the Bot
Don't just look at numbers. Occasionally watch a battle:
- Check `artifacts/recordings/` for `.mp4` files.
- **Note**: Recordings will be **black** if `env.use_gui` is set to `false`. To see the game in recordings (and for the bot to see the arena), set `use_gui: true` in `config/env/robocode.yaml`.
- Use `make serve` to run the bot "Live" against local sample bots in the Robocode GUI.

### 3. Scaling the Brain
If you notice the bot is learning too slowly but your GPU is empty:
- Use the **Aggressive Scaling** we built (40+ workers).
- This collects more "experience" per hour, compressing 1,000 iterations of learning into much less time.

---

## 📊 Knowing When You've Peaked (The Plateau)

Self-play can be tricky because the bot is chasing its own tail. Use these metrics to know when it has truly "mastered" its environment:

1.  **`league/elo_max` Stagnation**: This is your primary indicator. If the "all-time best" ELO hasn't increased in 1,000+ iterations, the bot has likely hit its strategic peak with the current hyperparameters.
2.  **`league/elo_mean` Convergence**: When the average ELO of the whole league catches up to the max ELO, the matches are becoming highly balanced.
3.  **`curriculum/win_rate` Anchor**: Even during self-play, your bot must maintain a near 100% win rate against sample bots (like `Crazy` or `Walls`). If this drops, the bot is "overfitting" to its own style and forgetting how to fight others.
4.  **`episode_reward_mean` Convergence to 0.0**: In perfectly matched self-play, matches should be 50/50, leading to a zero-sum reward. If the reward settles at 0.0 with very low variance, the bots have effectively "solved" each other.

---

## 🏆 Graduation: Using the Model

You are ready to graduate when you hit these milestones:

*   ✅ `league/elo_max` has been flat for several hours of training.
*   ✅ `curriculum/win_rate` against the full sample bot pool is > 95%.
*   ✅ Visual confirmation: In `artifacts/recordings/`, your bot shows tactical movement (dodging shells) rather than just lucky collisions.

**Steps to deploy:**
1.  Run `make export`.
2.  Your standalone weights will be at `artifacts/serving/bot_weights.pt`.
3.  Use `make serve` to run a final "Victory Lap" in the Robocode GUI!

