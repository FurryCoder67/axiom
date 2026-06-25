# Axiom

A terminal AI agent written in pure Rust with a **from-scratch neural network** — no ML frameworks, no pretrained models. The agent autonomously plans and executes shell commands to accomplish your goals, using the neural net as a **reward model** to rank candidate action sequences before running them. It learns from outcomes and persists that knowledge across sessions.

## How It Works

1. You type a goal (e.g., "list all files in this directory").
2. A rule-based planner generates N candidate action plans.
3. Each plan is encoded into an 8-feature vector (plan length, action type proportions, risk level, goal keyword overlap, historical success rate).
4. The neural net — a fully-connected feedforward network with ReLU hidden layers and a sigmoid output — predicts the success probability for each plan.
5. The highest-scoring plan is displayed with a probability bar, then executed step-by-step via `std::process::Command`.
6. After execution, the net is trained via backpropagation on the actual outcome (success=1.0, failure=0.0).
7. Weights and history are saved to `~/.axiom/` — the net gets smarter across runs.

## Quick Start

```bash
# Requires Rust stable toolchain
cd axiom
cargo run --release
```

## REPL Commands

| Command | Description |
|---|---|
| `<any text>` | Give Axiom a goal to plan and execute |
| `status` | Show current goal, net architecture, and success rate |
| `interrupt` | Set interrupt flag (halts at next step) |
| `redirect <goal>` | Abandon current task and pivot to a new goal |
| `inspect weights` | Show neural net layer shapes and weight statistics |
| `history` | Show past tasks and their outcomes |
| `help` | Show available commands |
| `quit` | Save state and exit |

## Example Session

```
  ╔══════════════════════════════════════╗
  ║              A X I O M                 ║
  ╚══════════════════════════════════════╝

  Terminal AI agent — from-scratch neural net
  → Net: [8x16 → 16x8 → 8x1] (lr=0.1) | Params: 209 | History: 0 tasks | Data: ~/.axiom
  Type a goal, or 'help' for commands.

axiom ▶ list files in this directory

  Goal: list files in this directory

  Candidate Plans:
    ▶ Plan 1: List directory contents
      ██████░░ 0.72
        1. ls -la — List all files in current directory
      Plan 2: List with disk usage summary
      ████░░░░ 0.51
        1. ls -la — List all files
        2. du -sh . — Show directory size
       Plan 3: Thorough exploration before action
      ███░░░░░ 0.38
        1. pwd — Show current directory
        2. ls -la — List all files
        3. file * — Identify file types
       Plan 4: Alternative: use find to list files
      █████░░░ 0.63
        1. find . -maxdepth 1 -print — Find files in current directory

  Executing Plan 1 (score: 0.72)...

  [step 1/1] List all files in current directory
  $ ls -la
    drwxr-xr-x  .  src  Cargo.toml  config.toml  README.md

  ✓ Task completed successfully. (trained on outcome: success)
```

## Architecture

```
axiom/
├── Cargo.toml           # Project manifest (serde, toml, rand — no ML crates)
├── config.toml           # Net architecture, agent config, blocklist
├── README.md
└── src/
    ├── main.rs           # REPL loop, goal handling, plan execution, config
    ├── neural_net.rs     # From-scratch NN: forward pass, backprop, Xavier init
    ├── types.rs           # Action, Plan, TaskRecord, ActionType
    ├── planner.rs         # Rule-based plan generation + feature encoding
    ├── safety.rs          # Blocklist checking
    ├── storage.rs         # Weight & history persistence to ~/.axiom/
    └── terminal.rs        # ANSI color helpers, score bar, prompt
```

## Neural Network

The net is a standard fully-connected feedforward network, implemented entirely in `neural_net.rs`:

- **Layers**: configurable via `config.toml` (default: `8 → 16 → 8 → 1`)
- **Hidden activations**: ReLU
- **Output activation**: sigmoid (produces a probability 0–1)
- **Initialization**: Xavier/Glorot uniform
- **Training**: online stochastic gradient descent — one backprop pass per completed task
- **Loss**: mean squared error against the actual outcome (1.0 success, 0.0 failure)

No external ML crates. Just `rand` for weight initialization and `serde` for serialization.

## Feature Encoding

Each candidate plan is encoded into an 8-dimensional vector:

| Index | Feature | Description |
|---|---|---|
| 0 | Plan length | Normalized by 10 |
| 1 | Read proportion | Fraction of actions that are reads |
| 2 | Write proportion | Fraction that are writes |
| 3 | Execute proportion | Fraction that are shell executes |
| 4 | Network proportion | Fraction that are network ops |
| 5 | Risk level | Weighted: writes×0.5 + executes×0.7 + network×0.3 |
| 6 | Goal overlap | How many goal keywords appear in the plan text |
| 7 | History success | Success rate of similar plans in past tasks |

## Persistent Memory

Axiom stores everything in `~/.axiom/`:

- `weights.json` — serialized neural net (layers, weights, biases, learning rate)
- `history.jsonl` — append-only log of every task (goal, plan, features, predicted vs actual outcome, steps completed)

On startup, the net is deserialized from `weights.json`. If absent, a fresh net is initialized with random Xavier weights. The history feeds feature encoding (historical success rate) and grows the net's training data across sessions.

## Safety

Before executing any plan, each command is checked against a blocklist (defined in `config.toml`). Commands matching blocked patterns (e.g., `rm -rf /`, `sudo`, `mkfs`) are flagged and the plan is aborted. Edit `~/.axiom/config.toml` or the local `config.toml` to customize the blocklist.

## License

MIT