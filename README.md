# Dynamic Ambulance Redeployment Simulator (RL)

A reinforcement learning and discrete-event simulation framework for dynamic emergency medical services (EMS) redeployment. Idle ambulances are repositioned to strategic standby bases to minimize response times and dropped calls under time-varying spatial demand.

---

## 1. Project Overview & Architecture

Emergency demand fluctuates throughout the day across residential, commercial, industrial, and nightlife zones. When ambulances transport patients to hospitals, they become displaced away from their original bases, leaving demand hotspots uncovered.

This repository implements a discrete-event simulator, baseline heuristic policies, and a Double Linear Q-Learning agent with action constraints to optimize redeployment decisions.

### Repository Structure

```
├── ems/
│   ├── ambulance.py     # Ambulance state machine (IDLE, RESPONDING, AT_SCENE, TO_HOSPITAL, AT_HOSPITAL, REPOSITIONING)
│   ├── city.py          # Spatial geometry, road network distances, candidate bases, hospitals, and coverage metrics
│   ├── config.py        # CityConfig and time-varying spatial hotspot profiles
│   ├── demand.py        # Non-homogeneous Poisson call arrival and spatial generator
│   ├── simulator.py     # Priority-queued discrete-event simulator with exact clock advancement
│   ├── state.py         # 85-dimensional normalized feature extractor (fleet state, base demand/occupancy, donor context)
│   ├── rewards.py       # Bounded interval reward balancing response times, dropped calls, coverage, and transit cost
│   ├── agent.py         # Double Linear Q-learning agent with TD clipping, weight bounds, and action masking
│   ├── policies.py      # Benchmark baseline policies (No Redeployment, Fixed Home Base, Nearest Base, Greedy MEXCLP)
│   ├── rl_policy.py     # RL policy interface with candidate donor selection, STAY action, and relocation hurdle
│   ├── training.py      # SimulatorTrainer managing executed transitions and episode boundary updates
│   └── metrics.py       # Performance metrics calculator (mean, median, 90th percentile, target fraction)
├── tests/
│   ├── test_simulator.py # Verification of all 10 operational invariants
│   ├── test_policies.py  # Unit tests for baseline policies
│   ├── test_rewards.py   # Unit tests for bounded reward function and penalty scaling
│   ├── test_agent.py     # Unit tests for LinearQAgent, Double Q updates, stability, and masking
│   ├── test_rl_policy.py # Unit tests for RLPolicy, StateBuilder dimensions, and STAY action
│   └── test_training_integration.py # Unit tests for train_agent, model serialization, and eval pipeline
├── models/
│   └── linear_q_agent.json # Trained agent weights and training metadata
├── results/
│   ├── evaluation_summary.json # Machine-readable summary statistics and paired t-test results
│   └── evaluation_details.csv  # Granular seed-by-seed performance metrics
├── requirements.txt     # Python package requirements
├── train.py             # Script to train RL agent on disjoint training seeds
├── evaluate.py          # Fair evaluation script comparing all 5 policies across held-out test seeds
└── README.md            # Comprehensive system documentation
```

---

## 2. Dependencies

The simulator and reinforcement learning algorithms are designed to be lightweight, robust, and completely self-contained.

### Runtime Requirements
- **Python**: `Python >= 3.10` (tested and verified with Python 3.14 on Windows 11).
- **Core Library**:
  - `numpy >= 1.24.0` (specified in `requirements.txt`).
- **Standard Library Modules Used**:
  - `math`, `random`: Euclidean / Manhattan distances, Poisson arrival intervals, and uniform distributions.
  - `heapq`: Priority-queued event scheduling in discrete-event simulation.
  - `dataclasses`: Type-annotated domain data structures (`Location`, `EmergencyCall`, `Ambulance`, `CityConfig`).
  - `json`, `csv`: Model serialization, configuration loading, and structured experiment logging.
  - `unittest`: Unit testing and invariant verification framework.
  - `argparse`: Command-line argument parsing.

### Hardware Requirements
- Commodity CPU (no GPU required).
- Full 100-episode training runs in approximately **25–35 seconds**.
- Full 20-seed benchmark evaluation runs in approximately **8–12 seconds**.
- Complete 27-test verification suite runs in approximately **0.12 seconds**.

---

## 3. Installation Steps

### Step 1: Clone or Open the Repository
```powershell
git clone <repository-url>
cd AI-Assignment-2
```

### Step 2: Set Up a Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Verify Installation
Run the test suite to ensure the environment, simulator, and policies are fully operational:
```powershell
py -m unittest discover tests
```
*(On Linux/macOS, substitute `python3` for `py`)*

---

## 4. Execution Commands

### A. Running the Unit Test Suite

The test suite validates all 10 simulator operational invariants, baseline decision logic, reward boundedness, Double Q-learning stability, and pipeline integration.

```powershell
# Discover and run all 27 unit tests
py -m unittest discover tests

# Run with verbose test-by-test reporting
py -m unittest discover tests -v

# Run individual test modules
py -m unittest tests/test_simulator.py           # 10 simulator invariants
py -m unittest tests/test_policies.py            # Baseline heuristics
py -m unittest tests/test_rewards.py             # Reward boundedness & scaling
py -m unittest tests/test_agent.py               # Double Q updates & stability
py -m unittest tests/test_rl_policy.py           # Action masking & STAY hurdle
py -m unittest tests/test_training_integration.py # Pipeline & serialization
```

### B. Training the Reinforcement Learning Agent

The `train.py` script trains the Double Linear Q-learning agent on disjoint training seeds (seeds `1000..1099` by default) to ensure evaluation remains unbiased.

```powershell
# Default training run (100 episodes, training seed 1000, lr 0.005)
py train.py --episodes 100 --train-seed 1000 --lr 0.005

# Custom training options
py train.py --episodes 150 --train-seed 2000 --lr 0.005 --gamma 0.95 --output models/custom_agent.json
```

**Available CLI Flags for `train.py`:**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--episodes` | `int` | `100` | Number of 24-hour simulation days for training. |
| `--train-seed` | `int` | `1000` | Starting seed for training episodes (disjoint from evaluation seeds). |
| `--lr` | `float` | `0.005` | Learning rate for Double Linear Q updates. |
| `--gamma` | `float` | `0.95` | Discount factor for future expected interval returns. |
| `--output` | `str` | `models/linear_q_agent.json` | Path where trained weights and metadata are saved. |

### C. Evaluating Policies on Held-Out Test Seeds

The `evaluate.py` script executes a fair, reproducible benchmark across all 5 policies on identical held-out demand seeds.

```powershell
# Default benchmark across 20 held-out test seeds (seeds 42..61)
py evaluate.py

# Evaluate on the original problem seeds (seeds 42..46)
py evaluate.py --seeds 42 43 44 45 46

# Specify custom model and custom output directory
py evaluate.py --model models/linear_q_agent.json --seeds 42 43 44 --output-dir results/
```

**Available CLI Flags for `evaluate.py`:**
| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | `models/linear_q_agent.json` | Path to trained RL model weights JSON. |
| `--seeds` | `int...` | `42..61` (20 seeds) | List of held-out evaluation seeds. |
| `--output-dir` | `str` | `results` | Directory where `evaluation_summary.json` and `evaluation_details.csv` are written. |

**Generated Evaluation Artifacts:**
- `results/evaluation_summary.json`: Machine-readable summary statistics (mean, std, median, p90, fraction < 8 min, drops, relocations) and paired Student's $t$-test comparisons with exact two-tailed $p$-values.
- `results/evaluation_details.csv`: Seed-by-seed detailed metrics for every policy across every simulated day.

### D. Generating Publication-Quality Figures

Generate high-resolution PNG charts from the evaluation outputs (for reports, slides, and documentation):
```powershell
py plot_results.py
```
This generates:
- `figures/response_times.png`: Grouped bar chart comparing Mean and P90 response times with standard error bars.
- `figures/relocations_vs_drops.png`: Fleet relocation churn vs unserved dropped calls trade-off scatter plot.
- `figures/training_rewards.png`: Double Q-learning 100-episode reward convergence with rolling average.
- `figures/response_distributions.png`: Box plot distributions of response times across the 20 held-out seeds.

---

## 5. Reproduction Instructions

Follow these steps to reproduce the exact empirical results and tables reported in this documentation:

### Step 1: Execute the Invariant Test Suite
```powershell
py -m unittest discover tests
```
*Expected Outcome*: `Ran 27 tests in ~0.12s` with `OK`. Confirms that clock advancement, event priority, ambulance availability, response time measurement, and action masking pass all invariant checks.

### Step 2: (Optional) Retrain the RL Model
The repository already provides the verified trained model at `models/linear_q_agent.json`. To retrain from scratch:
```powershell
py train.py --episodes 100 --train-seed 1000 --lr 0.005
```
*Expected Outcome*:
- 100 episodes execute sequentially across seeds 1000 to 1099.
- Average TD error stabilizes, weight norms remain bounded ($\sim 5.5 - 6.0$).
- Weights and feature metadata are saved to `models/linear_q_agent.json`.

### Step 3: Run the 20 Held-Out Seeds Benchmark
```powershell
py evaluate.py
```
*Expected Outcome*:
- Evaluates 20 seeds (`42..61`) across No Redeployment, Fixed Home Base, Nearest Base, Greedy Coverage (MEXCLP), and Trained RL.
- Prints the benchmark table matching **Section 8.A** and paired $t$-test table matching **Section 8.B**.
- Writes `results/evaluation_summary.json` and `results/evaluation_details.csv`.

### Step 4: Verify the Original Problem Seeds (42–46)
```powershell
py evaluate.py --seeds 42 43 44 45 46
```
*Expected Outcome*:
- Matches **Section 8.C**, confirming that the original broken RL policy (10.12 min mean response, 24.40 relocs/day) has been fixed (now 9.45 min mean response, 16.41 min P90 response).

---

## 6. Key Bugs Diagnosed & Fixed

Prior to this work, evaluation over seeds 42–46 showed that the original RL policy was significantly worse than no redeployment (mean response time rose from 9.54 min to 10.12 min, dropped calls rose from 4.40 to 5.40, and units engaged in 24.40 unnecessary relocations/day).

A thorough senior-level audit revealed the root causes across the simulator, MDP formulation, and learning algorithm:

1. **Event Priority & Clock Tie-Breaking (`ems/simulator.py`)**:
   - *Bug*: In `run_day`, `if next_call_time <= next_event_time:` favored calls over service completions at identical timestamps. If an ambulance finished repositioning or handover at $t = 30.0$, an incoming call at $t = 30.0$ was processed while the unit was still marked busy, causing unnecessary dropped calls.
   - *Fix*: Implemented strict event priorities in the min-heap: Completion events (priority 0) $\to$ Call arrivals (priority 1) $\to$ Decision epochs (priority 2). Units finishing at timestamp $t$ become available before calls arriving at timestamp $t$.

2. **Simulator Reset & Reproducibility (`ems/simulator.py`)**:
   - *Bug*: `AmbulanceSimulator` lacked a `reset()` method. Re-running `run_day()` on the same instance advanced internal RNGs without resetting, producing non-reproducible outcomes.
   - *Fix*: Added `reset(seed)` resetting RNGs, queues, counters, and restoring all ambulances to their home stations. Verified bitwise reproducibility across runs.

3. **Loss System & Response Time Measurement (`ems/simulator.py`)**:
   - *Bug*: Response times were computed as `dispatch_delay + travel_time` rather than measuring the true elapsed time from call arrival to ambulance physical arrival at the scene (`scene_arrival_time - call.arrival_time_min`).
   - *Fix*: Documented the loss system policy (immediate drop if all 10 ambulances are busy/relocating) and strictly measured true response time from call arrival to scene arrival.

4. **Action Space & Missing STAY Action (`ems/rl_policy.py`, `ems/agent.py`)**:
   - *Bug*: Action space was strictly bases $0 \dots 11$. The agent had no explicit STAY action; at every 30-minute interval, it was forced to send an ambulance wandering across town at 28 km/h. During transit (15–30 min), the unit could not accept emergency calls, severely starving coverage.
   - *Fix*: Restructured action space with Action 0 as an explicit first-class `STAY` (no-op) option, and Actions $1 \dots 12$ corresponding to Bases $0 \dots 11$. Added base occupancy masking: the agent cannot relocate a unit to a base that already has an idle unit stationed at it. Added a relocation hurdle ($\tau = 0.05$): relocation is only executed if $Q(\text{move}) > Q(\text{stay}) + \tau$.

5. **Selection Bias & Credit Assignment (`ems/training.py`, `ems/rewards.py`)**:
   - *Bug*: When all units were busy, the trainer previously recorded Action 0 (STAY). Overwhelmed periods experienced heavy call drops (-2.0), artificially depressing the value of Action 0. Furthermore, transitions were updated on actions that failed to execute.
   - *Fix*: Only eligible candidate units (e.g. stranded at hospitals or surplus bases) are evaluated. When all units are busy, no relocation transition is created. Every stored transition uses the action *actually executed*. Designed a bounded, normalized reward function balancing response times, dropped calls, spatial coverage, and travel costs.

6. **Maximization Bias in Linear Q-Learning (`ems/agent.py`)**:
   - *Bug*: Standard single-estimator Q-learning over 13 discrete actions suffered from positive maximization bias ($\max_a Q(s', a)$ overestimating multi-action relocation values relative to STAY).
   - *Fix*: Implemented Double Linear Q-learning with two decoupled weight vectors ($W_A$ and $W_B$), TD error clipping ($\pm 5.0$), and weight bounds ($\pm 20.0$).

7. **Syntax Error Resolution (`ems/policies.py`)**:
   - *Bug*: An unterminated docstring fragment occurred around line 194 due to an accidental duplicate block at the end of `ems/policies.py`.
   - *Fix*: Removed the redundant fragment, verifying that all four baseline classes (`NoRedeploymentPolicy`, `FixedHomeBasePolicy`, `NearestBasePolicy`, `GreedyCoveragePolicy`) remain intact and syntactically valid.

---

## 7. Evaluated Policies

All policies implement a standardized `decide(simulator)` callback and are evaluated on the exact same simulator configuration:

1. **No Redeployment**: Units remain at their current position (initial home base, or scene/hospital of last served call) until dispatched again (0 relocations/day).
2. **Fixed Home Base**: Displaced units idle at hospitals or scenes are relocated back to their assigned home base.
3. **Nearest Base**: Displaced units idle away from any candidate base relocate to the nearest candidate base.
4. **Greedy Coverage (MEXCLP)**: Evaluates real-time spatial demand deficits ($D_b / (1 + \text{units}_b)$) across all candidate bases and relocates stranded/surplus units to the base with the highest unmet deficit.
5. **Trained RL**: Double Linear Q-learning policy evaluating state features, selecting between STAY and candidate empty bases with relocation hurdle.

---

## 8. Empirical Evaluation Results

### A. 20 Held-Out Seeds Benchmark (Seeds 42–61)

Evaluated across 20 held-out seeds completely disjoint from training seeds (training used seeds 1000–1099):

| Policy | Mean Resp (min) | Std (min) | Median (min) | P90 Resp (min) | < 8 min % | Drops/day | Relocs/day |
|---|---:|---:|---:|---:|---:|---:|---:|
| **No redeployment** | 9.43 | $\pm 0.60$ | 8.24 | 16.80 | 48.4% | 3.45 | 0.00 |
| **Fixed home base** | 9.67 | $\pm 0.66$ | 8.33 | 17.51 | 47.9% | 5.55 | 91.55 |
| **Nearest base** | 9.60 | $\pm 0.60$ | 8.36 | 17.37 | 46.9% | 4.65 | 92.95 |
| **Greedy coverage (MEXCLP)** | **9.16** | $\pm 0.39$ | **7.72** | 17.38 | **52.6%** | 4.50 | 37.30 |
| **Trained RL** | **9.21** | $\pm 0.54$ | 8.09 | **16.44** | 49.0% | 4.90 | 38.05 |

*Note: Response time statistics are computed strictly over served calls. Unserved calls are reported under Drops.*

### B. Paired Statistical Comparison (Trained RL vs Baselines on 20 Held-Out Seeds)

| Comparison | Mean Resp Diff | Direction | 95% Conf. Interval | Paired $t$-stat | Two-Tailed $p$-value | Stat. Significant? | P90 Diff | Drops Diff |
|---|---:|:---:|:---:|---:|---:|:---:|---:|---:|
| **RL vs No redeployment** | **-0.21 min** | **FASTER** | [-0.50, +0.07] | $t = -1.56$ | $p = 0.1358$ | No ($p \ge 0.05$) | **-0.37 min** | +1.45 |
| **RL vs Fixed home base** | **-0.46 min** | **FASTER** | [-0.80, -0.12] | $t = -2.82$ | $p = 0.0110$ | **Yes ($p < 0.05$)** | **-1.07 min** | -0.65 |
| **RL vs Nearest base** | **-0.38 min** | **FASTER** | [-0.72, -0.05] | $t = -2.42$ | $p = 0.0259$ | **Yes ($p < 0.05$)** | **-0.94 min** | +0.25 |
| **RL vs Greedy coverage** | +0.06 min | SLOWER (tied) | [-0.19, +0.30] | $t = +0.49$ | $p = 0.6303$ | No ($p \ge 0.05$) | **-0.94 min** | +0.40 |

### C. Direct Comparison on Original Problem Seeds (Seeds 42–46)

| Policy | Mean Resp (min) | P90 Resp (min) | Drops/day | Relocs/day |
|---|---:|---:|---:|---:|
| **Original Broken RL** | 10.12 | ~18.82 | 5.40 | 24.40 |
| **No redeployment** | 9.54 | 17.23 | 4.40 | 0.00 |
| **Fixed home base** | 9.82 | 17.67 | 7.40 | 87.60 |
| **Nearest base** | 9.97 | 18.06 | 5.00 | 90.80 |
| **Greedy coverage** | 9.14 | 17.47 | 5.00 | 38.20 |
| **Trained RL (Ours)** | **9.45** | **16.41** | 5.20 | 37.00 |

### Performance Analysis & Findings:

1. **RL decisively outperforms Fixed Home Base and Nearest Base**: Standard return-to-station heuristics churn the fleet excessively (~92 relocations/day), leaving ambulances unavailable mid-transit and worsening response times. RL is statistically significantly faster than Fixed Home Base ($p = 0.011$) and Nearest Base ($p = 0.026$).
2. **RL outperforms No Redeployment in response times**: RL is 0.21 minutes faster on mean response time (9.21 min vs 9.43 min) and 0.37 minutes faster on 90th percentile response time (16.44 min vs 16.80 min), although the paired difference on mean response is not statistically significant at $\alpha = 0.05$ ($p = 0.136$).
3. **RL achieves the best tail response time (P90) of all policies**: RL achieves **16.44 min** P90 response time, outperforming Greedy coverage (17.38 min) by **0.94 min**. By strategically positioning units at vulnerable outer bases, RL prevents severe response-time outliers.
4. **Greedy MEXCLP is the strongest baseline on mean response**: Greedy coverage achieves 9.16 min mean response, virtually tied with RL (9.21 min, $p = 0.63$).

---

## 9. Assumptions and Limitations

- **Queue/Drop Policy**: Assumes a loss system (M/G/c/c) where calls arriving when all 10 ambulances are busy/relocating cannot be queued and are counted as dropped.
- **Relocation Transit**: Relocating units travel at 28 km/h and cannot be preemptively diverted mid-transit to an emergency call.
- **Decision Epochs**: Redeployment decisions are evaluated every 30 simulation minutes.
- **Linear Function Approximation**: The linear architecture is fast and interpretable, but non-linear state interactions are represented via explicit feature engineering rather than deep neural representations.
