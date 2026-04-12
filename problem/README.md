# Offline to Online RL Final Project

## Setup

For general setup and Modal instructions, see Homework 1's README.

## Examples

Run commands from this `problem` directory (same layout as the course `final_project_offline_online` folder).

### Section 3: offline → online (`train_offline_online.py`)

Full project spec (500k offline + 100k online on `cube-single`):

```bash
# SAC+BC — sweep α ∈ {30, 100, 300, 1000}; aim for >75% eval success offline
uv run src/scripts/train_offline_online.py --run_group=s1_sacbc --base_config=sacbc \
  --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=100 \
  --offline_training_steps=500000 --online_training_steps=100000

# FQL — same α sweep; aim for >80% eval success offline
uv run src/scripts/train_offline_online.py --run_group=s1_fql --base_config=fql \
  --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=100 \
  --offline_training_steps=500000 --online_training_steps=100000
```

Use a second seed (e.g. `--seed=1`) for the report’s two-seed requirement where applicable.

### HW-style offline-only script (`run.py`)

* To run on a local machine:
  ```bash
  uv run src/scripts/run.py --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0
  ```


* To run Section 3 on Modal (saves under the Modal volume; `OFFLINE_ONLINE_LOGDIR` is set in `modal_run.py`):
  ```bash
  uv run modal run src/scripts/modal_run.py --run_group=s1_sacbc --base_config=sacbc \
    --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=100 \
    --offline_training_steps=500000 --online_training_steps=100000
  ```
  * You may request a different GPU type, CPU count, and memory size by changing variables in `src/scripts/modal_run.py`
  * Use `modal run --detach` to keep your job running in the background.


* To run four α sweeps in parallel on Modal (`run_njobs` uses `train_offline_online.py`):
  ```bash
  uv run modal run src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=30 --offline_training_steps=500000 --online_training_steps=100000" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100 --offline_training_steps=500000 --online_training_steps=100000" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300 --offline_training_steps=500000 --online_training_steps=100000" \
  "JOB --run_group=s1_sacbc --base_config=sacbc --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000 --offline_training_steps=500000 --online_training_steps=100000"
  ```
