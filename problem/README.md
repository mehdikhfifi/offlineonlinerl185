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

* To run section 4.1 (Adding offline data in online step):
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=10 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000 --offline_data=50000"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=0 --alpha=10 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=1 --alpha=100 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=2 --alpha=300 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=3 --alpha=1000 --offline_data=50000"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s2_offline --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=0 --alpha=3 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --alpha=10 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=2 --alpha=30 --offline_data=50000" \
  "JOB --run_group=s2_offline --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=3 --alpha=100 --offline_data=50000"
```

* To run section 4.2 (WSRL):
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=0 --alpha=10 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=1 --alpha=100 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=2 --alpha=300 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=3 --alpha=1000 --wsrl_steps=25000"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=0 --alpha=30 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=1 --alpha=100 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=2 --alpha=300 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=cube-double-play-singletask-task1-v0 --seed=3 --alpha=1000 --wsrl_steps=25000"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=0 --alpha=3 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --alpha=10 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=2 --alpha=30 --wsrl_steps=25000" \
  "JOB --run_group=s2_wsrl --base_config=fql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=3 --alpha=100 --wsrl_steps=25000"
```

* To run section 4.3 (IFQL):
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=3 \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=cube-single-play-singletask-task1-v0 --seed=0 --expectile=0.85" \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=cube-single-play-singletask-task1-v0 --seed=1 --expectile=0.9" \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=cube-single-play-singletask-task1-v0 --seed=2 --expectile=0.95"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=3 \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=cube-double-play-singletask-task1-v0 --seed=0 --expectile=0.85" \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=cube-double-play-singletask-task1-v0 --seed=1 --expectile=0.9" \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=cube-double-play-singletask-task1-v0 --seed=2 --expectile=0.95"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=3 \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=0 --expectile=0.85" \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --expectile=0.9" \
  "JOB --run_group=s2_ifql --base_config=ifql --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=2 --expectile=0.95"
```

* To run section 4.4 (DSRL):
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=3 \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=cube-single-play-singletask-task1-v0 --seed=0 --noise_scale=0.8" \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=cube-single-play-singletask-task1-v0 --seed=1 --noise_scale=1.0" \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=cube-single-play-singletask-task1-v0 --seed=2 --noise_scale=1.2"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=3 \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=cube-double-play-singletask-task1-v0 --seed=0 --noise_scale=0.8" \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=cube-double-play-singletask-task1-v0 --seed=1 --noise_scale=1.0" \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=cube-double-play-singletask-task1-v0 --seed=2 --noise_scale=1.2"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=3 \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=0 --noise_scale=0.8" \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --noise_scale=1.0" \
  "JOB --run_group=s2_dsrl --base_config=dsrl --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=2 --noise_scale=1.2"
```

* To run section 4.5 (QSM): sweep temperature **η** (`--inv_temp`) and DDPM balance **α** (`--alpha`) over `{10, 30, 100}` each — **3×3 = 9 jobs** per environment (project outline §4.5).
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=9 \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=0 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=1 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=2 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=100" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=3 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=4 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=5 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=100" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=6 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=7 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-single-play-singletask-task1-v0 --seed=8 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=100"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=9 \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=0 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=1 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=2 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=100" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=3 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=4 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=5 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=100" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=6 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=7 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=cube-double-play-singletask-task1-v0 --seed=8 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=100"
```
```bash
uv run modal run --detach src/scripts/modal_run.py --njobs=9 \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=0 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=1 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=2 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=10 --alpha=100" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=3 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=4 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=5 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=30 --alpha=100" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=6 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=10" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=7 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=30" \
  "JOB --run_group=s2_qsm --base_config=qsm --env_name=antsoccer-arena-navigate-singletask-task1-v0 --seed=8 --offline_training_steps=500000 --online_training_steps=100000 --inv_temp=100 --alpha=100"
```