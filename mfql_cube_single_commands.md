







We will compare mmfql with different one step size widths on cubesingle

# mfql commands on ant soccer k = 1
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 1" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 1" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 1" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 1"

# mfql commands on ant soccer k = 2
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 2" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 2" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 2" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 2"


# mfql commands on ant soccer k = 3
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 3" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 3" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 3" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 3"


# mfql commands on ant soccer k = 4
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 4" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 4" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 4" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 4"


# mfql commands on ant soccer k = 5
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 5" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 5" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 5" \
"JOB --run_group=s1_mfql --base_config=mfql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --k 5"