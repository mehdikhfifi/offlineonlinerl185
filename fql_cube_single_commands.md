







We will compare fql with different one step size widths on cubesingle

# fql commands on ant soccer factor = 1
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 1" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 1" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 1" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 1"

# fql commands on ant soccer actor_factor = 2
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 2" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 2" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 2" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 2"


# fql commands on ant soccer factor = 3
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 3" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 3" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 3" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 3"


# fql commands on ant soccer factor = 4
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 4" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 4" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 4" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 4"


# fql commands on ant soccer factor = 5
uv run modal run --detach src/scripts/modal_run.py --njobs=4 \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=3000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 5" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=1000 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 5" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=100 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 5" \
"JOB --run_group=s1_fql --base_config=fql --env_name=cube-single-play-singletask-task1-v0 --seed=15 --alpha=30 --offline_training_steps 500000 --online_training_steps 100001 --eval_interval 50000 --log_interval 50000 --actor_factor 5"