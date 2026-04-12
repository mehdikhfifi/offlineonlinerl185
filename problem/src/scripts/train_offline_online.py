import argparse
import os
from datetime import datetime

import numpy as np
import torch
import tqdm

import configs
from agents import agents
from infrastructure import utils
from infrastructure import pytorch_util as ptu
from infrastructure.log_utils import setup_wandb, Logger, dump_log
from infrastructure.replay_buffer import ReplayBuffer


def _episode_length(env) -> int:
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
        return spec.max_episode_steps
    return getattr(env, "max_episode_steps", 1000)


def run_offline_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, start_step: int = 0):
    """
    Run offline training loop
    """
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    
    env, dataset = config["make_env_and_dataset"]()

    example_batch = dataset.sample(1)
    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        example_batch['observations'].shape[1:],
        example_batch['actions'].shape[-1],
        **config["agent_kwargs"],
    )

    ep_len = _episode_length(env)

    for step in tqdm.trange(config["offline_training_steps"] + 1, dynamic_ncols=True):
        
        batch = dataset.sample(config["batch_size"])

        batch = {
            k: ptu.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in batch.items()
        }

        metrics = agent.update(
            batch["observations"],
            batch["actions"],
            batch["rewards"],
            batch["next_observations"],
            batch["dones"],
            step,
        )

        if step % args.log_interval == 0:
            train_logger.log(metrics, step=step)

        if step % args.eval_interval == 0:
            
            trajectories = utils.sample_n_trajectories(
                env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            
            successes = [t["episode_statistics"]["s"] for t in trajectories]

            eval_logger.log(
                {
                    "eval/success_rate": float(np.mean(successes)),
                },
                step=step,
            )

    return dump_log(agent, train_logger, eval_logger, config, args.save_dir)


def run_online_training_loop(config: dict, train_logger, eval_logger, args: argparse.Namespace, agent_path: str, start_step: int = 0):
    """
    Run online training loop
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    env, _ = config["make_env_and_dataset"]()
    eval_env, _ = config["make_env_and_dataset"]()

    ep_len = _episode_length(env)

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    agent_cls = agents[config["agent"]]
    agent = agent_cls(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )
    agent.load_state_dict(torch.load(agent_path, weights_only=True))

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation, _ = env.reset()

    for step in tqdm.trange(config["online_training_steps"], dynamic_ncols=True):
        log_step = start_step + step

        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=terminated,
        )

        if done:
            observation, _ = env.reset()
        else:
            observation = next_observation

        if len(replay_buffer) >= config["batch_size"]:
            batch = replay_buffer.sample(config["batch_size"])
            batch = ptu.from_numpy(batch)
            update_info = agent.update(
                batch["observations"],
                batch["actions"],
                batch["rewards"],
                batch["next_observations"],
                batch["dones"],
                log_step,
            )

            if log_step % args.log_interval == 0:
                train_logger.log(update_info, step=log_step)

        if log_step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                agent,
                args.num_eval_trajectories,
                ep_len,
            )
            successes = [t["episode_statistics"]["s"] for t in trajectories]

            eval_logger.log(
                {
                    "eval/success_rate": float(np.mean(successes)),
                },
                step=log_step,
            )

            dump_log(agent, train_logger, eval_logger, config, args.save_dir)

    return dump_log(agent, train_logger, eval_logger, config, args.save_dir)


def setup_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", type=str, default='sacbc')
    parser.add_argument("--env_name", type=str, default='cube-single-play-singletask-task1-v0')
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_group", type=str, default='Debug')
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--which_gpu", default=0)
    parser.add_argument("--offline_training_steps", type=int, default=500000)  
    parser.add_argument("--online_training_steps", type=int, default=100000)  
    parser.add_argument("--replay_buffer_capacity", type=int, default=1000000)
    parser.add_argument("--log_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=100000)
    parser.add_argument("--num_eval_trajectories", type=int, default=25)  
    
    
    parser.add_argument("--keep_off_data", type=bool, default=False)
    
    
    
    parser.add_argument("--wsrl", type=bool, default=False)
    
    
    
    parser.add_argument("--expectile", type=float, default=None)

    
    parser.add_argument("--alpha", type=float, default=None)

    
    parser.add_argument("--inv_temp", type=float, default=None)

    
    parser.add_argument("--noise_scale", type=float, default=None)

    
    parser.add_argument("--njobs", type=int, default=None)
    parser.add_argument("job_specs", nargs="*")

    args = parser.parse_args(args=args)

    return args


def main(args):
    # Default "exp/..." locally; Modal sets OFFLINE_ONLINE_LOGDIR to the mounted volume root.
    logdir_prefix = os.environ.get("OFFLINE_ONLINE_LOGDIR", "exp")

    config = configs.configs[args.base_config](args.env_name)

    
    config['seed'] = args.seed
    config['run_group'] = args.run_group
    config['offline_training_steps'] = args.offline_training_steps
    config['online_training_steps'] = args.online_training_steps
    config['log_interval'] = args.log_interval
    config['eval_interval'] = args.eval_interval
    config['num_eval_trajectories'] = args.num_eval_trajectories
    config['replay_buffer_capacity'] = args.replay_buffer_capacity
    
    

    exp_name = f"sd{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config['log_name']}"

    
    if args.expectile is not None:
        config['agent_kwargs']['expectile'] = args.expectile
        exp_name = f"{exp_name}_e{args.expectile}"
    if args.alpha is not None:
        config['agent_kwargs']['alpha'] = args.alpha
        exp_name = f"{exp_name}_a{args.alpha}"
    if args.inv_temp is not None:
        config['agent_kwargs']['inv_temp'] = args.inv_temp
        exp_name = f"{exp_name}_i{args.inv_temp}"
    if args.noise_scale is not None:
        config['agent_kwargs']['noise_scale'] = args.noise_scale
        exp_name = f"{exp_name}_n{args.noise_scale}"
    if args.online_training_steps > 0:
        exp_name = f"{exp_name}_online"
    if args.offline_training_steps > 0:
        exp_name = f"{exp_name}_offline"

    if args.online_training_steps > 0 and args.offline_training_steps == 0:
        raise ValueError(
            "Online training needs a checkpoint from offline training. "
            "Use --offline_training_steps > 0 in the same run (recommended), or add resume-from-checkpoint support."
        )

    setup_wandb(project='cs185_default_project', name=exp_name, group=args.run_group, config=config)
    args.save_dir = os.path.join(logdir_prefix, args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    train_logger = Logger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = Logger(os.path.join(args.save_dir, 'eval.csv'))

    start_step = 0
    agent_path = None
    if args.offline_training_steps > 0:
        print(f"Running offline training loop with {args.offline_training_steps} steps")
        agent_path = run_offline_training_loop(config, train_logger, eval_logger, args, start_step=0)
        start_step = args.offline_training_steps

    if args.online_training_steps > 0:
        assert agent_path is not None
        print(f"Running online training loop with {args.online_training_steps} steps")
        run_online_training_loop(config, train_logger, eval_logger, args, agent_path, start_step=start_step)


if __name__ == "__main__":
    args = setup_arguments()
    if args.njobs is not None and len(args.job_specs) > 0:
        from scripts.run_njobs_offline_online import main_njobs

        main_njobs(job_specs=args.job_specs, njobs=args.njobs)
    else:
        main(args)