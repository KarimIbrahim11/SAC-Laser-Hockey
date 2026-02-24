import torch
import random
import copy
from datetime import datetime
import os
import submodules.hockey_env.hockey.hockey_env as h_env
import numpy as np
from gymnasium import spaces
import optparse
import pickle

from agent import SACAgent, SelfPlayOpponent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)

_DEBUG_MODE = True

def get_distance_p1_to_puck(obs):
    """
    Calculate Euclidean distance between Player 1 and Puck based on indices:
    Player 1 pos: obs[0], obs[1]
    Puck pos:     obs[12], obs[13]
    """
    p1_pos = obs[0:2]
    puck_pos = obs[12:14]
    return np.linalg.norm(p1_pos - puck_pos)

def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="Hockey-selfplay",
                         help='Environment (default %default)')
    optParser.add_option('-t', '--train', action='store', type='int',
                         dest='train', default=1,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr', action='store', type='float',
                         dest='lr', default=0.0001,
                         help='learning rate for actor/policy (default %default)')
    optParser.add_option('-b', '--batch_size', action='store', type='int',
                         dest='batch_size', default=512)
    optParser.add_option('-m', '--maxepisodes', action='store', type='float',
                         dest='max_episodes', default=25000,
                         help='number of episodes (default %default)')
    optParser.add_option('-s', '--seed', action='store', type='int',
                         dest='seed', default=None,
                         help='random seed (default %default)')
    optParser.add_option('--static', action='store_false', dest='auto_alpha', default=True,
                         help='Disable automatic alpha tuning (use fixed initial_alpha)')
    optParser.add_option('--alpha', action='store', type='float', dest='alpha_val', default=0.2,
                            help='Initial or fixed alpha value')
    optParser.add_option('-d', '--debug', action='store_true', dest='debug', default=False,
                         help='Debug mode for logging')
    optParser.add_option('--finetune', action='store_true', default=False,
                        dest='finetune', help='Finetune an existing model')
    optParser.add_option('-c', '--ckpt', action='store', type='string',
                        dest='ckpt_path', default="results/agent.pth",
                         help='Default ckpt to load')
    optParser.add_option('--selfplay', action='store_true', dest='selfplay', default=False,
                         help='Enable self-play training')
    opts, args = optParser.parse_args()

    render = False
    log_interval = 100
    max_episodes = int(opts.max_episodes)
    max_timesteps = 250
    initial_random_steps =  100000
    train_iter = opts.train
    lr = opts.lr
    batch_size = opts.batch_size
    random_seed = opts.seed
    _DEBUG_MODE = opts.debug
    total_timesteps = 0
    env_name = opts.env_name
    env = h_env.HockeyEnv(mode=0)
    
    # Loading agent and ckpt
    agent_action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    sac = SACAgent(env.observation_space, 
                   agent_action_space, 
                   learning_rate_actor=lr, 
                   batch_size=batch_size,
                #    auto_alpha=opts.auto_alpha,
                   initial_alpha=opts.alpha_val)

    if opts.finetune:
        checkpoint = torch.load(opts.ckpt_path, weights_only=False)
        print(f"Loaded Model: {opts.ckpt_path}")
        initial_random_steps = 0
        sac.restore_state(checkpoint['model_state_dict'])
        with torch.no_grad():
            target_alpha = opts.alpha_val
            sac.log_alpha.fill_(np.log(target_alpha))

    # Choosing opponent
    if env_name == "Hockey-weak":
        opponent = h_env.BasicOpponent(weak=True)
    elif env_name == "Hockey-strong":
        opponent = h_env.BasicOpponent(weak=False)
    elif env_name == "Hockey-selfplay" and opts.selfplay:
        opponent_pool =[]
        if opts.finetune and opts.ckpt_path:
            opponent = SelfPlayOpponent(sac)
            opponent_pool.append(opponent)
            opponent_pool.append(h_env.BasicOpponent(weak=False))
        else:
            opponent = h_env.BasicOpponent(weak=False)
            opponent_pool.append(opponent)
    else:
        opponent = h_env.BasicOpponent(weak=True)

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists("./results"):
        os.makedirs("./results")

    # Logging variables
    total_rewards = []
    total_lengths = []
    total_losses = {"q_loss": [], "actor_loss": [], "alpha_loss": [], "alpha": [], "entropy": [],
                    "q_mean": [], "q_std": [], "target_mean": []}
    total_results = {"wins": 0, "losses": 0, "draws": 0}
    train_started_flag = False

    def save_statistics(episode_num, is_final=False):
        current_alpha = sac.alpha.item()
        suffix = "final" if is_final else f"ep{episode_num}"
        filename = f"./results/SAC_{env_name}_{run_id}_LR{lr}_B{batch_size}_alpha{current_alpha:.4f}_{suffix}.pkl"
        
        stats_data = {
            "metadata": {
                "date": run_id,
                "env": env_name,
                "learning_rate": lr,
                "batch_size": batch_size,
                "seed": random_seed,
                "alpha_final": current_alpha,
                "total_episodes": episode_num
            },
            "rewards": total_rewards,
            "lengths": total_lengths,
            "losses": total_losses,
            "results": total_results
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(stats_data, f)
        print(f"Stats saved to: {filename}")


    for i_episode in range(1, max_episodes + 1):        
        if opts.selfplay:
            opponent = random.choice(opponent_pool)

        ob, _info = env.reset()
        
        # Distance to puck
        prev_dist = get_distance_p1_to_puck(ob)
        
        episode_rewards = 0
        episode_shaping_reward = 0
        for t in range(max_timesteps+1):
            total_timesteps += 1
            # if total_timesteps < initial_random_steps or True: # Keep updating usually
            #     sac.obs_normalizer.update(ob)
                
            obs_scaled = sac.obs_normalizer.normalize(ob)
            obs_opponent = env.obs_agent_two()
            
            # SAC Action (Left player)
            if total_timesteps < initial_random_steps:
                sac.obs_normalizer.update(ob)
                a1 = agent_action_space.sample()
            else:
                # After initial_random_steps
                if total_timesteps == initial_random_steps:
                    print(f"Obs stats: mean={sac.obs_normalizer.mean[:5]}, std={np.sqrt(sac.obs_normalizer.var[:5])}")
                a1 = sac.act(obs_scaled, deterministic=False)
            
            # Basic Opponent Action
            a2 = opponent.act(obs_opponent)
            
            # Combine to 8D
            combined_action = np.hstack([a1, a2])
            
            new_ob,  env_reward, done, trunc, _info = env.step(combined_action)
            new_obs_scaled = sac.obs_normalizer.normalize(new_ob)
            
            # Distance to puck currently (from prev timestamp)
            shaping_reward = 0
            current_dist = get_distance_p1_to_puck(new_obs_scaled)
            dist_change = prev_dist - current_dist
            shaping_reward += dist_change * 0.3 

            # # Dense shaping (keep small)
            if "reward_touch_puck" in _info and _info["reward_touch_puck"] > 0:
                shaping_reward += 0.5
            # # if "reward_closeness_to_puck" in _info:
            # #     shaping_reward += _info["reward_closeness_to_puck"] * 0.1
            if "reward_puck_direction" in _info:
                shaping_reward += _info["reward_puck_direction"] * 0.3
            
            reward = env_reward + shaping_reward
            
            if done or trunc:
                winner = _info.get("winner", 0)
                if winner == 1:
                    # reward += 5.0  # Win
                    total_results["wins"] += 1
                elif winner == -1:
                    # reward += -5.0  # Loss
                    total_results["losses"] += 1
                else:
                    # reward += -3.5
                    total_results["draws"] += 1
            
            # Distance to puck.
            prev_dist = current_dist           
            
            if i_episode % log_interval == 0 and t == max_timesteps:
                print(f"DEBUG REWARDS Ep {i_episode} | Env: {reward:6.2f} | Shaping: {shaping_reward:6.2f} | Total: {reward:6.2f}")
            
            # Store the agent's action (4D) in buffer
            sac.store_transition((obs_scaled, a1, reward, new_obs_scaled, done))
            
            # Train if buffer has enough samples
            if sac.buffer.size > opts.batch_size and total_timesteps >= initial_random_steps:
                if not train_started_flag:
                    train_started_flag = True
                    print(f"*******************Training Started episode:{i_episode}, t:{total_timesteps}******************")
                loss_stats = sac.train(train_iter, i_episode)
                total_losses["q_loss"].append(loss_stats["q_loss"])
                total_losses["actor_loss"].append(loss_stats["actor_loss"])
                total_losses["alpha_loss"].append(loss_stats["alpha_loss"])
                total_losses["alpha"].append(loss_stats["alpha"])
                total_losses["entropy"].append(loss_stats["entropy"])
            
            ob = new_ob
            episode_rewards += reward
            episode_shaping_reward += shaping_reward
            
            if done or trunc:
                if _DEBUG_MODE and i_episode % log_interval == 0:
                    winner = _info.get("winner", 0)
                    print(f"EPISODE {i_episode} SUMMARY: Winner: {winner}")
                    print(f"  Episode Reward: {episode_rewards:7.2f}")
                    print(f"  Episode Shaping Reward : {episode_shaping_reward:7.2f}")
                break

        total_rewards.append(episode_rewards)
        total_lengths.append(t)

        # Save every 500 episodes
        if i_episode % 500 == 0:
            current_alpha = sac.alpha.item()
            model_filename = (
                f"./results/SAC_{env_name}_{run_id}_"
                f"EP{i_episode}_A{current_alpha:.4f}.pth"
            )
            
            print(f"\n--- Checkpoint at Episode {i_episode} ---")
            torch.save({
                'episode': i_episode,
                'model_state_dict': sac.state(),
                'optimizer_state_dict': sac.optimizer.state_dict(),
                'alpha': current_alpha,
            }, model_filename)
            
            save_statistics(i_episode)

    save_statistics(max_episodes, is_final=True)
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
if __name__ == '__main__':
    main()