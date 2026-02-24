import torch
import argparse
import numpy as np
import submodules.hockey_env.hockey.hockey_env as h_env
from agent import SACAgent


def evaluate(checkpoint_path, num_episodes=50, weak_opponent=False):
    env = h_env.HockeyEnv(mode=h_env.Mode.NORMAL)
    
    sac = SACAgent(
        env.observation_space, 
        h_env.spaces.Box(-1, 1, shape=(4,)), 
        hidden_sizes_actor=[256,256],
        hidden_sizes_critic= [256,256],
    )
    print(f"{'Loading Checkpoint:':<25} {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        weights = checkpoint['model_state_dict']
        ep = checkpoint.get('episode', '??')
        al = checkpoint.get('alpha', '??')
    else:
        weights = checkpoint


    try:
        sac.restore_state(weights)
        print("Successfully restored weights.")
    except Exception as e:
        print(f"Load Error: {e}")
    
    print("="*60 + "\n")
    
    # Setup the Opponent
    opponent = h_env.BasicOpponent(weak=weak_opponent)

    sac.actor_policy.eval()

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        trunc = False
        total_reward = 0.0
        step_count = 0
        
        while not (done or trunc):
            env.render() # Renders the game window
            
            scaled_obs = sac.obs_normalizer.normalize(obs)
            a1 = sac.act(scaled_obs, deterministic=True)

            obs_opponent = env.obs_agent_two()
            a2 = opponent.act(obs_opponent)

            obs, reward, done, trunc, info = env.step(np.hstack([a1, a2]))
            total_reward += reward
            step_count += 1
            
    env.close()
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC agent on Laser Hockey.")
    parser.add_argument("--checkpoint", type=str,
                        default="results/agent.pth",
                        help="Path to the agent checkpoint .pth file.")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of evaluation episodes.")
    parser.add_argument("--weak_opponent", action="store_true",
                        help="Use the weak opponent (default: strong opponent).")
    args = parser.parse_args()

    evaluate(args.checkpoint, num_episodes=args.num_episodes, weak_opponent=args.weak_opponent)