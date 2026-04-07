import gymnasium as gym


def main():
    env = gym.make("Pendulum-v1", render_mode="human")
    observation, info = env.reset(seed=42)

    total_reward = 0.0
    for step in range(500):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}, total reward: {total_reward}")
            break

    env.close()
    print(f"Final total reward: {total_reward}")


if __name__ == "__main__":
    main()
