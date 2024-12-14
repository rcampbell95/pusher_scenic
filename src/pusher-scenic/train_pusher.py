import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("Pusher-v4")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

model.save("PPO_pusher.zip")
