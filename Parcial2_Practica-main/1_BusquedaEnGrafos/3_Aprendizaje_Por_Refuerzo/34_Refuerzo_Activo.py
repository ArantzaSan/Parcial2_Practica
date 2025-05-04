import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Crear un entorno de ejemplo (por ejemplo, CartPole)
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# Crear un modelo DQN
model = DQN('MlpPolicy', env, verbose=1)

# Entrenar el modelo
model.learn(total_timesteps=10000)

# Guardar el modelo entrenado
model.save("dqn_cartpole")

# Evaluar el modelo entrenado
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Cargar el modelo entrenado y realizar una prueba
model = DQN.load("dqn_cartpole")

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
