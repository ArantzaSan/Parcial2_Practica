import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

# Crear un entorno de ejemplo (por ejemplo, CartPole)
env = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# Generar datos de interacciones (esto normalmente se haría fuera de línea)
def generate_offline_data(env, num_steps=1000):
    obs = env.reset()
    actions = []
    observations = []
    rewards = []
    dones = []
    next_observations = []

    for _ in range(num_steps):
        action = env.action_space.sample()  # Acción aleatoria
        next_obs, reward, done, _ = env.step(action)

        actions.append(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        next_observations.append(next_obs)

        obs = next_obs
        if done:
            obs = env.reset()

    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'next_observations': np.array(next_observations)
    }

# Generar datos de interacciones
offline_data = generate_offline_data(env)

# Crear un ReplayBuffer con los datos de interacciones
replay_buffer = ReplayBuffer(
    buffer_size=len(offline_data['observations']),
    observation_space=env.observation_space,
    action_space=env.action_space,
    device='auto'
)

for i in range(len(offline_data['observations'])):
    replay_buffer.add(
        offline_data['observations'][i],
        offline_data['next_observations'][i],
        offline_data['actions'][i],
        offline_data['rewards'][i],
        offline_data['dones'][i],
        [0.0]  # Info adicional, no utilizada aquí
    )

# Crear un modelo DQN
model = DQN('MlpPolicy', env, replay_buffer=replay_buffer, verbose=1)

# Entrenar el modelo utilizando los datos de interacciones
model.learn(total_timesteps=10000)

# Guardar el modelo entrenado
model.save("dqn_offline")

print("Modelo entrenado y guardado.")
