import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Crear un entorno de ejemplo (por ejemplo, CartPole)
entorno = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# Crear un modelo DQN
modelo = DQN('MlpPolicy', entorno, verbose=1)

# Entrenar el modelo
modelo.learn(total_timesteps=10000)

# Guardar el modelo entrenado
modelo.save("dqn_cartpole")

# Evaluar el modelo entrenado
recompensa_media, desviacion_estandar = evaluate_policy(modelo, entorno, n_eval_episodes=10)
print(f"Recompensa media: {recompensa_media} +/- {desviacion_estandar}")

# Cargar el modelo entrenado y realizar una prueba
modelo_cargado = DQN.load("dqn_cartpole")

observacion = entorno.reset()
for _ in range(1000):
    accion, _estados = modelo_cargado.predict(observacion)
    observacion, recompensas, terminados, informacion = entorno.step(accion)
    entorno.render()
