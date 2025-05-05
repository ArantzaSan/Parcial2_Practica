import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

# Crear un entorno de ejemplo (por ejemplo, CartPole)
entorno = DummyVecEnv([lambda: gym.make('CartPole-v1')])

# Generar datos de interacciones (esto normalmente se haría fuera de línea)
def generar_datos_offline(entorno, num_pasos=1000):
    observacion = entorno.reset()
    acciones = []
    observaciones = []
    recompensas = []
    terminados = []
    siguientes_observaciones = []

    for _ in range(num_pasos):
        accion = entorno.action_space.sample()  # Acción aleatoria
        siguiente_obs, recompensa, terminado, _ = entorno.step(accion)

        acciones.append(accion)
        observaciones.append(observacion)
        recompensas.append(recompensa)
        terminados.append(terminado)
        siguientes_observaciones.append(siguiente_obs)

        observacion = siguiente_obs
        if terminado:
            observacion = entorno.reset()

    return {
        'observaciones': np.array(observaciones),
        'acciones': np.array(acciones),
        'recompensas': np.array(recompensas),
        'terminados': np.array(terminados),
        'siguientes_observaciones': np.array(siguientes_observaciones)
    }

# Generar datos de interacciones
datos_offline = generar_datos_offline(entorno)

# Crear un ReplayBuffer con los datos de interacciones
buffer_experiencia = ReplayBuffer(
    buffer_size=len(datos_offline['observaciones']),
    observation_space=entorno.observation_space,
    action_space=entorno.action_space,
    device='auto'
)

for i in range(len(datos_offline['observaciones'])):
    buffer_experiencia.add(
        datos_offline['observaciones'][i],
        datos_offline['siguientes_observaciones'][i],
        datos_offline['acciones'][i],
        datos_offline['recompensas'][i],
        datos_offline['terminados'][i],
        [0.0]  # Info adicional, no utilizada aquí
    )

# Crear un modelo DQN
modelo = DQN('MlpPolicy', entorno, replay_buffer=buffer_experiencia, verbose=1)

# Entrenar el modelo utilizando los datos de interacciones
modelo.learn(total_timesteps=10000)

# Guardar el modelo entrenado
modelo.save("dqn_offline")

print("Modelo entrenado y guardado.")
