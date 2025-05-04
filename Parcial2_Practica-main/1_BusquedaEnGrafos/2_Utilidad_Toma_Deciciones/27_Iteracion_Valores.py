class InventoryOptimization:
    def __init__(self, initial_inventory, max_inventory, holding_cost, purchase_cost, selling_price, demand_probabilities):
        self.initial_inventory = initial_inventory
        self.max_inventory = max_inventory
        self.holding_cost = holding_cost
        self.purchase_cost = purchase_cost
        self.selling_price = selling_price
        self.demand_probabilities = demand_probabilities  # Probabilidades de demanda diaria
        self.value_function = [0] * (max_inventory + 1)  # Función de valor inicializada a 0

    def expected_reward(self, state, action):
        # Calcular la recompensa esperada de tomar una acción en un estado dado
        reward = 0
        new_inventory = state + action

        # Costo de compra y mantenimiento
        reward -= action * self.purchase_cost
        reward -= new_inventory * self.holding_cost

        # Calcular la recompensa esperada para cada nivel de demanda
        for demand, probability in self.demand_probabilities.items():
            sales = min(new_inventory, demand)
            reward += probability * (sales * self.selling_price)
            new_inventory -= sales

        return reward, new_inventory

    def value_iteration(self, gamma=0.9, epsilon=1e-3):
        # Iteración de valores
        while True:
            delta = 0
            for state in range(self.max_inventory + 1):
                max_value = float('-inf')
                for action in range(self.max_inventory - state + 1):
                    reward, new_inventory = self.expected_reward(state, action)
                    value = reward + gamma * self.value_function[new_inventory]
                    if value > max_value:
                        max_value = value
                delta = max(delta, abs(self.value_function[state] - max_value))
                self.value_function[state] = max_value
            if delta < epsilon:
                break

    def optimal_policy(self):
        policy = {}
        for state in range(self.max_inventory + 1):
            max_value = float('-inf')
            best_action = 0
            for action in range(self.max_inventory - state + 1):
                reward, new_inventory = self.expected_reward(state, action)
                value = reward + self.value_function[new_inventory]
                if value > max_value:
                    max_value = value
                    best_action = action
            policy[state] = best_action
        return policy

# Parámetros de la caja registradora
initial_inventory = 5
max_inventory = 10
holding_cost = 1  # Costo de mantener un bien en inventario por día
purchase_cost = 3  # Costo de compra de un bien
selling_price = 5  # Precio de venta de un bien
demand_probabilities = {0: 0.1, 1: 0.3, 2: 0.4, 3: 0.2}  # Probabilidades de demanda diaria

# Crear el optimizador de inventario y ejecutar la iteración de valores
optimizer = InventoryOptimization(initial_inventory, max_inventory, holding_cost, purchase_cost, selling_price, demand_probabilities)
optimizer.value_iteration()
optimal_policy = optimizer.optimal_policy()

# Mostrar la política óptima
print("Política óptima de compra:")
for state, action in optimal_policy.items():
    print(f"Si el inventario es {state}, comprar {action} bienes.")
