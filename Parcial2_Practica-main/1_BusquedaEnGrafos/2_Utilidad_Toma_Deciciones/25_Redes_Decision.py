class DecisionNode:
    def __init__(self, question, yes_branch, no_branch):
        self.question = question
        self.yes_branch = yes_branch
        self.no_branch = no_branch

    def decide(self):
        answer = input(self.question + " (si/no): ").strip().lower()
        if answer == 'si':
            return self.yes_branch.decide()
        elif answer == 'no':
            return self.no_branch.decide()
        else:
            print("Respuesta no válida. Por favor, responde con 'si' o 'no'.")
            return self.decide()

class ActionNode:
    def __init__(self, recommendation):
        self.recommendation = recommendation

    def decide(self):
        print(self.recommendation)
        return self.recommendation

# Crear nodos de decisión
root = DecisionNode(
    "¿Necesitas una fruta alta en vitamina C?",
    yes_branch=DecisionNode(
        "¿Prefieres algo dulce?",
        yes_branch=ActionNode("Deberías comer una naranja."),
        no_branch=ActionNode("Deberías comer un kiwi.")
    ),
    no_branch=DecisionNode(
        "¿Estás buscando una fruta baja en calorías?",
        yes_branch=ActionNode("Deberías comer una manzana."),
        no_branch=ActionNode("Deberías comer un plátano.")
    )
)

# Ejecutar la red de decisión
print("Bienvenido al asistente de selección de frutas.")
root.decide()
