class NodoDecision:
    def __init__(self, pregunta, rama_si, rama_no):
        self.pregunta = pregunta
        self.rama_si = rama_si
        self.rama_no = rama_no

    def decidir(self):
        respuesta = input(self.pregunta + " (sí/no): ").strip().lower()
        if respuesta == 'sí':
            return self.rama_si.decidir()
        elif respuesta == 'no':
            return self.rama_no.decidir()
        else:
            print("Respuesta inválida. Por favor, responde con 'sí' o 'no'.")
            return self.decidir()

class NodoAccion:
    def __init__(self, recomendacion):
        self.recomendacion = recomendacion

    def decidir(self):
        print(self.recomendacion)
        return self.recomendacion

# Crear nodos de decisión
raiz = NodoDecision(
    "¿Necesitas una fruta rica en vitamina C?",
    rama_si=NodoDecision(
        "¿Te apetece algo dulce?",
        rama_si=NodoAccion("Te recomiendo comer una naranja."),
        rama_no=NodoAccion("Te sugiero comer un kiwi.")
    ),
    rama_no=NodoDecision(
        "¿Buscas una fruta con pocas calorías?",
        rama_si=NodoAccion("Entonces, una manzana sería ideal."),
        rama_no=NodoAccion("Quizás un plátano sea lo que buscas.")
    )
)

# Ejecutar la red de decisión
print("Bienvenido al sistema de recomendación de frutas.")
raiz.decidir()
