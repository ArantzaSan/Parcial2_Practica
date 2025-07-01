from temporal_logic import LTL

# Importamos la librería necesaria para trabajar con lógica temporal

# Definimos una clase para modelar un sistema de ciberseguridad
class CyberSecuritySystem:
    def __init__(self):
        # Inicializamos los estados del sistema
        self.events = []
        self.alert_triggered = False

    def log_event(self, event):
        # Registramos un evento en el sistema
        self.events.append(event)

    def check_security_rules(self):
        # Definimos reglas de lógica temporal para detectar anomalías
        # Ejemplo: Si ocurre un evento de acceso no autorizado, debe seguir una alerta
        rule = LTL("G (unauthorized_access -> F alert_triggered)")

        # Evaluamos la regla en el historial de eventos
        return rule.evaluate(self.events)

    def trigger_alert(self):
        # Activamos una alerta de seguridad
        self.alert_triggered = True
        self.log_event("alert_triggered")

# Simulación del sistema de ciberseguridad
if __name__ == "__main__":
    # Creamos una instancia del sistema
    system = CyberSecuritySystem()

    # Simulamos eventos en el sistema
    system.log_event("user_login")
    system.log_event("unauthorized_access")  # Evento sospechoso
    system.trigger_alert()  # Activamos la alerta

    # Verificamos las reglas de seguridad
    if system.check_security_rules():
        print("Reglas de seguridad cumplidas.")
    else:
        print("Violación de reglas de seguridad detectada.")
