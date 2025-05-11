#logica de la rpta del bot, se agregan condiciones teniendo en cuenta las intenciones
def handle_message(message):
    if "hola" in message.lower():
        return "¡Hola! ¿Qué tipo de propiedad estás buscando? 🏡"
    return "Soy un bot inmobiliario 🤖, ¿puedo ayudarte a encontrar un departamento?"
