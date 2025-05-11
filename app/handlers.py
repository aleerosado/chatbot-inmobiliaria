#logica de la rpta del bot, se agregan condiciones teniendo en cuenta las intenciones
import json

#cargando propiedades
def cargar_propiedades():
    with open("data/properties.json", "r") as f:
        return json.load(f)


def handle_message(message):
    message = message.lower()
    propiedades = cargar_propiedades()

    if "hola" in message.lower():
        return "¡Hola! ¿Qué tipo de propiedad estás buscando? 🏡"
    return "Soy un bot inmobiliario 🤖, ¿puedo ayudarte a encontrar un departamento?"

    #filtrando por distrito
    resultados = []
    for propiedad in propiedades:
        if propiedad['distrito'].lower() in message:
            resultados.append(propiedad)
    
    if resultados:
        respuestas = []
        for prop in resultados:
            r = f"ID {prop['id']}: {prop['descripcion']}, {prop['habitaciones']} hab, {prop['metros']} m2, ${prop['precio']}"
            respuestas.append(r)
        return "Estas son algunas opciones:\n\n" + "\n".join(respuestas)

    return "No encontré propiedades con esa descripción. ¿Podrías especificar el distrito?"
