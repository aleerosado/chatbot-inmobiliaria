import json
from app.model import generate_response  # <-- importamos tu modelo fine-tuneado

# Cargar propiedades desde JSON
def cargar_propiedades():
    with open("data/properties.json", "r") as f:
        return json.load(f)

# Manejo del mensaje del usuario
def handle_message(message):
    message = message.lower()
    propiedades = cargar_propiedades()

    # Intento de filtrar por distrito
    resultados = []
    for propiedad in propiedades:
        if propiedad['distrito'].lower() in message:
            resultados.append(propiedad)

    if resultados:
        info = "\n".join([
            f"ID {p['id']}: {p['descripcion']}, {p['habitaciones']} hab, {p['metros']} m2, ${p['precio']}"
            for p in resultados
        ])
        prompt = f"Soy un asesor inmobiliario. El usuario busca propiedades en '{message}'. Estas son las opciones encontradas:\n{info}\n¿Cómo se las presentarías como asesor profesional?"
        return generate_response(prompt)

    # Si no se detecta distrito, usar solo el modelo directamente
    prompt = f"Eres un asesor inmobiliario profesional. Responde a este mensaje del cliente:\n'{message}'"
    return generate_response(prompt)
