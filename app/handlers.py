import json
from app.model import generate_response

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

        # Prompt más limpio, sin inyectar texto en respuesta
        prompt = f"""<|system|>
Eres un asesor inmobiliario profesional.
<|user|>
Estoy buscando propiedades en {message}.
<|assistant|>
"""
        respuesta_modelo = generate_response(prompt)
        return f"{info}\n\n{respuesta_modelo}"

    # Si no hay propiedades encontradas, usa el modelo directamente
    prompt = f"""<|system|>
Eres un asesor inmobiliario profesional.
<|user|>
{message}
<|assistant|>
"""
    return generate_response(prompt)
