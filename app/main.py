from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os

from app.handlers import handle_message
from app.memory import load_memory, save_memory
from app.model import generate_response

load_dotenv()

TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    try:
        incoming_msg = request.values.get("Body", "").strip()
        sender = request.values.get("From", "")
        print(f"📥 Mensaje recibido de {sender}: {incoming_msg}")

        memory = load_memory()
        session = memory.get(sender, {})

        # Procesar mensaje con manejo de errores
        respuesta = handle_message(incoming_msg)
        print(f"🤖 Respuesta generada: {respuesta}")

        # Guardar memoria
        memory[sender] = session
        save_memory(memory)

        # Enviar respuesta
        twilio_response = MessagingResponse()
        twilio_response.message(respuesta)
        return str(twilio_response)

    except Exception as e:
        print("❌ Error en webhook:", str(e))
        twilio_response = MessagingResponse()
        twilio_response.message("Ocurrió un error procesando tu mensaje. Intenta nuevamente.")
        return str(twilio_response)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
