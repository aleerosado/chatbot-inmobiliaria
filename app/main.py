from flask import Flask, request
from transformers import AutoTokenizer, AutoModelForCausalLM
from twilio.twiml.messaging_response import MessagingResponse
from dotenv import load_dotenv
import os
import torch

#variables de entorno
load_dotenv()

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER")

#Iniciando flask
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def whatsapp_bot():
    incoming_msg = request.values.get("Body", "").lower()
    resp = MessagingResponse()
    msg = resp.message()

    # Respuesta simple temporal
    if "hola" in incoming_msg:
        msg.body("¡Hola! ¿Qué tipo de propiedad estás buscando? 🏡")
    else:
        msg.body("Soy un bot inmobiliario 🤖, ¿puedo ayudarte a encontrar una propiedad?")

    return str(resp)

if __name__ == "__main__":
    app.run(debug=True)