#Verificando que el servidor se ejecute ok
from flask import Blueprint

routes = Blueprint('routes', __name__)

@routes.route("/")
def index():
    return "Chatbot Inmobiliario activo 🚀"