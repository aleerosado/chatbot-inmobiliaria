#para que el boy guarde los datos del usuario entre interacciones
import json

def load_memory():
    try:
        with open("data/memory.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_memory(data):
    with open("data/memory.json", "w") as f:
        json.dump(data, f)
