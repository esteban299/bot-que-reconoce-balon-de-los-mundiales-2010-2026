# ==========================================================
# DISCORD AI BOT
# Plantilla del ejercicio
#
# Objetivo:
# Conectar un bot de Discord con una red neuronal entrenada
# usando YOLO para identificar imágenes.
# ==========================================================

import discord
from discord.ext import commands
from ultralytics import YOLO
from dotenv import load_dotenv
import os

# ==========================================================
# CONFIGURACIÓN
# ==========================================================


# Cargar el modelo entrenado
model = YOLO("best.pt")

# ==========================================================
# INFORMACIÓN DE LAS CLASES
#
# Completa este diccionario con información sobre las clases
# que reconoce tu modelo.
# ==========================================================

info = {

    "mundial 2010": {
        "nombre": "Jabulani (Sudáfrica 2010)",
        "descripcion": "Balón oficial del Mundial de Sudáfrica 2010 fabricado por Adidas con 8 paneles sellados térmicamente.",
        "dato": "Fue famoso por sus trayectorias impredecibles en el aire debido a su diseño aerodinámico."
    },

    "mundial 2014": {
        "nombre": "Brazuca (Brasil 2014)",
        "descripcion": "Balón oficial del Mundial de Brasil 2014 compuesto por 6 paneles simétricos con textura de microesferas.",
        "dato": "Fue el primer balón de la Copa Mundial nombrado por los propios aficionados en Brasil."
    },

    "mundial 2018": {
        "nombre": "Telstar 18 (Rusia 2018)",
        "descripcion": "Balón oficial del Mundial de Rusia 2018 que rindió homenaje al clásico Telstar de 1970 con un diseño de píxeles.",
        "dato": "Incluyó por primera vez un chip NFC integrado para interactuar con teléfonos inteligentes."
    },

    "mundial 2022": {
        "nombre": "Al Rihla (Catar 2022)",
        "descripcion": "Balón oficial del Mundial de Catar 2022 diseñado para desplazarse más rápido en el aire que cualquier otro balón.",
        "dato": "Contaba con un sensor de medición inercial (IMU) en el centro para apoyar la tecnología del fuera de juego semiautomatizado."
    },

    "mundial 2026": {
        "nombre": "Balón Oficial (Norteamérica 2026)",
        "descripcion": "Balón oficial para el Mundial de Estados Unidos, México y Canadá 2026.",
        "dato": "Diseñado para adaptarse a las diferentes condiciones climáticas y altitudes de las tres sedes del torneo."
    }

}

# ==========================================================
# FUNCIÓN DE PREDICCIÓN
# ==========================================================

def predict(image_path):

    results = model(image_path)

    if results[0].probs is not None:
        clase_idx = results[0].probs.top1
        clase = model.names[clase_idx]
        confianza = float(results[0].probs.top1conf)
    else:
        clase_idx = int(results[0].boxes.cls[0])
        clase = model.names[clase_idx]
        confianza = float(results[0].boxes.conf[0])

    return clase, confianza

# ==========================================================
# CONFIGURAR DISCORD
# ==========================================================

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(
    command_prefix="!",
    intents=intents
)

# ==========================================================
# BOT LISTO
# ==========================================================

@bot.event
async def on_ready():

    print(f"{bot.user} está conectado.")

# ==========================================================
# COMANDO DE AYUDA
# ==========================================================

@bot.command()

async def start(ctx):

    mensaje = (
        "🤖 Hola.\n\n"
        "Soy un bot con Inteligencia Artificial.\n"
        "Escribe el comando correspondiente y adjunta una imagen."
    )

    await ctx.send(mensaje)

# ==========================================================
# COMANDO PRINCIPAL
#
# Cambia el nombre del comando según tu proyecto.
#
# Ejemplo:
#
# !animal
# !fruta
# !flor
# !perro
# !pokemon
# ==========================================================

@bot.command(name="mundial")

async def analizar(ctx):

    # -----------------------------------
    # 1. Verificar que exista una imagen
    # -----------------------------------

    if not ctx.message.attachments:

        await ctx.send("Adjunta una imagen.")
        return

    attachment = ctx.message.attachments[0]

    # -----------------------------------
    # 2. Guardar imagen temporal
    # -----------------------------------

    image_path = f"temp_{attachment.filename}"

    await attachment.save(image_path)

    try:

        # -----------------------------------
        # 3. Ejecutar la IA
        # -----------------------------------

        clase, confianza = predict(image_path)

        porcentaje = round(confianza * 100, 2)

        # -----------------------------------
        # 4. Buscar información
        # -----------------------------------

        datos = info.get(str(clase).lower(), {
            "nombre": clase,
            "descripcion": "Sin descripción disponible.",
            "dato": "Sin dato curioso disponible."
        })

        # -----------------------------------
        # 5. Construir respuesta
        # -----------------------------------

        respuesta = (
            f"Clase detectada: {datos['nombre']}\n\n"
            f"Certeza: {porcentaje}%\n\n"
            f"Descripción:\n"
            f"{datos['descripcion']}\n\n"
            f"Dato curioso:\n"
            f"{datos['dato']}"
        )

        await ctx.send(respuesta)

    except Exception as e:

        print(e)

        await ctx.send("Ocurrió un error.")

    finally:

        if os.path.exists(image_path):
            os.remove(image_path)

# ==========================================================
# INICIAR BOT
# ==========================================================

if __name__ == "__main__":

    bot.run('token')