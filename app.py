"""Microservicio de IA en Python, pensado para entenderse rápido.

Qué hace:
  - /clasificar  -> Machine Learning local: dice si un mensaje de cliente es
                    de ventas, soporte o queja.
  - /chat        -> LLM por API (Claude): responde preguntas. Necesita la
                    variable de entorno ANTHROPIC_API_KEY.

Cómo correrlo:  uvicorn app:app --port 8000
"""
import csv
import json
import os
import urllib.request
import threading
import time
from datetime import datetime
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1) Datos de ejemplo: el modelo aprende de estas frases.
EJEMPLOS = {
    "ventas": [
        "quiero comprar el producto", "cuánto cuesta el plan anual",
        "me interesa contratar el servicio", "tienen descuento por volumen",
        "quiero una cotización", "dónde puedo pagar mi pedido",
        "cuáles son los precios", "quiero suscribirme al plan premium",
    ],
    "soporte": [
        "no puedo iniciar sesión", "la página no carga", "olvidé mi contraseña",
        "me sale un error al subir el archivo", "cómo cambio mi correo",
        "la aplicación se cierra sola", "no me llega el código de verificación",
        "necesito ayuda para configurar mi cuenta",
    ],
    "queja": [
        "el servicio es pésimo", "llevo una semana esperando respuesta",
        "me cobraron de más y nadie me ayuda", "estoy muy molesto con la atención",
        "quiero cancelar, es una pérdida de tiempo", "nunca contestan mis mensajes",
        "me parece una falta de respeto", "exijo que me devuelvan mi dinero",
    ],
}
textos = [t for frases in EJEMPLOS.values() for t in frases]
etiquetas = [c for c, frases in EJEMPLOS.items() for _ in frases]

# 2) Entrenamiento (tarda milisegundos): convierte texto en números y ajusta pesos.
modelo = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                       LogisticRegression(max_iter=1000))
modelo.fit(textos, etiquetas)

# 3) La API web.
app = FastAPI(title="Servicio de IA")

# 4) Registro de uso: guarda fecha y hora de cada petición en peticiones.csv.
#    Con ese historial, prediccion.py calcula cuántas peticiones habrá a cierta hora.
ARCHIVO_LOG = os.getenv("LOG_PETICIONES", "peticiones.csv")
_candado = threading.Lock()

# 5) Alerta: avisa una vez cuando las peticiones de la hora actual llegan al límite.
#    Se configura con variables de entorno (valores por defecto entre paréntesis):
#    LIMITE_PETICIONES (30), ALERTAS_ARCHIVO (alertas.log), WEBHOOK_URL (vacío = sin webhook)
LIMITE = int(os.getenv("LIMITE_PETICIONES", "30"))
ARCHIVO_ALERTAS = os.getenv("ALERTAS_ARCHIVO", "alertas.log")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
_hora_actual, _conteo = "", 0

def _enviar_webhook(mensaje):
    try:
        cuerpo = json.dumps({"text": mensaje, "content": mensaje}).encode()  # Slack usa "text", Discord "content"
        peticion = urllib.request.Request(WEBHOOK_URL, cuerpo, {"Content-Type": "application/json"})
        urllib.request.urlopen(peticion, timeout=5)
    except Exception as e:
        print("No se pudo enviar el webhook:", e, flush=True)

def contar_y_avisar():
    global _hora_actual, _conteo
    ahora = datetime.now()
    hora = ahora.strftime("%Y-%m-%d %H")
    with _candado:
        if hora != _hora_actual:          # empezó una hora nueva: el contador vuelve a 0
            _hora_actual, _conteo = hora, 0
        _conteo += 1
        llego = _conteo == LIMITE
    if llego:
        mensaje = f"ALERTA: se alcanzaron {LIMITE} peticiones en la hora {ahora:%H}:00 ({ahora:%Y-%m-%d %H:%M:%S})"
        print(mensaje, flush=True)
        with open(ARCHIVO_ALERTAS, "a", encoding="utf-8") as f:
            f.write(mensaje + "\n")
        if WEBHOOK_URL:
            threading.Thread(target=_enviar_webhook, args=(mensaje,), daemon=True).start()

@app.get("/estado")
def estado():
    esta_hora = _hora_actual == datetime.now().strftime("%Y-%m-%d %H")
    return {"peticiones_esta_hora": _conteo if esta_hora else 0, "limite": LIMITE}

@app.middleware("http")
async def registrar(request: Request, call_next):
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    if request.url.path in ("/clasificar", "/chat"):
        ms = round((time.perf_counter() - inicio) * 1000, 1)
        with _candado, open(ARCHIVO_LOG, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"),
                                    request.url.path, respuesta.status_code, ms])
        contar_y_avisar()
    return respuesta

class Mensaje(BaseModel):
    texto: str

@app.get("/salud")
def salud():
    return {"estado": "ok"}

@app.post("/clasificar")
def clasificar(m: Mensaje):
    probs = modelo.predict_proba([m.texto])[0]
    i = probs.argmax()
    return {"categoria": str(modelo.classes_[i]), "confianza": round(float(probs[i]), 2)}

@app.post("/chat")
def chat(m: Mensaje):
    if not os.getenv("ANTHROPIC_API_KEY"):
        return {"error": "Falta ANTHROPIC_API_KEY en el servidor"}
    import anthropic  # se importa aquí para no gastar memoria si no se usa
    r = anthropic.Anthropic().messages.create(
        model="claude-sonnet-5-5", max_tokens=500,
        system="Eres el asistente de esta plataforma. Responde en español, breve.",
        messages=[{"role": "user", "content": m.texto}],
    )
    return {"respuesta": r.content[0].text}
