import os
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar variable de entorno desde .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("[ERROR] No se encontró GEMINI_API_KEY en el archivo .env")
    exit(1)

genai.configure(api_key=api_key)

# Modelo recomendado por la API
MODELO = "gemini-3.6-flash"

print(f"[INFO] Probando generación con el modelo: {MODELO}...")

try:
    model = genai.GenerativeModel(MODELO)
    response = model.generate_content("Responde en una sola frase: 'Motor de IA Ergonómica activo y listo.'")
    
    print("\n✅ ¡Conexión exitosa con Google AI Studio!")
    print(f"Respuesta del modelo: {response.text.strip()}")

except Exception as e:
    print(f"\n❌ Error al conectar: {e}")
