import cv2
import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def clasificar_puesto_automaticamente(video_path: str) -> dict:
    """
    Analiza visualmente el entorno del puesto y clasifica el método ergonómico óptimo (ROSA / REBA / RULA).
    """
    if not os.path.exists(video_path):
        return {"metodo": "ROSA", "tipo_estacion": "PVD/Oficina", "justificacion_pericial": "Video no encontrado, fallback ROSA"}

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extraer fotograma al 20% del video
    target_frame = max(0, int(total_frames * 0.20))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return {"metodo": "ROSA", "tipo_estacion": "PVD/Oficina", "justificacion_pericial": "Fallo al leer frame, fallback ROSA"}

    temp_img_path = "reportes/img/temp_triage.jpg"
    os.makedirs("reportes/img", exist_ok=True)
    cv2.imwrite(temp_img_path, frame)

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        sample_file = genai.upload_file(path=temp_img_path, display_name="Triage_Frame")

        prompt = """
        Eres un Perito Auditor en Ergonomía Ocupacional.
        Analiza esta imagen y clasifica el puesto de trabajo en UNO de los tres métodos ergonómicos oficiales:

        1. "ROSA": Puesto administrativo, teletrabajo o de oficina con Pantalla de Visualización de Datos (PVD/Laptop/Monitor), teclado, mouse o escritorio.
        2. "REBA": Puesto industrial, línea de empaque/envasado, maquinaria pesada, manipulación manual de cargas o bipedestación con demanda de cuerpo completo.
        3. "RULA": Puesto de ensamble de precisión, inspección en banco de trabajo o manipulación estática donde la carga principal es en extremidades superiores.

        Responde ÚNICAMENTE con este JSON exacto:
        {
          "metodo": "ROSA",
          "tipo_estacion": "Descripción corta del puesto",
          "justificacion_pericial": "Explicación técnica en 2 líneas del porqué aplica este método"
        }
        """

        response = model.generate_content(
            [sample_file, prompt],
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
        )

        resultado = json.loads(response.text)
        print(f"\n🤖 [IA Triage Ergonómico]: Método Detectado -> {resultado['metodo']}")
        print(f"   Justificación: {resultado['justificacion_pericial']}\n")
        return resultado

    except Exception as e:
        print(f"[WARN] Fallback en clasificación visual ({e}). Usando ROSA por defecto.")
        return {"metodo": "ROSA", "tipo_estacion": "PVD/Oficina", "justificacion_pericial": "Asignado por fallback seguro"}
