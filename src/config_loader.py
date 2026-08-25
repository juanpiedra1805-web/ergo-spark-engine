import json
import os

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))

def cargar_configuracion_metodo(nombre_metodo: str = "ROSA") -> dict:
    metodo = nombre_metodo.upper().strip()
    archivo_map = {
        "ROSA": os.path.join(CONFIG_DIR, "metodo_rosa.json"),
        "REBA": os.path.join(CONFIG_DIR, "metodo_reba.json"),
        "RULA": os.path.join(CONFIG_DIR, "metodo_rula.json")
    }
    
    ruta_archivo = archivo_map.get(metodo, archivo_map["ROSA"])
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo de configuración en: {ruta_archivo}")
        
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config

if __name__ == "__main__":
    for m in ["ROSA", "REBA", "RULA"]:
        cfg = cargar_configuracion_metodo(m)
        print(f"✅ Método [{m}] cargado correctamente: {cfg['ambito_aplicacion']}")
