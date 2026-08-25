import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

print("[INFO] Generando lote de prueba sintético (1000 frames)...")
frames = 1000

data = {
    "session_id": ["AUDITORIA_PILOTO_01"] * frames,
    "worker_id": ["PUESTO_OPERATIVO_A"] * frames,
    "frame_index": list(range(frames)),
    "timestamp_ms": [int(i * (1000/30)) for i in range(frames)],
    "nose": [{"x": 0.0, "y": 0.6 + np.sin(i/20)*0.05, "z": 0.0, "score": 0.99} for i in range(frames)],
    "l_shoulder": [{"x": -0.2, "y": 0.4, "z": 0.0, "score": 0.99} for i in range(frames)],
    "r_shoulder": [{"x": 0.2, "y": 0.4, "z": 0.0, "score": 0.99} for i in range(frames)],
    "l_hip": [{"x": -0.15, "y": -0.1, "z": 0.0, "score": 0.99} for i in range(frames)],
    "r_hip": [{"x": 0.15, "y": -0.1, "z": 0.0, "score": 0.99} for i in range(frames)],
    "l_elbow": [{"x": -0.3, "y": 0.1, "z": 0.0, "score": 0.99} for i in range(frames)],
    "r_elbow": [{"x": 0.35, "y": 0.25 + np.sin(i/15)*0.1, "z": 0.1, "score": 0.99} for i in range(frames)],
    "l_wrist": [{"x": -0.3, "y": -0.1, "z": 0.0, "score": 0.99} for i in range(frames)],
    "r_wrist": [{"x": 0.4, "y": 0.1, "z": 0.1, "score": 0.99} for i in range(frames)],
    "r_knee": [{"x": 0.15, "y": -0.5, "z": 0.0, "score": 0.99} for i in range(frames)],
    "r_ankle": [{"x": 0.15, "y": -0.9, "z": 0.0, "score": 0.99} for i in range(frames)],
    "ctx_base_silla": [3] * frames,
    "ctx_base_perif": [2] * frames
}

os.makedirs("data/raw_landmarks", exist_ok=True)
tabla = pa.Table.from_pydict(data)
pq.write_table(tabla, "data/raw_landmarks/muestra_prueba.parquet")
print("✅ Archivo 'data/raw_landmarks/muestra_prueba.parquet' creado.")
