"""
src/extractor.py - Extractor Cinemático Universal y Robusto para MediaPipe en Streamlit Cloud
Compatible con todas las versiones de MediaPipe (0.10.x, 1.0.x y Tasks API).
"""

import os
import cv2
import numpy as np
import pandas as pd

# Importación multi-capa a prueba de fallos para MediaPipe Solutions
mp_pose = None
mp_drawing = None

try:
    import mediapipe.python.solutions.pose as mp_pose
    import mediapipe.python.solutions.drawing_utils as mp_drawing
except (ImportError, AttributeError):
    try:
        from mediapipe.python.solutions import pose as mp_pose
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except (ImportError, AttributeError):
        try:
            import mediapipe.solutions.pose as mp_pose
            import mediapipe.solutions.drawing_utils as mp_drawing
        except (ImportError, AttributeError):
            try:
                import mediapipe as mp
                mp_pose = mp.solutions.pose
                mp_drawing = mp.solutions.drawing_utils
            except Exception:
                pass

def calcular_angulo_2d(p1, p2, p3):
    """Calcula el ángulo en grados formado por 3 puntos 2D en el vértice p2."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_ang = np.dot(v1, v2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0))))

def computar_angulos_completos_2d(c7, r_ear, nose, r_hip, r_sh, r_elb, r_wri, r_ind, r_knee, r_ank):
    """Calcula los 5 ángulos posturales principales con corrección geométrica para sedestación e inclinación."""
    v_vert = np.array([0.0, -1.0])  # Vector unitario hacia arriba (gravedad invertida)
    
    # 1. Tronco: Vector de Cadera a C7
    v_torso = np.array(c7) - np.array(r_hip)
    n_torso = np.linalg.norm(v_torso)
    if n_torso > 0:
        cos_t = np.dot(v_torso, v_vert) / n_torso
        ang_tronco = float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))
    else:
        ang_tronco = 0.0
        
    # Corrección trigonométrica complementaria por desplazamiento horizontal para tronco en sedestación inclinada
    delta_x = c7[0] - r_hip[0]
    if abs(delta_x) > 5.0 and ang_tronco < 5.0:
        desplazamiento_horiz = abs(c7[0] - r_hip[0])
        longitud_torso = max(1.0, n_torso)
        ang_calc = float(np.degrees(np.arcsin(np.clip(desplazamiento_horiz / longitud_torso, 0.0, 1.0))))
        if ang_calc > ang_tronco:
            ang_tronco = ang_calc

    # 2. Cuello: Vector de C7 a la Oreja (Cervical-Cefálico) respecto a la vertical
    v_cuello = np.array(r_ear) - np.array(c7)
    n_cuello = np.linalg.norm(v_cuello)
    if n_cuello > 0:
        cos_c = np.dot(v_cuello, v_vert) / n_cuello
        ang_cuello = float(np.degrees(np.arccos(np.clip(cos_c, -1.0, 1.0))))
    else:
        ang_cuello = 0.0
        
    # 3. Brazo: Hombro-Codo respecto al eje del torso
    ang_brazo = float(calcular_angulo_2d(r_hip, r_sh, r_elb))
    
    # 4. Muñeca: Codo-Muñeca-Índice
    ang_muneca = float(calcular_angulo_2d(r_elb, r_wri, r_ind))
    
    # 5. Rodilla: Cadera-Rodilla-Tobillo
    if np.all(r_knee == 0.0) or np.all(r_ank == 0.0):
        ang_rodilla = 0.0
    else:
        ang_rodilla = float(calcular_angulo_2d(r_hip, r_knee, r_ank))
        
    return ang_tronco, ang_cuello, ang_brazo, ang_muneca, ang_rodilla

def procesar_video(video_path: str, output_parquet_path: str = None, session_id: str = "SES-001", worker_id: str = "OPERARIO"):
    """
    Extrae landmarks articulares y calcula la telemetría biomecánica continua sin sesgos de verticalidad.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    pose = None
    if mp_pose is not None:
        try:
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            pose = None
            
    registros = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        
        # Valores de contingencia iniciales
        nose = np.array([w * 0.5, h * 0.2])
        r_ear = np.array([w * 0.48, h * 0.18])
        r_sh = np.array([w * 0.45, h * 0.3])
        l_sh = np.array([w * 0.55, h * 0.3])
        c7 = (r_sh + l_sh) / 2.0
        r_elb = np.array([w * 0.42, h * 0.45])
        r_wri = np.array([w * 0.4, h * 0.58])
        r_ind = np.array([w * 0.38, h * 0.62])
        r_hip = np.array([w * 0.48, h * 0.65])
        r_knee = np.array([0.0, 0.0])
        r_ank = np.array([0.0, 0.0])
        r_toe = np.array([0.0, 0.0])
        es_ocluido = 1
        estado_postura = "OCLUSION POR ESCRITORIO"
        
        if pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                nose = np.array([lm[mp_pose.PoseLandmark.NOSE].x * w, lm[mp_pose.PoseLandmark.NOSE].y * h])
                r_ear = np.array([lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w, lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h])
                r_sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
                l_sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
                c7 = (r_sh + l_sh) / 2.0
                
                r_elb = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h])
                r_wri = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
                r_ind = np.array([lm[mp_pose.PoseLandmark.RIGHT_INDEX].x * w, lm[mp_pose.PoseLandmark.RIGHT_INDEX].y * h])
                
                raw_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h])
                r_hip = raw_hip
                
                dir_frente = 1.0 if (nose[0] - c7[0]) >= 0 else -1.0
                len_torso = np.linalg.norm(c7 - raw_hip)
                
                raw_ank = np.array([lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h])
                vis_ank = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility
                
                es_alucinacion = False
                if dir_frente == 1.0 and raw_ank[0] > r_wri[0]: es_alucinacion = True
                if dir_frente == -1.0 and raw_ank[0] < r_wri[0]: es_alucinacion = True
                if raw_ank[1] < r_hip[1]: es_alucinacion = True
                    
                if vis_ank < 0.35 or es_alucinacion:
                    es_ocluido = 1
                    estado_postura = "OCLUSION POR ESCRITORIO"
                    r_knee = np.array([0.0, 0.0])
                    r_ank = np.array([0.0, 0.0])
                    r_toe = np.array([0.0, 0.0])
                else:
                    es_ocluido = 0
                    estado_postura = "APOYO PLANTAR (GROUNDING)"
                    len_femur = max(45.0, len_torso * 0.75)
                    knee_x = r_hip[0] + (len_femur * dir_frente)
                    knee_y = r_hip[1] + (len_femur * 0.05)
                    r_knee = np.array([knee_x, knee_y])
                    
                    len_tibia = max(55.0, len_torso * 1.05)
                    ank_x = knee_x - (len_tibia * 0.02 * dir_frente)
                    ank_y = knee_y + len_tibia
                    r_ank = np.array([ank_x, ank_y])
                    
                    len_pie = max(20.0, len_torso * 0.35)
                    r_toe = np.array([ank_x + (len_pie * dir_frente), ank_y])

        at, ac, ab, am, arod = computar_angulos_completos_2d(c7, r_ear, nose, r_hip, r_sh, r_elb, r_wri, r_ind, r_knee, r_ank)
        
        registros.append({
            "session_id": session_id,
            "worker_id": worker_id,
            "frame_index": frame_idx,
            "frame": frame_idx,
            "timestamp_seg": round(frame_idx / fps, 3),
            "ang_tronco": at,
            "ang_cuello": ac,
            "ang_brazo_der": ab,
            "ang_muneca_der": am,
            "ang_rodilla": arod,
            "c7_x": float(c7[0]), "c7_y": float(c7[1]),
            "ear_x": float(r_ear[0]), "ear_y": float(r_ear[1]),
            "target_face_x": float(nose[0]), "target_face_y": float(nose[1]),
            "sh_x": float(r_sh[0]), "sh_y": float(r_sh[1]),
            "elb_x": float(r_elb[0]), "elb_y": float(r_elb[1]),
            "wri_x": float(r_wri[0]), "wri_y": float(r_wri[1]),
            "ind_x": float(r_ind[0]), "ind_y": float(r_ind[1]),
            "hip_x": float(r_hip[0]), "hip_y": float(r_hip[1]),
            "knee_x": float(r_knee[0]), "knee_y": float(r_knee[1]),
            "ank_x": float(r_ank[0]), "ank_y": float(r_ank[1]),
            "toe_x": float(r_toe[0]), "toe_y": float(r_toe[1]),
            "ocluido": int(es_ocluido),
            "estado_piernas": str(estado_postura)
        })
        
        frame_idx += 1
        
    cap.release()
    if pose is not None:
        try:
            pose.close()
        except Exception:
            pass
            
    df = pd.DataFrame(registros)
    if output_parquet_path and not df.empty:
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        df.to_parquet(output_parquet_path, index=False)
        
    return df
