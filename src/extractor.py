import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
# ❌ Línea original:
# mp_pose = mp.solutions.pose

# ✅ Importación resiliente:
try:
    import mediapipe as mp
    try:
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
    except AttributeError:
        from mediapipe.python.solutions import pose as mp_pose
        from mediapipe.python.solutions import drawing_utils as mp_drawing
except Exception:
    import mediapipe as mp
    mp_pose = getattr(mp, "solutions", None)
def procesar_video(video_path: str, output_parquet_path: str = None, session_id: str = "SES-001", worker_id: str = "OPERARIO"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    registros = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            # --- TREN SUPERIOR ---
            nose = np.array([lm[mp_pose.PoseLandmark.NOSE].x * w, lm[mp_pose.PoseLandmark.NOSE].y * h])
            r_ear = np.array([lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w, lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h])
            r_sh = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h])
            l_sh = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h])
            c7 = (r_sh + l_sh) / 2.0
            
            r_elb = np.array([lm[mp_pose.PoseLandmark.RIGHT_ELBOW].x * w, lm[mp_pose.PoseLandmark.RIGHT_ELBOW].y * h])
            r_wri = np.array([lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w, lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h])
            r_ind = np.array([lm[mp_pose.PoseLandmark.RIGHT_INDEX].x * w, lm[mp_pose.PoseLandmark.RIGHT_INDEX].y * h])
            raw_hip = np.array([lm[mp_pose.PoseLandmark.RIGHT_HIP].x * w, lm[mp_pose.PoseLandmark.RIGHT_HIP].y * h])
            
            # Vector absoluto Escápulo-Facial para orientación sagital
            dir_frente = 1.0 if (nose[0] - c7[0]) >= 0 else -1.0
            len_torso = np.linalg.norm(c7 - raw_hip)
            
            # Corrección de anclaje de cadera al plano del asiento
            hip_x = raw_hip[0] - (len_torso * 0.05 * dir_frente)
            hip_y = raw_hip[1] + (len_torso * 0.28)
            r_hip = np.array([hip_x, hip_y])
            
            # --- EVALUACIÓN DE OCLUSIÓN Y POSTURA (ANTI-ALUCINACIONES) ---
            raw_ank = np.array([lm[mp_pose.PoseLandmark.RIGHT_ANKLE].x * w, lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y * h])
            vis_ank = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility
            
            es_ocluido = 0
            estado_postura = "DESCONOCIDO"
            
            es_alucinacion = False
            if dir_frente == 1.0 and raw_ank[0] > r_wri[0]:
                es_alucinacion = True
            if dir_frente == -1.0 and raw_ank[0] < r_wri[0]:
                es_alucinacion = True
            if raw_ank[1] < r_hip[1]:
                es_alucinacion = True
            
            if vis_ank < 0.35 or es_alucinacion:
                es_ocluido = 1
                estado_postura = "OCLUSION POR ESCRITORIO"
            else:
                dist_x_ank = (raw_ank[0] - hip_x) * dir_frente
                dist_y_ank = raw_ank[1] - hip_y
                
                if dist_y_ank < len_torso * 0.4:
                    estado_postura = "PIES SOBRE LA SILLA"
                elif dist_x_ank < -(len_torso * 0.15):
                    estado_postura = "PIERNAS RECOGIDAS ATRAS"
                else:
                    estado_postura = "APOYO PLANTAR (GROUNDING)"
            
            # --- TRAZADO BIOMECÁNICO (INVERSE KINEMATICS) ---
            if es_ocluido == 1 or estado_postura != "APOYO PLANTAR (GROUNDING)":
                r_knee = np.array([0.0, 0.0])
                r_ank = np.array([0.0, 0.0])
                r_toe = np.array([0.0, 0.0])
                arod = 0.0
            else:
                len_femur = max(45.0, len_torso * 0.75)
                knee_x = hip_x + (len_femur * dir_frente)
                knee_y = hip_y + (len_femur * 0.05)
                r_knee = np.array([knee_x, knee_y])
                
                len_tibia = max(55.0, len_torso * 1.05)
                ank_x = knee_x - (len_tibia * 0.02 * dir_frente)
                ank_y = knee_y + len_tibia
                r_ank = np.array([ank_x, ank_y])
                
                len_pie = max(20.0, len_torso * 0.35)
                r_toe = np.array([ank_x + (len_pie * dir_frente), ank_y])
                
                _, _, _, _, arod = computar_angulos_completos_2d(c7, r_ear, nose, r_hip, r_sh, r_elb, r_wri, r_ind, r_knee, r_ank)

            at, ac, ab, am, _ = computar_angulos_completos_2d(c7, r_ear, nose, r_hip, r_sh, r_elb, r_wri, r_ind, r_hip, r_hip)
            
            registros.append({
                "session_id": session_id,
                "worker_id": worker_id,
                "frame_index": frame_idx,
                "timestamp_seg": round(frame_idx / fps, 3),
                "ang_tronco": at,
                "ang_cuello": ac,
                "ang_brazo_der": ab,
                "ang_muneca_der": am,
                "ang_rodilla": arod,
                "c7_x": c7[0], "c7_y": c7[1],
                "ear_x": r_ear[0], "ear_y": r_ear[1],
                "target_face_x": nose[0], "target_face_y": nose[1],
                "sh_x": r_sh[0], "sh_y": r_sh[1],
                "elb_x": r_elb[0], "elb_y": r_elb[1],
                "wri_x": r_wri[0], "wri_y": r_wri[1],
                "ind_x": r_ind[0], "ind_y": r_ind[1],
                "hip_x": r_hip[0], "hip_y": r_hip[1],
                "knee_x": r_knee[0], "knee_y": r_knee[1],
                "ank_x": r_ank[0], "ank_y": r_ank[1],
                "toe_x": r_toe[0], "toe_y": r_toe[1],
                "ocluido": es_ocluido,
                "estado_piernas": estado_postura
            })
            
        frame_idx += 1
        
    cap.release()
    pose.close()
    
    df = pd.DataFrame(registros)
    if output_parquet_path and not df.empty:
        os.makedirs(os.path.dirname(output_parquet_path), exist_ok=True)
        df.to_parquet(output_parquet_path, index=False)
        
    return df
