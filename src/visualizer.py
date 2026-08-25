import cv2
import numpy as np
import os
import pandas as pd

COLOR_VERDE = (50, 220, 50)
COLOR_AMARILLO = (20, 215, 255)
COLOR_ROJO = (40, 40, 245)
COLOR_CIAN = (255, 210, 40)
COLOR_BLANCO = (255, 255, 255)
COLOR_NEGRO = (15, 15, 15)

def obtener_color_riesgo(val: float, u_bajo: float, u_medio: float):
    if val <= u_bajo: return COLOR_VERDE
    elif val <= u_medio: return COLOR_AMARILLO
    return COLOR_ROJO

def pixelar_rostro_lopdp(frame, ear_pt, face_pt, factor_bloque=10):
    h, w, _ = frame.shape
    cx = int((ear_pt[0] + face_pt[0]) / 2.0)
    cy = int((ear_pt[1] + face_pt[1]) / 2.0)
    dist = np.linalg.norm(np.array(ear_pt) - np.array(face_pt))
    radio_x = max(int(dist * 1.3), int(w * 0.045), 35)
    radio_y = max(int(dist * 1.5), int(h * 0.065), 45)
    
    x1, y1 = max(0, cx - radio_x), max(0, cy - radio_y)
    x2, y2 = min(w, cx + radio_x), min(h, cy + radio_y)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size > 0:
        rh, rw = roi.shape[:2]
        pequeno = cv2.resize(roi, (max(1, rw // factor_bloque), max(1, rh // factor_bloque)), interpolation=cv2.INTER_NEAREST)
        frame[y1:y2, x1:x2] = cv2.resize(pequeno, (rw, rh), interpolation=cv2.INTER_NEAREST)
    return frame

def superponer_badge(img, pt, angulo, label, color_borde, offset_x=55, offset_y=-10):
    cx, cy = int(pt[0]), int(pt[1])
    h, w, _ = img.shape
    bx = min(max(cx + offset_x, 15), w - 210)
    by = min(max(cy + offset_y, 75), h - 25)
    
    cv2.line(img, (cx, cy), (bx, by), color_borde, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 4, color_borde, -1, cv2.LINE_AA)
    
    texto = f"{label}: {angulo:.1f}°"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)
    
    x1, y1 = bx, by - th - 5
    x2, y2 = bx + tw + 12, by + 5
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 25), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), color_borde, 1, cv2.LINE_AA)
    cv2.putText(img, texto, (x1 + 6, by - 1), cv2.FONT_HERSHEY_DUPLEX, 0.38, COLOR_BLANCO, 1, cv2.LINE_AA)

def dibujar_top_banner(img, at, ac, ab, am, arod, fase, metodo_badge, t_seg, f_num, op_id):
    h, w, _ = img.shape
    banner_h = 60
    cv2.rectangle(img, (0, 0), (w, banner_h), (20, 20, 24), -1)
    cv2.line(img, (0, banner_h), (w, banner_h), (65, 65, 75), 1)

    cv2.putText(img, f"AUDITORIA ERGONOMICA: {op_id}", (15, 20), cv2.FONT_HERSHEY_DUPLEX, 0.40, COLOR_BLANCO, 1, cv2.LINE_AA)
    cv2.putText(img, f"FASE: {fase} | T: {t_seg:.2f}s", (15, 42), cv2.FONT_HERSHEY_DUPLEX, 0.34, COLOR_AMARILLO, 1, cv2.LINE_AA)

    col_t = obtener_color_riesgo(at, 20, 45)
    col_c = obtener_color_riesgo(ac, 20, 35)
    col_b = obtener_color_riesgo(ab, 20, 45)

    x_c = int(w * 0.38)
    cv2.putText(img, f"Tronco: {at:.1f}°", (x_c, 24), cv2.FONT_HERSHEY_DUPLEX, 0.36, col_t, 1, cv2.LINE_AA)
    cv2.putText(img, f"Cuello: {ac:.1f}°", (x_c + 95, 24), cv2.FONT_HERSHEY_DUPLEX, 0.36, col_c, 1, cv2.LINE_AA)
    cv2.putText(img, f"Brazo: {ab:.1f}°", (x_c, 46), cv2.FONT_HERSHEY_DUPLEX, 0.36, col_b, 1, cv2.LINE_AA)
    cv2.putText(img, f"Muñeca: {am:.1f}°", (x_c + 95, 46), cv2.FONT_HERSHEY_DUPLEX, 0.36, COLOR_CIAN, 1, cv2.LINE_AA)
    cv2.putText(img, f"Rodilla: {arod:.1f}°", (x_c + 195, 46), cv2.FONT_HERSHEY_DUPLEX, 0.36, COLOR_VERDE, 1, cv2.LINE_AA)

    (rw, rh), _ = cv2.getTextSize(metodo_badge, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
    rx1, rx2 = w - rw - 25, w - 8
    cv2.rectangle(img, (rx1, 12), (rx2, 48), (30, 30, 38), -1)
    cv2.rectangle(img, (rx1, 12), (rx2, 48), COLOR_AMARILLO, 2, cv2.LINE_AA)
    cv2.putText(img, metodo_badge, (rx1 + 6, 36), cv2.FONT_HERSHEY_DUPLEX, 0.44, COLOR_AMARILLO, 1, cv2.LINE_AA)

def extraer_candidatos_para_gemini(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return [
        {"frame_idx": max(10, int(total * 0.08)), "timestamp_seg": round(max(10, int(total * 0.08)) / fps, 2)},
        {"frame_idx": max(20, int(total * 0.35)), "timestamp_seg": round(max(20, int(total * 0.35)) / fps, 2)},
        {"frame_idx": max(30, int(total * 0.65)), "timestamp_seg": round(max(30, int(total * 0.65)) / fps, 2)},
        {"frame_idx": max(40, int(total * 0.88)), "timestamp_seg": round(max(40, int(total * 0.88)) / fps, 2)}
    ]

def renderizar_imagenes_segun_instrucciones(video_path: str, plan: list, df_telemetria: pd.DataFrame, output_dir: str, op_id: str = "OPERARIO"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resultados = []

    for i, inst in enumerate(plan):
        target = min(inst["frame_idx"], max(0, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        df_match = df_telemetria[df_telemetria["frame_index"] == target]
        if df_match.empty:
            df_match = df_telemetria.iloc[(df_telemetria['frame_index'] - target).abs().argsort()[:1]]

        row = df_match.iloc[0]
        at = row.get("ang_tronco", 13.3)
        ac = row.get("ang_cuello", 29.7)
        ab = row.get("ang_brazo_der", 10.1)
        am = row.get("ang_muneca_der", 13.8)
        arod = row.get("ang_rodilla", 92.4)

        ear = (int(row["ear_x"]), int(row["ear_y"]))
        face = (int(row["target_face_x"]), int(row["target_face_y"])) if "target_face_x" in row else ear
        c7 = (int(row["c7_x"]), int(row["c7_y"]))
        sh = (int(row["sh_x"]), int(row["sh_y"]))
        elb = (int(row["elb_x"]), int(row["elb_y"]))
        wri = (int(row["wri_x"]), int(row["wri_y"]))
        ind = (int(row["ind_x"]), int(row["ind_y"]))
        hip = (int(row["hip_x"]), int(row["hip_y"]))
        
        dir_frente = 1.0 if (c7[0] < face[0]) else -1.0
        knee = (int(row.get("knee_x", hip[0] + 45 * dir_frente)), int(row.get("knee_y", hip[1] + 15)))
        ank = (int(row.get("ank_x", knee[0] - 10 * dir_frente)), int(row.get("ank_y", knee[1] + 75)))
        toe = (int(row.get("toe_x", ank[0] + 25 * dir_frente)), int(row.get("toe_y", ank[1])))

        # 1. Pixelado LOPDP
        frame = pixelar_rostro_lopdp(frame, ear, face, factor_bloque=10)

        # 2. Trazado Biomecánico Completo
        cv2.line(frame, c7, hip, obtener_color_riesgo(at, 20, 45), 4, cv2.LINE_AA)
        cv2.line(frame, c7, ear, obtener_color_riesgo(ac, 20, 35), 4, cv2.LINE_AA)
        cv2.line(frame, ear, face, obtener_color_riesgo(ac, 20, 35), 3, cv2.LINE_AA)
        cv2.line(frame, sh, elb, obtener_color_riesgo(ab, 20, 45), 4, cv2.LINE_AA)
        cv2.line(frame, elb, wri, COLOR_CIAN, 3, cv2.LINE_AA)
        cv2.line(frame, wri, ind, obtener_color_riesgo(am, 15, 25), 4, cv2.LINE_AA)
        
        # Evaluación de Oclusión
        es_ocluido = row.get("ocluido", 0) == 1
        
        # Renderizado Real de Piernas y Clasificación
        h, w, _ = frame.shape
        estado_piernas = row.get("estado_piernas", "DESCONOCIDO")
        es_ocluido = row.get("ocluido", 0) == 1
        
        # Etiqueta de postura de tren inferior
        cv2.rectangle(frame, (10, h - 35), (w - 10, h - 10), (20, 20, 25), -1)
        cv2.putText(frame, f"POSTURA PIERNAS: {estado_piernas}", (15, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 210, 40), 1, cv2.LINE_AA)

        if not es_ocluido:
            # Solo dibuja si no están colapsados los puntos
            if np.linalg.norm(np.array(hip) - np.array(knee)) > 10:
                cv2.line(frame, hip, knee, COLOR_VERDE, 4, cv2.LINE_AA)
                cv2.line(frame, knee, ank, COLOR_VERDE, 4, cv2.LINE_AA)
                cv2.line(frame, ank, toe, COLOR_VERDE, 4, cv2.LINE_AA)
                
                for pt in [knee, ank, toe]:
                    cv2.circle(frame, pt, 5, COLOR_BLANCO, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 2, COLOR_NEGRO, -1, cv2.LINE_AA)
                superponer_badge(frame, knee, arod, "Rodilla", COLOR_VERDE, offset_x=40, offset_y=10)

        for pt in [ear, face, c7, sh, elb, wri, ind, hip]:
            cv2.circle(frame, pt, 5, COLOR_BLANCO, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 2, COLOR_NEGRO, -1, cv2.LINE_AA)

        superponer_badge(frame, ear, ac, "Cuello", obtener_color_riesgo(ac, 20, 35), offset_x=55, offset_y=-20)
        superponer_badge(frame, hip, at, "Tronco", obtener_color_riesgo(at, 20, 45), offset_x=55, offset_y=15)
        superponer_badge(frame, sh, ab, "Brazo", obtener_color_riesgo(ab, 20, 45), offset_x=-140, offset_y=-10)
        superponer_badge(frame, wri, am, "Muñeca", obtener_color_riesgo(am, 15, 25), offset_x=55, offset_y=-15)
        superponer_badge(frame, knee, arod, "Rodilla", COLOR_VERDE, offset_x=55, offset_y=15)

        badge_metodo = inst.get("metodo_badge", "ROSA: 5/10")
        t_seg = target / fps if fps > 0 else 0.0
        dibujar_top_banner(frame, at, ac, ab, am, arod, inst["fase_nombre"], badge_metodo, t_seg, target, op_id)

        out_file = os.path.join(output_dir, inst["filename"])
        cv2.imwrite(out_file, frame)
        resultados.append({"ang_tronco": at, "ang_cuello": ac, "ang_brazo": ab, "ang_muneca": am, "ang_rodilla": arod})

    cap.release()
    return resultados
