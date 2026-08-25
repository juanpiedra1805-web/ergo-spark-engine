import numpy as np

def calcular_angulo_vectores_2d(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))

def computar_angulos_completos_2d(c7, ear, eye_or_nose, hip, sh, elb, wri, ind, knee=None, ankle=None):
    # Tronco respecto a la vertical
    vt = hip - c7
    at = abs(calcular_angulo_vectores_2d(vt, np.array([0, 1])))
    if at > 90.0: at = abs(180.0 - at)

    # Cuello / Cráneo-cervical (Frankfurt + C7)
    dx_head = abs(eye_or_nose[0] - ear[0])
    dy_head = eye_or_nose[1] - ear[1]
    pitch_cabeza = np.degrees(np.arctan2(dy_head, max(1e-5, dx_head)))
    pitch_cabeza = max(0.0, float(pitch_cabeza))

    head_cm = (ear + eye_or_nose) / 2.0
    vc = head_cm - c7
    ac_col = abs(calcular_angulo_vectores_2d(vc, np.array([0, -1])))
    if ac_col > 90.0: ac_col = abs(180.0 - ac_col)
    ac = max(pitch_cabeza, ac_col)

    # Brazo
    vb = elb - sh
    ab = abs(calcular_angulo_vectores_2d(vb, vt))
    if ab > 180.0: ab = 360.0 - ab

    # Muñeca
    va = wri - elb
    vm = ind - wri
    am_raw = abs(calcular_angulo_vectores_2d(va, vm))
    am = abs(180.0 - am_raw) if am_raw > 90.0 else am_raw

    # Miembros Inferiores: Ángulo Poplíteo de Rodilla (Muslo vs Pierna)
    if knee is not None and ankle is not None:
        v_muslo = knee - hip
        v_pierna = ankle - knee
        arod_raw = abs(calcular_angulo_vectores_2d(v_muslo, v_pierna))
        arod = round(arod_raw, 1)
    else:
        arod = 90.0

    return round(at, 1), round(ac, 1), round(ab, 1), round(am, 1), round(arod, 1)

# ROSA
def calcular_matriz_rosa_oficial(tronco_p50, cuello_p50, brazo_p50, muneca_p50, rodilla_p50=90.0):
    score_silla = 2
    if tronco_p50 > 20.0: score_silla += 1
    if brazo_p50 > 20.0: score_silla += 1
    # Penalización si no hay ángulo de 90° en rodillas (compresión poplítea)
    if abs(rodilla_p50 - 90.0) > 15.0: score_silla += 1

    score_pantalla = 2
    if cuello_p50 > 25.0: score_pantalla += 2
    elif cuello_p50 > 15.0: score_pantalla += 1
    
    score_perif = 2
    if muneca_p50 > 15.0: score_perif += 1
    if brazo_p50 > 45.0: score_perif += 1

    score_b = min(9, score_pantalla + score_perif - 1)
    
    tabla_c = {
        (2, 2): 2, (2, 3): 2, (2, 4): 3, (2, 5): 4, (2, 6): 5, (2, 7): 6,
        (3, 2): 2, (3, 3): 3, (3, 4): 4, (3, 5): 5, (3, 6): 6, (3, 7): 7,
        (4, 2): 3, (4, 3): 4, (4, 4): 5, (4, 5): 6, (4, 6): 7, (4, 7): 8,
        (5, 2): 4, (5, 3): 5, (5, 4): 6, (5, 5): 7, (5, 6): 8, (5, 7): 9,
        (6, 2): 5, (6, 3): 6, (6, 4): 7, (6, 5): 8, (6, 6): 9, (6, 7): 10,
    }
    s_idx = min(6, max(2, score_silla))
    b_idx = min(7, max(2, score_b))
    return int(tabla_c.get((s_idx, b_idx), 5))

# RULA
def calcular_matriz_rula_oficial(tronco_p50, cuello_p50, brazo_p50, muneca_p50, rodilla_p50=90.0):
    s_brazo = 4 if brazo_p50 > 90 else (3 if brazo_p50 > 45 else (2 if brazo_p50 > 20 else 1))
    s_antebrazo = 2 if (brazo_p50 > 30 or muneca_p50 > 15) else 1
    s_muneca = 3 if muneca_p50 > 15 else (2 if muneca_p50 > 5 else 1)
    
    tabla_a = [
        [[1, 2], [2, 2], [2, 3], [3, 3]],
        [[2, 2], [2, 3], [3, 3], [3, 4]],
        [[2, 3], [3, 3], [3, 4], [4, 4]],
        [[3, 4], [4, 4], [4, 5], [5, 5]],
        [[4, 4], [4, 5], [5, 5], [6, 6]],
        [[5, 5], [5, 6], [6, 7], [7, 7]]
    ]
    idx_br = min(5, max(0, s_brazo - 1))
    idx_ant = min(1, max(0, s_antebrazo - 1))
    idx_mun = min(3, max(0, s_muneca - 1))
    score_a = tabla_a[idx_br][idx_mun][idx_ant] + 1

    s_cuello = 4 if cuello_p50 > 35 else (3 if cuello_p50 > 20 else (2 if cuello_p50 > 10 else 1))
    s_tronco = 4 if tronco_p50 > 60 else (3 if tronco_p50 > 20 else (2 if tronco_p50 > 5 else 1))
    s_piernas = 2 if abs(rodilla_p50 - 90.0) > 20.0 else 1

    tabla_b = [
        [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
        [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
        [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
        [[5, 5], [5, 6], [6, 7], [7, 7], [7, 8], [8, 8]],
        [[7, 7], [7, 7], [7, 8], [8, 8], [8, 9], [9, 9]],
        [[8, 8], [8, 8], [8, 9], [9, 9], [9, 9], [9, 9]]
    ]
    idx_cue = min(5, max(0, s_cuello - 1))
    idx_tro = min(5, max(0, s_tronco - 1))
    idx_pie = min(1, max(0, s_piernas - 1))
    score_b = tabla_b[idx_cue][idx_tro][idx_pie] + 1

    tabla_c = [
        [1, 2, 3, 3, 4, 5, 5],
        [2, 2, 3, 4, 4, 5, 5],
        [3, 3, 3, 4, 4, 5, 6],
        [3, 3, 3, 4, 5, 6, 6],
        [4, 4, 4, 5, 6, 7, 7],
        [4, 4, 5, 6, 6, 7, 7],
        [5, 5, 6, 6, 7, 7, 7],
        [5, 5, 6, 7, 7, 7, 7]
    ]
    return int(tabla_c[min(7, max(0, score_a - 1))][min(6, max(0, score_b - 1))])

# REBA
def calcular_matriz_reba_oficial(tronco_p50, cuello_p50, brazo_p50, muneca_p50, rodilla_p50=90.0):
    s_tronco = 4 if tronco_p50 > 60 else (3 if tronco_p50 > 20 else (2 if tronco_p50 > 0 else 1))
    s_cuello = 3 if cuello_p50 > 35 else (2 if cuello_p50 > 20 else 1)
    
    # Evaluación explícita de piernas en REBA
    s_piernas = 1
    if abs(rodilla_p50 - 90.0) > 30.0: s_piernas = 2
    if rodilla_p50 < 60.0: s_piernas += 1 # Flexión severa / cuclillas

    tabla_a = [
        [[1, 2], [2, 3], [3, 4]],
        [[2, 3], [3, 4], [4, 5]],
        [[3, 4], [4, 5], [5, 6]],
        [[4, 5], [5, 6], [6, 7]],
        [[6, 7], [7, 8], [8, 9]]
    ]
    t_idx = min(4, max(0, s_tronco - 1))
    c_idx = min(2, max(0, s_cuello - 1))
    p_idx = min(1, max(0, s_piernas - 1))
    score_a = tabla_a[t_idx][c_idx][p_idx]

    s_brazo = 4 if brazo_p50 > 90 else (3 if brazo_p50 > 45 else (2 if brazo_p50 > 20 else 1))
    s_antebrazo = 2 if (brazo_p50 > 30 or muneca_p50 > 15) else 1
    s_muneca = 2 if muneca_p50 > 15 else 1

    tabla_b = [
        [[1, 2], [1, 2]],
        [[1, 2], [2, 3]],
        [[3, 4], [4, 5]],
        [[4, 5], [5, 6]],
        [[6, 7], [7, 8]],
        [[7, 8], [8, 9]]
    ]
    b_idx = min(5, max(0, s_brazo - 1))
    a_idx = min(1, max(0, s_antebrazo - 1))
    m_idx = min(1, max(0, s_muneca - 1))
    score_b = tabla_b[b_idx][a_idx][m_idx]

    tabla_c = [
        [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
        [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
        [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
        [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
        [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
        [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
        [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
        [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
        [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
        [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
        [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
        [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    ]
    return int(tabla_c[min(11, max(0, score_a - 1))][min(11, max(0, score_b - 1))] + 1)

# Alias de retrocompatibilidad para src/extractor.py
def computar_angulos_craneo_cervicales_2d(c7, ear, eye_or_nose, hip, sh, elb, wri, ind, knee=None, ankle=None):
    res = computar_angulos_completos_2d(c7, ear, eye_or_nose, hip, sh, elb, wri, ind, knee, ankle)
    return res[0], res[1], res[2], res[3]
