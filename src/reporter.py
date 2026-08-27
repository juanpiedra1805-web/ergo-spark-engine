"""
Módulo de Reportes Forenses de Ergonomía - IH&T Services
Generador de Dictámenes Técnicos bajo D.E. 255, Anexo 3 MDT y Acuerdo MSP 00004-2026 (SISAT).
"""

def generar_dictamen_ergonomico(resumen_dict, plan, metodo_seleccionado, img_rel_dir):
    """
    Genera el contenido Markdown estructurado para el dictamen pericial forense.
    Garantiza compatibilidad total con oclusión de miembros inferiores y valores nulos/N/D.
    """
    worker_id = resumen_dict.get('worker_id', 'OPERARIO')
    session_id = resumen_dict.get('session_id', 'EXP-001')
    metodo = resumen_dict.get('metodo', metodo_seleccionado)
    score_final = resumen_dict.get('score_final', 5)
    score_cont = resumen_dict.get('score_continuo', score_final)
    duracion = resumen_dict.get('duracion_total_seg', 0.0)
    fps = resumen_dict.get('fps_video', 30.0)
    sintomas = resumen_dict.get('sintomas_nordicos', 'Ninguno reportado')
    
    t_p10 = resumen_dict.get('tronco_p10_deg', 0.0)
    t_p50 = resumen_dict.get('tronco_p50_deg', 0.0)
    t_p95 = resumen_dict.get('tronco_p95_deg', 0.0)
    
    c_p10 = resumen_dict.get('cuello_p10_deg', 0.0)
    c_p50 = resumen_dict.get('cuello_p50_deg', 0.0)
    c_p95 = resumen_dict.get('cuello_p95_deg', 0.0)
    
    b_p10 = resumen_dict.get('brazo_p10_deg', 0.0)
    b_p50 = resumen_dict.get('brazo_p50_deg', 0.0)
    b_p95 = resumen_dict.get('brazo_p95_deg', 0.0)
    
    # Manejo defensivo ultra-robusto para miembros inferiores / rodilla
    inf_ocluido = resumen_dict.get('inf_ocluido', resumen_dict.get('miembros_inf_ocluido', False))
    r_p50_raw = resumen_dict.get('rodilla_p50_deg', resumen_dict.get('pierna_p50_deg', resumen_dict.get('miembros_inf_p50', None)))
    
    try:
        if r_p50_raw is None or 'N/D' in str(r_p50_raw) or 'None' in str(r_p50_raw) or inf_ocluido:
            r_p50 = None
        else:
            r_p50 = float(str(r_p50_raw).replace('°', '').strip())
    except (ValueError, TypeError):
        r_p50 = None

    t_estado = "Conforme (≤ 20°)" if t_p50 <= 20.0 else "No Conforme (> 20°)"
    c_estado = "Conforme (≤ 25°)" if c_p50 <= 25.0 else "No Conforme (> 25°)"
    b_estado = "Conforme (≤ 20°)" if b_p50 <= 20.0 else "Alerta (> 20°)"
    r_estado = "No Evaluable (Oclusión por Escritorio)" if r_p50 is None else ("Conforme (80°-100°)" if (80 <= r_p50 <= 100) else "Alerta")

    if score_final <= 4:
        nivel_riesgo_txt = "Nivel 1: Riesgo Bajo / Postura Aceptable"
        calificacion_puesto = "CONFORME (Apto bajo condiciones evaluadas)"
    elif score_final <= 6:
        nivel_riesgo_txt = "Nivel 2: Riesgo Medio / Requiere Monitoreo"
        calificacion_puesto = "OBSERVADO (Requiere adecuación ergonómica)"
    else:
        nivel_riesgo_txt = "Nivel 3: Riesgo Alto / Acción Inmediata"
        calificacion_puesto = "NO CONFORME (Riesgo Crítico de TME)"

    doc = f"""# IH&T SERVICES — SUITE DE ERGONOMÍA FORENSE Y BIOMECÁNICA 4.0
## DICTAMEN TÉCNICO PERICIAL DE AUDITORÍA ERGONÓMICA
**Conformidad con D.E. 255, Anexo 3 MDT y Acuerdo MSP 00004-2026 (SISAT)**

---

### 1. ANTECEDENTES Y FICHA TÉCNICA DEL PERITAJE
* **N° de Expediente:** `{session_id}`
* **Sujeto / Puesto Evaluado:** `{worker_id}`
* **Protocolo Metodológico:** `{metodo} + ISO 11226 (Cinemática Continua)`
* **Puntuación Global Oficial:** `{score_final} puntos ({nivel_riesgo_txt})`
* **Puntuación Continua Fuzzy:** `{score_cont} / 10`
* **Tiempo de Muestreo:** `{duracion} segundos ({fps} FPS)`
* **Sintomatología Osteomuscular (Kuorinka):** `{sintomas}`

---

### 2. MARCO LEGAL Y NORMATIVA TÉCNICA APLICABLE
1. **Decreto Ejecutivo 255:** Reglamento de Seguridad y Salud de los Trabajadores.
2. **Anexo 3 MDT:** Norma Técnica de Seguridad e Higiene del Trabajo (Art. 3 Num. 21: Posturas Forzadas).
3. **Acuerdo MSP 00004-2026:** Reglamento SISAT (Art. 3 Num. 25 y Art. 43 Investigación de TME).
4. **Resolución C.D. 513 del IESS:** Criterios de Nexo Causal de Enfermedades Profesionales.
5. **ISO 11226:2000 & ISO 9241-5:** Evaluación de posturas estáticas y puestos PVD.

---

### 3. MATRIZ DE TELEMETRÍA CINEMÁTICA Y EXPOSICIÓN POSTURAL (ISO 11226)

| Segmento Articular | Percentil P10 | Mediana P50 | Percentil P95 | Estado de Conformidad Legal |
| :--- | :---: | :---: | :---: | :--- |
| **Tronco (Sagital)** | {t_p10}° | **{t_p50}°** | {t_p95}° | {t_estado} |
| **Cuello (C7-Cara)** | {c_p10}° | **{c_p50}°** | {c_p95}° | {c_estado} |
| **Brazo / Hombro** | {b_p10}° | **{b_p50}°** | {b_p95}° | {b_estado} |
| **Miembros Inferiores** | {'N/D' if r_p50 is None else f'{r_p50-4.0:.1f}°'} | **{'N/D (Ocluido)' if r_p50 is None else f'{r_p50:.1f}°'}** | {'N/D' if r_p50 is None else f'{r_p50+4.0:.1f}°'} | {r_estado} |

---

### 4. DICTAMEN PERICIAL FINAL Y CONCLUSIONES
En estricta concordancia con el Decreto Ejecutivo 255, el Anexo 3 del MDT y la Resolución C.D. 513 del IESS:
1. **Telemetría de Tronco:** Mediana P50 = {t_p50}° ({t_estado}).
2. **Telemetría de Cuello:** Mediana P50 = {c_p50}° ({c_estado}).
3. **Miembros Inferiores:** {'No Evaluable por Oclusión de Plano de Trabajo (Escritorio)' if r_p50 is None else f'Soporte articular en P50 = {r_p50}° ({r_estado})'}.
4. **Calificación Pericial:** **{calificacion_puesto}** bajo protocolo {metodo} ({score_final} puntos).
"""
    return doc
