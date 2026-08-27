import numpy as np
import pandas as pd

# RANGOS ANATÓMICOS FISIOLÓGICOS (ISO 11226 / Biomecánica Ocupacional)
LIMITES_ROM = {
    "ang_tronco": (0.0, 90.0),      # Flexión máxima funcional
    "ang_cuello": (0.0, 75.0),      # Flexión cervical fisiológica
    "ang_brazo_der": (0.0, 160.0),  # Elevación humeral
    "ang_muneca_der": (0.0, 60.0),  # Desviación de plano neutro
    "ang_rodilla": (20.0, 180.0)    # Flexión poplítea funcional (180°=extendida, <20°=cuclillas extrema)
}

MAX_VELOCIDAD_ANGULAR_FRAME = 35.0  # Grados por frame (a 30 fps = > 1050°/s es artefacto óptico)


def auditar_y_limpiar_telemetria_spark(df_angles):
    """
    Ejecuta el control de calidad pericial y compuerta de coherencia sobre PySpark DataFrame.
    (Carga perezosa de PySpark para entornos distribuidos).
    """
    try:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window
    except ImportError:
        raise ImportError("PySpark no está disponible en este entorno. Usa 'validar_coherencia_pandas'.")

    total_frames = df_angles.count()
    if total_frames == 0:
        return df_angles, {"score_confiabilidad": 0.0, "estado": "RECHAZADO_SIN_DATOS"}

    # 1. GATE 1: Filtro de Rango Anatómico (ROM Filter)
    df_valid = df_angles.withColumns({
        "valido_tronco": (F.col("ang_tronco") >= LIMITES_ROM["ang_tronco"][0]) & (F.col("ang_tronco") <= LIMITES_ROM["ang_tronco"][1]),
        "valido_cuello": (F.col("ang_cuello") >= LIMITES_ROM["ang_cuello"][0]) & (F.col("ang_cuello") <= LIMITES_ROM["ang_cuello"][1]),
        "valido_brazo": (F.col("ang_brazo_der") >= LIMITES_ROM["ang_brazo_der"][0]) & (F.col("ang_brazo_der") <= LIMITES_ROM["ang_brazo_der"][1]),
        "valido_muneca": (F.col("ang_muneca_der") >= LIMITES_ROM["ang_muneca_der"][0]) & (F.col("ang_muneca_der") <= LIMITES_ROM["ang_muneca_der"][1])
    })

    # 2. GATE 2: Filtro de Continuidad Temporal
    w_spec = Window.partitionBy("session_id", "worker_id").orderBy("frame_index")
    df_continuity = df_valid.withColumns({
        "lag_tronco": F.lag("ang_tronco", 1).over(w_spec),
        "lag_brazo": F.lag("ang_brazo_der", 1).over(w_spec),
    }).withColumns({
        "delta_tronco": F.coalesce(F.abs(F.col("ang_tronco") - F.col("lag_tronco")), F.lit(0.0)),
        "delta_brazo": F.coalesce(F.abs(F.col("ang_brazo_der") - F.col("lag_brazo")), F.lit(0.0)),
    }).withColumn(
        "es_continuo",
        (F.col("delta_tronco") <= MAX_VELOCIDAD_ANGULAR_FRAME) & (F.col("delta_brazo") <= MAX_VELOCIDAD_ANGULAR_FRAME)
    )

    df_flagged = df_continuity.withColumn(
        "frame_valido",
        F.col("valido_tronco") & F.col("valido_cuello") & F.col("valido_brazo") & F.col("valido_muneca") & F.col("es_continuo")
    )

    frames_validos = df_flagged.filter(F.col("frame_valido") == True).count()
    ratio_calidad = round((frames_validos / total_frames) * 100.0, 1)

    if ratio_calidad >= 90.0:
        dictamen_calidad = "EXCELENTE (Apto para Dictamen Pericial Forense)"
        color_badge = "success"
    elif ratio_calidad >= 75.0:
        dictamen_calidad = "ACEPTABLE CON OBSERVACIONES (Filtro de Ruido Aplicado)"
        color_badge = "warning"
    else:
        dictamen_calidad = "NO CONFORME (Elevada Oclusión / Ángulo de Cámara Inadecuado)"
        color_badge = "danger"

    metricas_calidad = {
        "total_frames_analizados": total_frames,
        "frames_validos_limpios": frames_validos,
        "frames_anomalos_filtrados": total_frames - frames_validos,
        "score_confiabilidad_pct": ratio_calidad,
        "dictamen_integridad": dictamen_calidad,
        "color_badge": color_badge
    }

    df_clean = df_flagged.filter(F.col("frame_valido") == True) if frames_validos > 10 else df_angles
    return df_clean, metricas_calidad


def validar_coherencia_pandas(df_pdf: pd.DataFrame) -> tuple:
    """
    Versión vectorizada ultra-rápida de la compuerta para ejecución en la App Streamlit.
    """
    total = len(df_pdf)
    if total == 0:
        return df_pdf, {"score_confiabilidad_pct": 0.0, "dictamen_integridad": "SIN_DATOS", "color_badge": "danger"}

    # Máscara de rangos fisiológicos
    m_tronco = (df_pdf["ang_tronco"] >= LIMITES_ROM["ang_tronco"][0]) & (df_pdf["ang_tronco"] <= LIMITES_ROM["ang_tronco"][1])
    m_cuello = (df_pdf["ang_cuello"] >= LIMITES_ROM["ang_cuello"][0]) & (df_pdf["ang_cuello"] <= LIMITES_ROM["ang_cuello"][1])
    m_brazo = (df_pdf["ang_brazo_der"] >= LIMITES_ROM["ang_brazo_der"][0]) & (df_pdf["ang_brazo_der"] <= LIMITES_ROM["ang_brazo_der"][1])
    m_muneca = (df_pdf["ang_muneca_der"] >= LIMITES_ROM["ang_muneca_der"][0]) & (df_pdf["ang_muneca_der"] <= LIMITES_ROM["ang_muneca_der"][1])

    # Rodilla: si la pierna está ocluida, ang_rodilla=0.0 es un placeholder (no una medición fuera de rango)
    if "ang_rodilla" in df_pdf.columns and "ocluido" in df_pdf.columns:
        rango_ok = (df_pdf["ang_rodilla"] >= LIMITES_ROM["ang_rodilla"][0]) & (df_pdf["ang_rodilla"] <= LIMITES_ROM["ang_rodilla"][1])
        m_rodilla = (df_pdf["ocluido"] == 1) | rango_ok
    else:
        m_rodilla = pd.Series(True, index=df_pdf.index)

    # Continuidad temporal
    d_tronco = df_pdf["ang_tronco"].diff().abs().fillna(0.0)
    d_brazo = df_pdf["ang_brazo_der"].diff().abs().fillna(0.0)
    m_continuidad = (d_tronco <= MAX_VELOCIDAD_ANGULAR_FRAME) & (d_brazo <= MAX_VELOCIDAD_ANGULAR_FRAME)

    valido = m_tronco & m_cuello & m_brazo & m_muneca & m_rodilla & m_continuidad
    df_clean = df_pdf[valido].copy()
    
    valid_count = len(df_clean)
    score_pct = round((valid_count / total) * 100.0, 1)

    if score_pct >= 90.0:
        dictamen = "CERTIFICADO (Alta Confiabilidad Biomecánica)"
        badge = "success"
    elif score_pct >= 75.0:
        dictamen = "APROBADO (Conclusiones tras Filtrado de Ruido)"
        badge = "warning"
    else:
        dictamen = "CRÍTICO (Excesiva Oclusión o Interferencia Óptica)"
        badge = "danger"

    metricas_calidad = {
        "total_frames_analizados": total,
        "frames_validos_limpios": valid_count,
        "frames_anomalos_filtrados": total - valid_count,
        "score_confiabilidad_pct": score_pct,
        "dictamen_integridad": dictamen,
        "color_badge": badge,
        "diagnostico_compuertas_pct": {
            "rango_tronco": round(float(m_tronco.mean()) * 100.0, 1),
            "rango_cuello": round(float(m_cuello.mean()) * 100.0, 1),
            "rango_brazo": round(float(m_brazo.mean()) * 100.0, 1),
            "rango_muneca": round(float(m_muneca.mean()) * 100.0, 1),
            "rango_rodilla": round(float(m_rodilla.mean()) * 100.0, 1),
            "continuidad_temporal": round(float(m_continuidad.mean()) * 100.0, 1),
        }
    }

    return (df_clean if valid_count > 10 else df_pdf), metricas_calidad
