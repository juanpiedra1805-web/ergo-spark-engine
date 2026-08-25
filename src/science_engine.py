import numpy as np

BIBLIOTECA_CIENTIFICA = {
    "ROSA": [
        "Sonne, M., Villalta, D. L., & Andrews, D. M. (2012). Development and evaluation of an office ergonomic risk checklist: ROSA – Rapid Office Strain Assessment. Applied Ergonomics, 43(1), 98-108.",
        "ISO 9241-5:1998. Ergonomic requirements for office work with visual display terminals (VDTs) – Part 5: Workstation layout and postural requirements.",
        "ISO 11226:2000. Ergonomics – Evaluation of static working postures. International Organization for Standardization."
    ],
    "REBA": [
        "Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA). Applied Ergonomics, 31(2), 201-205.",
        "UNE-EN 1005-4:2005+A1:2009. Seguridad de las máquinas. Comportamiento físico del ser humano. Parte 4: Evaluación de las posturas y movimientos de trabajo.",
        "Waters, T. R., Putz-Anderson, V., Garg, A., & Fine, L. J. (1993). Revised NIOSH equation for the design and evaluation of manual lifting tasks. Ergonomics, 36(7), 749-776."
    ],
    "RULA": [
        "McAtamney, L., & Corlett, E. N. (1993). RULA: a survey method for the investigation of work-related upper limb disorders. Applied Ergonomics, 24(2), 91-99.",
        "ISO 11226:2000. Ergonomics – Evaluation of static working postures."
    ],
    "EXO_HOMBRO": [
        "Maurice, P., et al. (2019). Objective and subjective effects of a passive upper-limb exoskeleton on overhead tasks. Applied Ergonomics, 80, 246-258.",
        "Spada, S., et al. (2017). Analysis of an upper limb passive exoskeleton in automotive assembly line. Procedia Manufacturing, 11, 405-412.",
        "ASTM F48 Committee on Exoskeletons and Exosuits (2020). Standards for Industrial and Occupational Wearable Robotics."
    ],
    "EXO_LUMBAR": [
        "de Looze, M. P., et al. (2016). Exoskeletons for industrial application and their potential effects on physical work load: A systematic review. Work, 54(1), 167-181.",
        "Bosch, T., et al. (2016). The effects of a passive back exoskeleton on muscle activity and discomfort during static holding tasks. Applied Ergonomics, 54, 11-17.",
        "Graham, R. B., et al. (2020). Evaluating the efficacy of a lumbar-support exoskeleton during repetitive material handling. Safety Science, 131, 104924."
    ]
}

def diagnosticar_intervencion_cientifica(metricas: dict, metodo: str = "ROSA") -> dict:
    t_p50 = metricas.get("tronco_p50_deg", 0.0)
    t_p95 = metricas.get("tronco_p95_deg", 0.0)
    c_p50 = metricas.get("cuello_p50_deg", 0.0)
    b_p50 = metricas.get("brazo_p50_deg", 0.0)
    b_p95 = metricas.get("brazo_p95_deg", 0.0)
    
    acciones = []
    prescripcion_exo = []
    citas_aplicables = list(BIBLIOTECA_CIENTIFICA.get(metodo, []))

    if metodo.upper() == "ROSA":
        if c_p50 > 25.0:
            acciones.append("Elevación del plano visual del monitor (10-15 cm) mediante brazo articulado para reducir el momento flexor cervical (C7-T1) a < 15° (ISO 9241-5).")
        if b_p50 > 20.0:
            acciones.append("Ajuste de altura de apoyabrazos y aproximación de teclado/ratón hacia el plano neutro de trabajo para suprimir la activación estática del trapecio superior.")
        if t_p50 > 20.0:
            acciones.append("Regulación de soporte lumbar dinámico en respaldo para preservar la lordosis fisiológica L1-L5 (ISO 11226).")
    else:
        if b_p50 > 45.0 or b_p95 > 60.0:
            prescripcion_exo.append({
                "tecnologia": "Exoesqueleto Pasivo de Asistencia Escapular y Miembros Superiores",
                "modelo_ref": "Paexo Shoulder / Levitate AIRFRAME (ASTM F48)",
                "indicacion_biomecanica": f"Elevación de brazos sostenida (Mediana: {b_p50}°, Pico P95: {b_p95}°). Trabajo sobre plano de hombros.",
                "beneficio_esperado": "Reducción de hasta un 50% de la actividad electromiográfica (EMG) en deltoides anterior y supraespinoso; transferencia de torque hacia las crestas ilíacas.",
                "evidencia": BIBLIOTECA_CIENTIFICA["EXO_HOMBRO"]
            })
            citas_aplicables.extend(BIBLIOTECA_CIENTIFICA["EXO_HOMBRO"])

        if t_p50 > 30.0 or t_p95 > 45.0:
            prescripcion_exo.append({
                "tecnologia": "Exoesqueleto Pasivo de Soporte Lumbar y Erector Espinal",
                "modelo_ref": "Laevo FLEX / Auxivo LiftSuit / Paexo Back (ISO 13482)",
                "indicacion_biomecanica": f"Flexión anterior forzada de tronco (Mediana: {t_p50}°, Pico P95: {t_p95}°). Sobrecarga en L5-S1.",
                "beneficio_esperado": "Disminución de 25-35% de la compresión intradiscal L5-S1 mediante láminas de fibra elástica que asisten la extensión del tronco.",
                "evidencia": BIBLIOTECA_CIENTIFICA["EXO_LUMBAR"]
            })
            citas_aplicables.extend(BIBLIOTECA_CIENTIFICA["EXO_LUMBAR"])

        if not prescripcion_exo:
            acciones.append("Rediseño antropométrico del plano de agarre y asistencia con polipastos mecánicos convencionales.")

    return {
        "acciones_ingenieria": acciones,
        "prescripcion_exoesqueletos": prescripcion_exo,
        "referencias_cientificas": list(set(citas_aplicables))
    }
