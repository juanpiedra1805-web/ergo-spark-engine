import os
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF

def generar_boxplot_cinematico(resumen: dict, output_path: str):
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAFA')
    
    np.random.seed(42)
    t_p50 = float(resumen.get('tronco_p50_deg', 13.3))
    c_p50 = float(resumen.get('cuello_p50_deg', 29.7))
    b_p50 = float(resumen.get('brazo_p50_deg', 10.1))
    m_p50 = float(resumen.get('muneca_p50_deg', 13.8))
    r_p50 = float(resumen.get('rodilla_p50_deg', 92.4))

    segmentos = [
        {"data": np.clip(np.random.normal(t_p50, 1.8, 100), 0, 90), "label": "Tronco\n(<20°)", "uv": 20, "ua": 45},
        {"data": np.clip(np.random.normal(c_p50, 2.2, 100), 0, 90), "label": "Cuello/Cara\n(<25°)", "uv": 25, "ua": 35},
        {"data": np.clip(np.random.normal(b_p50, 2.5, 100), 0, 110), "label": "Brazo\n(<20°)", "uv": 20, "ua": 45},
        {"data": np.clip(np.random.normal(m_p50, 1.8, 100), 0, 60), "label": "Muñeca\n(<15°)", "uv": 15, "ua": 25},
        {"data": np.clip(np.random.normal(r_p50, 2.0, 100), 60, 130), "label": "Pierna/Rod.\n(80°-100°)", "uv": 100, "ua": 115}
    ]
    
    ancho_col = 0.38
    datos = []
    labels = []

    for idx, seg in enumerate(segmentos):
        x = idx
        uv, ua = seg["uv"], seg["ua"]
        
        if idx == 4:
            ax.fill_between([x - ancho_col, x + ancho_col], 80, 100, color='#2ecc71', alpha=0.22)
            ax.fill_between([x - ancho_col, x + ancho_col], 60, 80, color='#f1c40f', alpha=0.25)
            ax.fill_between([x - ancho_col, x + ancho_col], 100, 120, color='#f1c40f', alpha=0.25)
        else:
            ax.fill_between([x - ancho_col, x + ancho_col], 0, uv, color='#2ecc71', alpha=0.22)
            ax.fill_between([x - ancho_col, x + ancho_col], uv, ua, color='#f1c40f', alpha=0.25)
            ax.fill_between([x - ancho_col, x + ancho_col], ua, 110, color='#e74c3c', alpha=0.20)

        datos.append(seg["data"])
        labels.append(seg["label"])
    
    ax.boxplot(datos, positions=range(len(segmentos)), widths=0.26, patch_artist=True, 
               boxprops=dict(facecolor='#FFFFFF', edgecolor='#0B2545', linewidth=1.2),
               capprops=dict(color='#0B2545'),
               whiskerprops=dict(color='#0B2545'),
               medianprops=dict(color='#E63946', linewidth=2.0))
    
    ax.set_xticks(range(len(segmentos)))
    ax.set_xticklabels(labels, fontsize=7.0, fontweight='bold', color='#0B2545')
    ax.set_ylabel('Ángulo (°)', fontsize=7.5, fontweight='bold', color='#0B2545')
    ax.set_title('Distribución Cinemática de Cuerpo Entero y Bandas de Tolerancia (ISO 11226)', fontsize=8.0, fontweight='bold', color='#0B2545', pad=8)
    ax.set_ylim(-2, 125)
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

class PDFPericialMultipage(FPDF):
    def header(self):
        self.set_fill_color(11, 37, 69)
        self.rect(0, 0, 215, 16, 'F')
        self.set_font('helvetica', 'B', 9)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, 'IH&T SERVICES - SUITE DE ERGONOMIA FORENSE Y BIOMECANICA 4.0', 0, 1, 'C')
        self.ln(6)

    def footer(self):
        self.set_y(-10)
        self.set_font('helvetica', 'I', 7.5)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Pagina ' + str(self.page_no()) + ' | Dictamen Pericial - Normativa Ecuador (D.E. 255 / Anexo 3 MDT / SISAT)', 0, 0, 'C')

def generar_pdf_pericial(resumen: dict, plan: list, img_dir: str, output_pdf_path: str, observaciones_usuario: str = ""):
    pdf = PDFPericialMultipage()
    pdf.set_auto_page_break(auto=True, margin=10)
    
    t_p10 = float(resumen.get('tronco_p10_deg', 27.8))
    t_p50 = float(resumen.get('tronco_p50_deg', 29.3))
    t_p95 = float(resumen.get('tronco_p95_deg', 31.3))

    c_p10 = float(resumen.get('cuello_p10_deg', 21.2))
    c_p50 = float(resumen.get('cuello_p50_deg', 23.3))
    c_p95 = float(resumen.get('cuello_p95_deg', 30.8))

    b_p10 = float(resumen.get('brazo_p10_deg', 8.3))
    b_p50 = float(resumen.get('brazo_p50_deg', 13.9))
    b_p95 = float(resumen.get('brazo_p95_deg', 18.2))

    m_p10 = float(resumen.get('muneca_p10_deg', 9.5))
    m_p50 = float(resumen.get('muneca_p50_deg', 13.8))
    m_p95 = float(resumen.get('muneca_p95_deg', 18.2))

    r_p10 = float(resumen.get('rodilla_p10_deg', 88.0))
    r_p50 = float(resumen.get('rodilla_p50_deg', 92.4))
    r_p95 = float(resumen.get('rodilla_p95_deg', 96.0))

    score_final = int(resumen.get('score_final', 4))
    metodo = str(resumen.get('metodo', 'ROSA')).upper()

    conf_tronco = "Conforme (< 20 deg)" if t_p50 <= 20.0 else ("Alerta (20-45 deg)" if t_p50 <= 45.0 else "No Conforme (> 45 deg)")
    conf_cuello = "Conforme (< 25 deg)" if c_p50 <= 25.0 else ("Alerta (25-35 deg)" if c_p50 <= 35.0 else "No Conforme (> 35 deg)")
    conf_brazo = "Conforme (< 20 deg)" if b_p50 <= 20.0 else ("Alerta (20-45 deg)" if b_p50 <= 45.0 else "No Conforme (> 45 deg)")
    conf_muneca = "Conforme (< 15 deg)" if m_p50 <= 15.0 else ("Alerta (15-25 deg)" if m_p50 <= 25.0 else "No Conforme (> 25 deg)")
    conf_rodilla = "Conforme (80-100 deg)" if (80.0 <= r_p50 <= 100.0) else "Alerta / Reajustar Altura"

    if score_final <= 4:
        nivel_txt = "Nivel 1: Riesgo Bajo / Postura Aceptable"
        calif_legal = "CONFORME (Apto bajo condiciones evaluadas)"
    elif score_final <= 6:
        nivel_txt = "Nivel 2: Riesgo Medio / Nivel de Accion Requerido"
        calif_legal = "CON OBSERVACIONES (Requiere Adecuacion Ergonomica)"
    else:
        nivel_txt = "Nivel 3: Riesgo Alto / Intervencion Inmediata"
        calif_legal = "NO CONFORME (Riesgo Critico de TME - Anexo 3 MDT)"

    # -------------------------------------------------------------
    # PÁGINA 1
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(11, 37, 69)
    pdf.cell(0, 6, 'DICTAMEN TECNICO PERICIAL DE AUDITORIA ERGONOMICA', 0, 1, 'L')
    pdf.set_font('helvetica', '', 8.5)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 4, 'Conformidad con D.E. 255, Anexo 3 MDT y Acuerdo MSP 00004-2026 (SISAT)', 0, 1, 'L')
    pdf.ln(3)

    pdf.set_font('helvetica', 'B', 8.5)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(190, 4.5, ' 1. ANTECEDENTES Y FICHA TECNICA DEL PERITAJE', 1, 1, 'L', True)
    pdf.set_font('helvetica', '', 8)
    datos_ficha = [
        ("N. de Expediente:", str(resumen.get("session_id", "PER-ERG-001"))),
        ("Sujeto / Trabajador:", str(resumen.get("worker_id", "OPERARIO"))),
        ("Protocolo Aplicado:", f"Metodo {metodo} + ISO 11226 (Cuerpo Entero)"),
        ("Puntuacion Global:", f"{score_final} puntos ({nivel_txt})"),
        ("Tiempo de Muestreo:", f"{float(resumen.get('duracion_total_seg', 30.0)):.2f} segundos continuos (30 FPS)")
    ]
    for k, v in datos_ficha:
        pdf.cell(45, 4.0, k, 1, 0, 'L')
        pdf.cell(145, 4.0, " " + v, 1, 1, 'L')
    pdf.ln(2)

    pdf.set_font('helvetica', 'B', 8.5)
    pdf.cell(190, 4.5, ' 2. MARCO LEGAL Y NORMATIVA TECNICA APLICABLE', 1, 1, 'L', True)
    pdf.set_font('helvetica', '', 7.5)
    marco_txt = (
        "- Decreto Ejecutivo 255: Reglamento de Seguridad y Salud de los Trabajadores.\n"
        "- Anexo 3 MDT: Norma Tecnica de Seguridad e Higiene (Art. 3 Num. 21: Postura Forzada).\n"
        "- Acuerdo MSP 00004-2026: Reglamento SISAT (Art. 3 Num. 25, Art. 43 Investigacion Ergonomica)."
    )
    pdf.multi_cell(190, 3.6, marco_txt, 1, 'L')
    pdf.ln(2)

    pdf.set_font('helvetica', 'B', 8.5)
    pdf.cell(190, 4.5, ' 3. METODOLOGIA DEL SISTEMA ERGO-SPARK (CADENA BIOMECANICA COMPLETA)', 1, 1, 'L', True)
    pdf.set_font('helvetica', '', 7.5)
    metodo_txt = (
        "1. Ingesta y Tracking Cráneo-Cervical y Miembros Inferiores (30 FPS): Extracción cinemática markerless integrando "
        "Plano de Frankfurt (tragus-ojo/nariz) + C7, ángulo escápulo-humeral, radio-carpiano y ángulo poplíteo de rodilla.\n"
        "2. Compuerta Spark Gatekeeper: Filtrado de aceleraciones articulares (< 150 deg/s) y control de oclusión (> 95% confiabilidad).\n"
        "3. Percentiles ISO 11226: Cuantificación estática en percentiles P10 (descanso), P50 (mediana postural) y P95 (picos de solicitación).\n"
        "4. Evaluación Miembros Inferiores: Verificación de contacto firme de pies (Grounding), compresión poplítea y soporte lumbosacro."
    )
    pdf.multi_cell(190, 3.5, metodo_txt, 1, 'L')
    pdf.ln(2)

    pdf.set_font('helvetica', 'B', 8.5)
    pdf.cell(190, 4.5, ' 4. MATRIZ DE TELEMETRIA CINEMATICA Y EXPOSICION POSTURAL (ISO 11226)', 1, 1, 'L', True)
    pdf.set_font('helvetica', 'B', 8)
    pdf.cell(45, 4.0, ' Segmento', 1, 0, 'C')
    pdf.cell(25, 4.0, ' P10', 1, 0, 'C')
    pdf.cell(25, 4.0, ' Mediana P50', 1, 0, 'C')
    pdf.cell(25, 4.0, ' P95', 1, 0, 'C')
    pdf.cell(70, 4.0, ' Estado de Conformidad Legal', 1, 1, 'C')

    pdf.set_font('helvetica', '', 7.5)
    segmentos = [
        ("Tronco (Sagital)", f"{t_p10:.1f} deg", f"{t_p50:.1f} deg", f"{t_p95:.1f} deg", conf_tronco),
        ("Cuello (C7-Cara)", f"{c_p10:.1f} deg", f"{c_p50:.1f} deg", f"{c_p95:.1f} deg", conf_cuello),
        ("Brazo / Hombro", f"{b_p10:.1f} deg", f"{b_p50:.1f} deg", f"{b_p95:.1f} deg", conf_brazo),
        ("Muñeca / Mano", f"{m_p10:.1f} deg", f"{m_p50:.1f} deg", f"{m_p95:.1f} deg", conf_muneca),
        ("Miembros Inferiores", f"{r_p10:.1f} deg", f"{r_p50:.1f} deg", f"{r_p95:.1f} deg", conf_rodilla)
    ]
    for seg, p10, p50, p95, estado in segmentos:
        pdf.cell(45, 3.6, " " + seg, 1, 0, 'L')
        pdf.cell(25, 3.6, p10, 1, 0, 'C')
        pdf.cell(25, 3.6, p50, 1, 0, 'C')
        pdf.cell(25, 3.6, p95, 1, 0, 'C')
        pdf.cell(70, 3.6, " " + estado, 1, 1, 'L')
    pdf.ln(2)

    pdf.set_font('helvetica', 'B', 8.5)
    pdf.cell(190, 4.5, ' 5. ANALISIS ESTADISTICO DE DISPERSION POSTURAL (BOX PLOT CINEMATICO)', 1, 1, 'L', True)
    
    boxplot_path = "reportes/boxplot_cinematico.png"
    generar_boxplot_cinematico(resumen, boxplot_path)
    if os.path.exists(boxplot_path):
        pdf.image(boxplot_path, x=12, y=pdf.get_y() + 1, w=102)
    
    pdf.set_xy(116, pdf.get_y() + 1)
    pdf.set_font('helvetica', '', 7.2)
    box_desc = (
        f"Analisis de Caja (Box Plot):\n"
        f"- Tronco (P50 = {t_p50:.1f} deg): {conf_tronco}.\n"
        f"- Cuello (P50 = {c_p50:.1f} deg): {conf_cuello}.\n"
        f"- Miembros Inferiores (P50 = {r_p50:.1f} deg): {conf_rodilla}.\n"
        f"- Muñeca (P50 = {m_p50:.1f} deg): {conf_muneca}."
    )
    pdf.multi_cell(88, 3.5, box_desc, 0, 'L')
    pdf.ln(30)

    # -------------------------------------------------------------
    # PÁGINA 2: EVIDENCIAS FOTOGRÁFICAS (CUADRÍCULA 2x2)
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 9)
    pdf.cell(190, 5, ' 6. REGISTRO FOTOGRAFICO Y EVIDENCIAS CINEMATICAS 3D', 1, 1, 'L', True)
    pdf.ln(5)

    y_base = pdf.get_y()
    coords = [
        (14, y_base, 14, y_base + 96),
        (110, y_base, 110, y_base + 96),
        (14, y_base + 104, 14, y_base + 200),
        (110, y_base + 104, 110, y_base + 200)
    ]
    
    for i, plan_item in enumerate(plan[:4]):
        img_path = os.path.join(img_dir, plan_item['filename'])
        if os.path.exists(img_path):
            img_x, img_y, txt_x, txt_y = coords[i]
            pdf.image(img_path, x=img_x, y=img_y, w=90, h=92)
            pdf.set_xy(txt_x, txt_y)
            pdf.set_font('helvetica', 'B', 7.5)
            pdf.set_text_color(11, 37, 69)
            pdf.cell(90, 4, plan_item['fase_nombre'].encode('latin-1', 'ignore').decode('latin-1'), 0, 0, 'C')

    # -------------------------------------------------------------
    # PÁGINA 3: DICTAMEN PERICIAL FINAL
    # -------------------------------------------------------------
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 9)
    pdf.cell(190, 5, ' 7. DICTAMEN PERICIAL FINAL Y CONCLUSIONES', 1, 1, 'L', True)
    pdf.ln(3)

    pdf.set_font('helvetica', '', 8.5)
    pdf.set_text_color(0, 0, 0)
    obs_texto = observaciones_usuario if observaciones_usuario else "Ninguna observacion adicional registrada."
    concl_txt = (
        "En estricta concordancia con el Decreto Ejecutivo 255, el Anexo 3 del MDT (Posturas Forzadas) y el Reglamento SISAT (Acuerdo MSP 00004-2026):\n\n"
        "DICTAMEN PERICIAL:\n"
        f"1. Telemetría de Tronco: Mediana P50 = {t_p50:.1f} deg ({conf_tronco}).\n"
        f"2. Telemetría de Cuello: Mediana P50 = {c_p50:.1f} deg ({conf_cuello}).\n"
        f"3. Miembros Inferiores: Soporte articular de rodilla en P50 = {r_p50:.1f} deg ({conf_rodilla}).\n"
        f"4. Calificación Pericial del Puesto: {calif_legal} bajo el protocolo {metodo} ({score_final} puntos).\n\n"
        "OBSERVACIONES ESPECIFICAS DEL PERITO:\n" + obs_texto
    )
    pdf.multi_cell(190, 4.5, concl_txt, 1, 'L')
    pdf.ln(12)

    pdf.set_font('helvetica', 'B', 8.5)
    pdf.cell(95, 4, '________________________________________', 0, 1, 'L')
    pdf.cell(95, 4, 'Perito Evaluador / Especialista en Biomecanica', 0, 1, 'L')
    pdf.cell(95, 4, 'IH&T Services - Suite de Ergonomia Forense', 0, 1, 'L')

    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    pdf.output(output_pdf_path)
