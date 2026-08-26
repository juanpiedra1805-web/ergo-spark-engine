import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tempfile
import sqlite3
import cv2
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="IH&T Services | Suite de Ergonomía Forense 4.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.extractor import procesar_video
from src.classifier import clasificar_puesto_automaticamente
from src.reporter import planificar_evidencias, generar_dictamen_ergonomico
from src.visualizer import extraer_candidatos_para_gemini, renderizar_imagenes_segun_instrucciones
from src.analytics import inicializar_y_guardar_bd, generar_boxplot_ergonomico
from src.science_engine import diagnosticar_intervencion_cientifica
from src.pdf_generator import generar_pdf_pericial
from src.kinematics import calcular_matriz_rosa_oficial
from src.coherence_validator import validar_coherencia_pandas

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .iht-header-container {
        background: linear-gradient(135deg, #0B2545 0%, #133E87 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(11, 37, 69, 0.15);
    }
    .iht-title { font-size: 1.85rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }
    .iht-subtitle { font-size: 0.95rem; color: #93C5FD; }
    .iht-tagline {
        display: inline-block;
        background: rgba(255, 255, 255, 0.12);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 8px;
        color: #60A5FA;
    }
    .kpi-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .kpi-title { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; color: #64748B; margin-bottom: 6px; }
    .kpi-value { font-size: 2.1rem; font-weight: 700; line-height: 1.1; margin-bottom: 6px; }
    .kpi-sub { font-size: 0.8rem; font-weight: 500; }
    .border-danger { border-left: 5px solid #DC2626; }
    .border-warning { border-left: 5px solid #D97706; }
    .border-success { border-left: 5px solid #0D9488; }
    .text-danger { color: #DC2626; }
    .text-warning { color: #D97706; }
    .text-success { color: #0D9488; }
    .quality-banner {
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .quality-success { background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; }
    .quality-warning { background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }
    .quality-danger { background: #FEF2F2; border: 1px solid #FECACA; color: #991B1B; }
    .exo-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="iht-header-container">
    <div class="iht-tagline">SISTEMA INTEGRAL DE AUDITORÍA OCUPACIONAL</div>
    <div class="iht-title">IH&T Services — Ergonomía & Biomecánica 4.0</div>
    <div class="iht-subtitle">Plataforma Unificada bajo D.E. 255, Anexo 3 MDT y Acuerdo MSP 00004-2026 (SISAT).</div>
</div>
""", unsafe_allow_html=True)

st.markdown("#### **1. Carga de Registro Fílmico para Auditoría**")
uploaded_file = st.file_uploader(
    "Selecciona el archivo de video (.mp4, .mov, .avi) de la estación evaluada:",
    type=["mp4", "mov", "avi"]
)

default_worker_id = "OPERARIO_01"
if uploaded_file is not None:
    base_name = os.path.splitext(uploaded_file.name)[0].upper()
    default_worker_id = f"OPERARIO_{base_name}" if not base_name.startswith("OPERARIO") else base_name
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        st.session_state["auditoria_completada"] = False
        st.session_state["last_uploaded_name"] = uploaded_file.name

with st.sidebar:
    st.markdown("### **IH&T Services**")
    st.caption("Industrial Hygiene & Occupational Health Consulting")
    st.markdown("---")
    
    st.markdown("#### **Parámetros del Dictamen**")
    worker_id = st.text_input("Identificador del Sujeto / Puesto", value=default_worker_id)
    session_id = st.text_input("N° Expediente de Auditoría", value=f"PER-ERG-{default_worker_id}")
    
    metodo_opcion = st.selectbox(
        "Protocolo Metodológico",
        options=["AUTO (Triage con IA)", "ROSA (PVD / Sedestación)", "REBA (Cuerpo Entero / Dinámico)", "RULA (Carga Postural Superior)"],
        index=0
    )
    st.markdown("---")
    st.markdown("#### **Marco Normativo Ecuatoriano**")
    st.markdown("""
- **D.E. 255:** Reglamento Seguridad y Salud.
- **Anexo 3 MDT:** Norma Técnica Posturas Forzadas.
- **Acuerdo MSP 00004-2026:** Reglamento SISAT.
- **Res. C.D. 513 IESS:** Causalidad de TME.
- **Decisión 584 CAN:** Instrumento Andino SST.
- **ISO 11226 / ISO 9241-5:** Biomecánica y PVD.
""")
    st.markdown("---")
    st.caption("🌐 [www.ih-t.net](https://www.ih-t.net)")

if uploaded_file is not None:
    col_btn, _ = st.columns([1.5, 3])
    with col_btn:
        ejecutar_btn = st.button("🔬 Iniciar Auditoría Biomecánica", type="primary", use_container_width=True)

    if ejecutar_btn:
        with st.spinner("⏳ Ejecutando tracking cráneo-cervical, validación de coherencia y renderizado determinista..."):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            if "AUTO" in metodo_opcion:
                triage = clasificar_puesto_automaticamente(video_path)
                metodo_seleccionado = triage.get("metodo", "ROSA")
            elif "ROSA" in metodo_opcion:
                metodo_seleccionado = "ROSA"
            elif "REBA" in metodo_opcion:
                metodo_seleccionado = "REBA"
            else:
                metodo_seleccionado = "RULA"

            temp_parquet = tempfile.NamedTemporaryFile(delete=False, suffix='.parquet').name
            procesar_video(video_path, temp_parquet, session_id=session_id, worker_id=worker_id)

            df_raw = pd.read_parquet(temp_parquet)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            pdf_continuous, met_calidad = validar_coherencia_pandas(df_raw)

            t_clean = pdf_continuous["ang_tronco"].values
            c_clean = pdf_continuous["ang_cuello"].values
            b_clean = pdf_continuous["ang_brazo_der"].values
            m_clean = pdf_continuous["ang_muneca_der"].values

            t_p50 = float(np.median(t_clean)) if len(t_clean) > 0 else 20.0
            c_p50 = float(np.median(c_clean)) if len(c_clean) > 0 else 25.0
            b_p50 = float(np.median(b_clean)) if len(b_clean) > 0 else 20.0
            m_p50 = float(np.median(m_clean)) if len(m_clean) > 0 else 5.0

            if metodo_seleccionado == "ROSA":
                score_final = calcular_matriz_rosa_oficial(t_p50, c_p50, b_p50, m_p50)
            else:
                score_final = 8 if (t_p50 > 30.0 or b_p50 > 45.0) else 6

            pdf_continuous["SCORE_FINAL"] = score_final

            resumen_dict = {
                "session_id": session_id,
                "worker_id": worker_id,
                "metodo": metodo_seleccionado,
                "score_final": score_final,
                "duracion_total_seg": round(total_frames / fps, 2),
                "tronco_p10_deg": round(float(np.percentile(t_clean, 10)), 1) if len(t_clean) > 0 else 0.0,
                "tronco_p50_deg": round(t_p50, 1),
                "tronco_p95_deg": round(float(np.percentile(t_clean, 95)), 1) if len(t_clean) > 0 else 0.0,
                "cuello_p10_deg": round(float(np.percentile(c_clean, 10)), 1) if len(c_clean) > 0 else 0.0,
                "cuello_p50_deg": round(c_p50, 1),
                "cuello_p95_deg": round(float(np.percentile(c_clean, 95)), 1) if len(c_clean) > 0 else 0.0,
                "brazo_p10_deg": round(float(np.percentile(b_clean, 10)), 1) if len(b_clean) > 0 else 0.0,
                "brazo_p50_deg": round(b_p50, 1),
                "brazo_p95_deg": round(float(np.percentile(b_clean, 95)), 1) if len(b_clean) > 0 else 0.0
            }

            out_img_dir = f"reportes/img/{worker_id}"
            os.makedirs(out_img_dir, exist_ok=True)
            candidatos = extraer_candidatos_para_gemini(video_path)
            plan = planificar_evidencias(candidatos, metodo_seleccionado, score_final)
            renderizar_imagenes_segun_instrucciones(video_path, plan, pdf_continuous, out_img_dir, worker_id)

            boxplot_path = f"{out_img_dir}/boxplot_distribucion_postural.png"
            generar_boxplot_ergonomico(pdf_continuous, boxplot_path, worker_id, metodo_seleccionado)

            inicializar_y_guardar_bd(pdf_continuous, resumen_dict, "data/ergo_database.db")

            informe_md = generar_dictamen_ergonomico(resumen_dict, plan, metodo_seleccionado, f"img/{worker_id}")
            archivo_reporte = f"reportes/Informe_{session_id}_{worker_id}.md"
            with open(archivo_reporte, "w", encoding="utf-8") as f:
                f.write(informe_md)

            st.session_state["auditoria_completada"] = True
            st.session_state["resumen_dict"] = resumen_dict
            st.session_state["plan"] = plan
            st.session_state["informe_md"] = informe_md
            st.session_state["out_img_dir"] = out_img_dir
            st.session_state["boxplot_path"] = boxplot_path
            st.session_state["met_calidad"] = met_calidad
            st.session_state["diag_ciencia"] = diagnosticar_intervencion_cientifica(resumen_dict, metodo_seleccionado)
            st.success("✅ ¡Auditoría Biomecánica Unificada Finalizada con Éxito!")

if st.session_state.get("auditoria_completada", False):
    res = st.session_state["resumen_dict"]
    plan = st.session_state["plan"]
    img_dir = st.session_state["out_img_dir"]
    boxplot_file = st.session_state["boxplot_path"]
    inf_md = st.session_state["informe_md"]
    diag_cie = st.session_state["diag_ciencia"]
    calidad = st.session_state["met_calidad"]

    st.markdown("---")
    
    q_class = f"quality-{calidad['color_badge']}"
    st.markdown(f"""
    <div class="quality-banner {q_class}">
        <div>
            <b>🛡️ SELLO DE AUDITORÍA Y CONTROL DE CALIDAD BIOMECÁNICA (SPARK GATEKEEPER)</b><br>
            <span style="font-size:0.85rem;">Confiabilidad de Señal: <b>{calidad['score_confiabilidad_pct']}%</b> | Dictamen: <b>{calidad['dictamen_integridad']}</b></span>
        </div>
        <div style="font-size:1.1rem; font-weight:700;">
            {calidad['frames_validos_limpios']} / {calidad['total_frames_analizados']} Frames Íntegros
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### **2. Tablero de Control: `{res['worker_id']}` ({res['session_id']})**")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        score_val = res['score_final']
        b_color = "border-danger" if score_val >= 7 else ("border-warning" if score_val >= 5 else "border-success")
        t_color = "text-danger" if score_val >= 7 else ("text-warning" if score_val >= 5 else "text-success")
        st.markdown(f"""
        <div class="kpi-box {b_color}">
            <div class="kpi-title">Puntuación Global ({res['metodo']})</div>
            <div class="kpi-value {t_color}">{score_val} / 10</div>
            <div class="kpi-sub {t_color}">{'Nivel de Acción 3 (Muy Alto)' if score_val>=7 else 'Nivel de Acción 2 (Medio)'}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        t_val = res['tronco_p50_deg']
        b_color = "border-danger" if t_val > 20 else "border-success"
        t_color = "text-danger" if t_val > 20 else "text-success"
        st.markdown(f"""
        <div class="kpi-box {b_color}">
            <div class="kpi-title">Tronco — Mediana P50</div>
            <div class="kpi-value {t_color}">{t_val}°</div>
            <div class="kpi-sub">ISO 11226: {'No Conforme (>20°)' if t_val>20 else 'Conforme (<20°)'}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        c_val = res['cuello_p50_deg']
        b_color = "border-danger" if c_val > 25 else "border-success"
        t_color = "text-danger" if c_val > 25 else "text-success"
        st.markdown(f"""
        <div class="kpi-box {b_color}">
            <div class="kpi-title">Cuello (C7-Cara) — Mediana P50</div>
            <div class="kpi-value {t_color}">{c_val}°</div>
            <div class="kpi-sub">ISO 11226: {'No Conforme (>25°)' if c_val>25 else 'Conforme (<25°)'}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        b_val = res['brazo_p50_deg']
        b_color = "border-warning" if b_val > 20 else "border-success"
        t_color = "text-warning" if b_val > 20 else "text-success"
        st.markdown(f"""
        <div class="kpi-box {b_color}">
            <div class="kpi-title">Brazo / Hombro — Mediana P50</div>
            <div class="kpi-value {t_color}">{b_val}°</div>
            <div class="kpi-sub">ISO 11226: {'Alerta (>20°)' if b_val>20 else 'Conforme (<20°)'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📸 Evidencias Cinemáticas 3D", 
        "📊 Distribución Postural (ISO 11226)", 
        "🛡️ Integridad & Compuerta Spark",
        "🦾 Prescripción de Exoesqueletos",
        "📄 Dictamen Pericial Oficial", 
        "🗄️ Repositorio & Vigilancia Médica"
    ])

    with tab1:
        st.markdown("#### **Reconstrucción Fotográfica de Fases Biomecánicas**")
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        cols = [c1, c2, c3, c4]
        for i, col in enumerate(cols):
            if i < len(plan):
                fpath = os.path.join(img_dir, plan[i]['filename'])
                with col:
                    if os.path.exists(fpath):
                        st.image(fpath, caption=f"Figura {i+1}: {plan[i]['fase_nombre']}", use_container_width=True)

    with tab2:
        st.markdown("#### **Diagrama de Cajas y Bigotes con Bandas Normativas ISO 11226**")
        if os.path.exists(boxplot_file):
            st.image(boxplot_file, use_container_width=True)

    with tab3:
        st.markdown("#### **Reporte de Auditoría de Datos y Compuerta de Coherencia**")
        st.markdown(f"""
        * **Total de Muestras Cinemáticas:** `{calidad['total_frames_analizados']} fotogramas`
        * **Muestras Válidas Filtradas:** `{calidad['frames_validos_limpios']} fotogramas`
        * **Artefactos / Saltos de Tracking Descartados:** `{calidad['frames_anomalos_filtrados']} fotogramas`
        * **Índice de Confiabilidad Pericial:** **`{calidad['score_confiabilidad_pct']}%`**
        * **Veredicto:** `{calidad['dictamen_integridad']}`
        """)

    with tab4:
        st.markdown("#### **Prescripción Técnica de Exoesqueletos Ocupacionales**")
        if diag_cie.get("prescripcion_exoesqueletos"):
            for exo in diag_cie["prescripcion_exoesqueletos"]:
                st.markdown(f"""
                <div class="exo-card">
                    <h4 style="color:#15803D; margin-bottom:5px;">🦾 {exo['tecnologia']}</h4>
                    <p><b>Modelo de Referencia / Estándar:</b> {exo['modelo_ref']}</p>
                    <p><b>Criterio Biomecánico:</b> {exo['indicacion_biomecanica']}</p>
                    <p><b>Beneficio Fisiológico Demostrado:</b> {exo['beneficio_esperado']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ **Criterio Pericial:** El puesto evaluado (PVD/Oficina) se resuelve prioritariamente mediante adecuaciones antropométricas de ingeniería física convencional (ISO 9241-5). No se requiere equipamiento vestible.")

    with tab5:
        st.markdown("#### **Dictamen Técnico Pericial de Ergonomía Ocupacional**")
        
        st.markdown("---")
        st.markdown("##### **✍️ Campo de Observaciones y Recomendaciones del Perito**")
        comentarios_perito = st.text_area(
            "Ingrese notas de campo, detalles del trabajador o recomendaciones específicas que desea integrar directamente en la Sección 6 del PDF:",
            value=f"Evaluación realizada para el puesto de trabajo de {res['worker_id']}. Se recomienda realizar un reajuste ergonómico del atril de pantalla y seguimiento médico ocupacional en el SISAT en un plazo no mayor a 30 días.",
            height=110
        )
        
        os.makedirs("reportes", exist_ok=True)
        pdf_filename = f"reportes/Dictamen_{res['session_id']}_{res['worker_id']}.pdf"
        generar_pdf_pericial(res, plan, img_dir, pdf_filename, observaciones_usuario=comentarios_perito)
        
        with open(pdf_filename, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📥 Descargar Dictamen en Formato PDF (Oficial)",
                data=pdf_bytes,
                file_name=f"Dictamen_{res['session_id']}_{res['worker_id']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        with col_dl2:
            st.download_button(
                label="📄 Descargar Versión Fuente (.md)",
                data=inf_md,
                file_name=f"Dictamen_{res['session_id']}_{res['worker_id']}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        st.markdown("---")
        st.markdown("### Vista Previa del Informe Pericial")
        st.markdown(inf_md)

    with tab6:
        st.markdown("#### **Base de Datos Consolidada de Vigilancia Epidemiológica (SQLite)**")
        db_path = "data/ergo_database.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            try:
                df_bd = pd.read_sql_query("SELECT * FROM ergo_resumen_gold ORDER BY fecha_registro DESC", conn)
                if not df_bd.empty:
                    st.dataframe(df_bd, use_container_width=True)
                else:
                    st.info("ℹ️ La base de datos está inicializada pero aún no contiene registros consolidados.")
            except Exception:
                st.info("ℹ️ Aún no se han consolidado registros en la tabla histórica. Procesa una evaluación para generar la primera entrada.")
            finally:
                conn.close()
        else:
            st.info("ℹ️ Repositorio local SQLite pendiente de inicialización tras la primera auditoría.")
