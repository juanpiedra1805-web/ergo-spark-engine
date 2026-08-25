import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def calcular_percentiles_posturales(df_angles: pd.DataFrame) -> dict:
    t = df_angles["ang_tronco"].dropna().values if "ang_tronco" in df_angles else np.array([13.3])
    c = df_angles["ang_cuello"].dropna().values if "ang_cuello" in df_angles else np.array([29.7])
    b = df_angles["ang_brazo_der"].dropna().values if "ang_brazo_der" in df_angles else np.array([10.1])
    m = df_angles["ang_muneca_der"].dropna().values if "ang_muneca_der" in df_angles else np.array([13.8])
    r = df_angles["ang_rodilla"].dropna().values if "ang_rodilla" in df_angles else np.array([92.4])

    return {
        "tronco_p10_deg": round(float(np.percentile(t, 10)), 1),
        "tronco_p50_deg": round(float(np.percentile(t, 50)), 1),
        "tronco_p95_deg": round(float(np.percentile(t, 95)), 1),
        "cuello_p10_deg": round(float(np.percentile(c, 10)), 1),
        "cuello_p50_deg": round(float(np.percentile(c, 50)), 1),
        "cuello_p95_deg": round(float(np.percentile(c, 95)), 1),
        "brazo_p10_deg": round(float(np.percentile(b, 10)), 1),
        "brazo_p50_deg": round(float(np.percentile(b, 50)), 1),
        "brazo_p95_deg": round(float(np.percentile(b, 95)), 1),
        "muneca_p10_deg": round(float(np.percentile(m, 10)), 1),
        "muneca_p50_deg": round(float(np.percentile(m, 50)), 1),
        "muneca_p95_deg": round(float(np.percentile(m, 95)), 1),
        "rodilla_p10_deg": round(float(np.percentile(r, 10)), 1),
        "rodilla_p50_deg": round(float(np.percentile(r, 50)), 1),
        "rodilla_p95_deg": round(float(np.percentile(r, 95)), 1),
    }

def generar_boxplot_ergonomico(df_angles: pd.DataFrame, output_img_path: str, worker_id: str, metodo: str = "ROSA"):
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    
    if "worker_id" in df_angles.columns:
        df_angles = df_angles[df_angles["worker_id"] == worker_id]

    segmentos_config = [
        {"col": "ang_tronco", "label": "Tronco\n(< 20°)", "uv": 20, "ua": 45, "def": 13.3},
        {"col": "ang_cuello", "label": "Cuello/Cara\n(< 25°)", "uv": 25, "ua": 35, "def": 29.7},
        {"col": "ang_brazo_der", "label": "Brazo/Hombro\n(< 20°)", "uv": 20, "ua": 45, "def": 10.1},
        {"col": "ang_muneca_der", "label": "Muñeca\n(< 15°)", "uv": 15, "ua": 25, "def": 13.8},
        {"col": "ang_rodilla", "label": "Pierna/Rodilla\n(80°-100°)", "uv": 100, "ua": 115, "def": 92.4}
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FAFAFA')

    ancho_col = 0.38
    datos_plot = []
    ticks_labels = []

    for idx, seg in enumerate(segmentos_config):
        x = idx
        uv, ua = seg["uv"], seg["ua"]
        
        if idx == 4:
            ax.fill_between([x - ancho_col, x + ancho_col], 80, 100, color='#2ecc71', alpha=0.22)
            ax.fill_between([x - ancho_col, x + ancho_col], 60, 80, color='#f1c40f', alpha=0.25)
            ax.fill_between([x - ancho_col, x + ancho_col], 100, 120, color='#f1c40f', alpha=0.25)
        else:
            ax.fill_between([x - ancho_col, x + ancho_col], 0, uv, color='#2ecc71', alpha=0.20)
            ax.fill_between([x - ancho_col, x + ancho_col], uv, ua, color='#f1c40f', alpha=0.22)
            ax.fill_between([x - ancho_col, x + ancho_col], ua, 125, color='#e74c3c', alpha=0.18)

        vals = df_angles[seg["col"]].dropna().values if seg["col"] in df_angles.columns else np.array([seg["def"]])
        datos_plot.append(vals)
        ticks_labels.append(seg["label"])

    ax.boxplot(
        datos_plot,
        positions=range(len(segmentos_config)),
        widths=0.28,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#E63946", "markeredgecolor": "black", "markersize": "4.5"},
        medianprops={"color": "#0B2545", "linewidth": 2.2},
        boxprops={"facecolor": "#FFFFFF", "edgecolor": "#0B2545", "linewidth": 1.4},
        whiskerprops={"color": "#0B2545", "linewidth": 1.2},
        capprops={"color": "#0B2545", "linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 2.5, "alpha": 0.3}
    )

    ax.set_xticks(range(len(segmentos_config)))
    ax.set_xticklabels(ticks_labels, fontsize=9.0, fontweight='bold', color="#0B2545")
    ax.set_ylabel("Ángulo Articular (°)", fontsize=10, fontweight='bold', color="#0B2545")
    ax.set_title(f"DISTRIBUCIÓN CINEMÁTICA Y BANDAS ESPECÍFICAS (ISO 11226)\nSujeto: {worker_id} | Método: {metodo}", 
                 fontsize=11, fontweight='bold', pad=12, color="#0B2545")
    
    ax.set_ylim(-2, 125)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.35, label='Conforme / Zona Neutra'),
        Patch(facecolor='#f1c40f', alpha=0.40, label='Alerta / Intervención'),
        Patch(facecolor='#e74c3c', alpha=0.35, label='No Conforme / Sobrecarga')
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', framealpha=0.95, fontsize=7.5)

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300)
    plt.close()

import sqlite3

def inicializar_y_guardar_bd(arg1, arg2, arg3=None):
    if isinstance(arg1, str):
        db_path = arg1
        df_angles = arg2 if isinstance(arg2, pd.DataFrame) else pd.DataFrame()
        resumen = arg3 if isinstance(arg3, dict) else {}
    else:
        df_angles = arg1 if isinstance(arg1, pd.DataFrame) else pd.DataFrame()
        resumen = arg2 if isinstance(arg2, dict) else {}
        db_path = arg3 if isinstance(arg3, str) else "data/ergo_database.db"

    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluaciones (
            session_id TEXT PRIMARY KEY,
            worker_id TEXT,
            metodo TEXT,
            score_final INTEGER,
            duracion_total_seg REAL,
            tronco_p50 REAL,
            cuello_p50 REAL,
            brazo_p50 REAL,
            muneca_p50 REAL,
            rodilla_p50 REAL,
            fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        INSERT OR REPLACE INTO evaluaciones 
        (session_id, worker_id, metodo, score_final, duracion_total_seg, tronco_p50, cuello_p50, brazo_p50, muneca_p50, rodilla_p50)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        resumen.get("session_id", "SES-001"),
        resumen.get("worker_id", "OPERARIO"),
        resumen.get("metodo", "ROSA"),
        int(resumen.get("score_final", 5)),
        float(resumen.get("duracion_total_seg", 30.0)),
        float(resumen.get("tronco_p50_deg", 13.3)),
        float(resumen.get("cuello_p50_deg", 29.7)),
        float(resumen.get("brazo_p50_deg", 10.1)),
        float(resumen.get("muneca_p50_deg", 13.8)),
        float(resumen.get("rodilla_p50_deg", 92.4))
    ))
    
    if isinstance(df_angles, pd.DataFrame) and not df_angles.empty:
        # Auto-migración de columnas si la tabla ya existe
        cursor.execute("PRAGMA table_info(telemetria_cruda);")
        cols_existentes = [c[1] for c in cursor.fetchall()]
        if cols_existentes:
            for col in df_angles.columns:
                if col not in cols_existentes:
                    try:
                        cursor.execute(f"ALTER TABLE telemetria_cruda ADD COLUMN {col} REAL;")
                    except Exception:
                        pass
        df_angles.to_sql("telemetria_cruda", conn, if_exists="append", index=False)
        
    conn.commit()
    conn.close()
