import sys
import os
import argparse
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.schemas import TELEMETRY_SCHEMA
from src.kinematics import compute_vector_angle_3d_udf
from src.scoring import make_rosa_matrix_udf, make_reba_matrix_udf
from src.config_loader import cargar_configuracion_metodo
from src.classifier import clasificar_puesto_automaticamente
from src.reporter import planificar_evidencias, generar_dictamen_ergonomico
from src.visualizer import extraer_candidatos_para_gemini, renderizar_imagenes_segun_instrucciones
from src.analytics import inicializar_y_guardar_bd, generar_boxplot_ergonomico

def main():
    parser = argparse.ArgumentParser(description="Motor Ergonómico Spark con Base de Datos y Boxplot")
    parser.add_argument("input_parquet", help="Parquet de entrada")
    parser.add_argument("output_parquet", help="Parquet de salida")
    parser.add_argument("video_path", help="Video MP4")
    parser.add_argument("--metodo", default="AUTO", choices=["AUTO", "ROSA", "REBA", "RULA"])
    parser.add_argument("--worker_id", default="OPERARIO_P2")
    args = parser.parse_args()

    # 1. Triage Ergonómico Automático
    metodo_seleccionado = args.metodo
    if metodo_seleccionado == "AUTO":
        print(f"\n[INFO] Evaluando puesto con IA en: {args.video_path}...")
        triage = clasificar_puesto_automaticamente(args.video_path)
        metodo_seleccionado = triage.get("metodo", "ROSA")

    cfg = cargar_configuracion_metodo(metodo_seleccionado)

    spark = SparkSession.builder \
        .appName(f"ErgoEngine_{metodo_seleccionado}") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()

    print(f"[INFO] === Procesando con Método: [{metodo_seleccionado}] | Video: {args.video_path} ===")
    df = spark.read.schema(TELEMETRY_SCHEMA).parquet(args.input_parquet)

    if args.worker_id:
        df = df.withColumn("worker_id", F.lit(args.worker_id))

    has_ear = "l_ear" in df.columns and "r_ear" in df.columns
    has_index = "r_index" in df.columns

    ear_x_expr = ((F.col("l_ear.x") + F.col("r_ear.x")) / 2.0) if has_ear else F.col("nose.x")
    ear_y_expr = ((F.col("l_ear.y") + F.col("r_ear.y")) / 2.0) if has_ear else F.col("nose.y")
    ear_z_expr = ((F.col("l_ear.z") + F.col("r_ear.z")) / 2.0) if has_ear else F.col("nose.z")

    mano_x_expr = F.col("r_index.x") if has_index else (F.col("r_wrist.x") + F.lit(0.05))
    mano_y_expr = F.col("r_index.y") if has_index else F.col("r_wrist.y")
    mano_z_expr = F.col("r_index.z") if has_index else F.col("r_wrist.z")

    df_vectors = df.withColumns({
        "c7_x": (F.col("l_shoulder.x") + F.col("r_shoulder.x")) / 2.0,
        "c7_y": (F.col("l_shoulder.y") + F.col("r_shoulder.y")) / 2.0,
        "c7_z": (F.col("l_shoulder.z") + F.col("r_shoulder.z")) / 2.0,
        "mid_hip_x": (F.col("l_hip.x") + F.col("r_hip.x")) / 2.0,
        "mid_hip_y": (F.col("l_hip.y") + F.col("r_hip.y")) / 2.0,
        "mid_hip_z": (F.col("l_hip.z") + F.col("r_hip.z")) / 2.0,
        "ear_x": ear_x_expr,
        "ear_y": ear_y_expr,
        "ear_z": ear_z_expr,
    }).withColumns({
        "v_tronco_x": F.col("mid_hip_x") - F.col("c7_x"),
        "v_tronco_y": F.col("mid_hip_y") - F.col("c7_y"),
        "v_tronco_z": F.col("mid_hip_z") - F.col("c7_z"),
        "v_cuello_x": F.col("ear_x") - F.col("c7_x"),
        "v_cuello_y": F.col("ear_y") - F.col("c7_y"),
        "v_cuello_z": F.col("ear_z") - F.col("c7_z"),
        "v_brazo_r_x": F.col("r_elbow.x") - F.col("r_shoulder.x"),
        "v_brazo_r_y": F.col("r_elbow.y") - F.col("r_shoulder.y"),
        "v_brazo_r_z": F.col("r_elbow.z") - F.col("r_shoulder.z"),
        "v_antebrazo_r_x": F.col("r_wrist.x") - F.col("r_elbow.x"),
        "v_antebrazo_r_y": F.col("r_wrist.y") - F.col("r_elbow.y"),
        "v_antebrazo_r_z": F.col("r_wrist.z") - F.col("r_elbow.z"),
        "v_mano_r_x": mano_x_expr - F.col("r_wrist.x"),
        "v_mano_r_y": mano_y_expr - F.col("r_wrist.y"),
        "v_mano_r_z": mano_z_expr - F.col("r_wrist.z"),
    })

    df_angles = df_vectors.withColumns({
        "ang_tronco": F.abs(F.lit(180.0) - compute_vector_angle_3d_udf(
            F.col("v_tronco_x"), F.col("v_tronco_y"), F.col("v_tronco_z"),
            F.lit(0.0), F.lit(-1.0), F.lit(0.0)
        )),
        "ang_cuello": compute_vector_angle_3d_udf(
            F.col("v_cuello_x"), F.col("v_cuello_y"), F.col("v_cuello_z"),
            F.lit(0.0), F.lit(-1.0), F.lit(0.0)
        ),
        "ang_brazo_der": compute_vector_angle_3d_udf(
            F.col("v_brazo_r_x"), F.col("v_brazo_r_y"), F.col("v_brazo_r_z"),
            F.col("v_tronco_x"), F.col("v_tronco_y"), F.col("v_tronco_z")
        ),
        "ang_muneca_der": compute_vector_angle_3d_udf(
            F.col("v_antebrazo_r_x"), F.col("v_antebrazo_r_y"), F.col("v_antebrazo_r_z"),
            F.col("v_mano_r_x"), F.col("v_mano_r_y"), F.col("v_mano_r_z")
        )
    })

    if metodo_seleccionado == "ROSA":
        rosa_udf = make_rosa_matrix_udf(spark)
        df_evaluated = df_angles.withColumns({
            "penal_tronco": F.when(F.col("ang_tronco") > 20.0, 1).otherwise(0),
            "penal_cuello": F.when(F.col("ang_cuello") > 25.0, 2).when(F.col("ang_cuello") > 15.0, 1).otherwise(0),
            "penal_brazo": F.when(F.col("ang_brazo_der") > 45.0, 2).when(F.col("ang_brazo_der") > 20.0, 1).otherwise(0),
            "score_silla": F.lit(3) + F.col("penal_tronco"),
            "score_perif": F.lit(3) + F.col("penal_cuello") + F.col("penal_brazo")
        }).withColumn("SCORE_FINAL", rosa_udf("score_silla", "score_perif"))
    else: # REBA
        reba_udf = make_reba_matrix_udf(spark)
        df_evaluated = df_angles.withColumns({
            "score_a": F.when(F.col("ang_tronco") > 60.0, 4).when(F.col("ang_tronco") > 20.0, 3).otherwise(2),
            "score_b": F.when(F.col("ang_brazo_der") > 90.0, 4).when(F.col("ang_brazo_der") > 45.0, 3).otherwise(2),
            "actividad": F.lit(1)
        }).withColumn("SCORE_FINAL", reba_udf("score_a", "score_b", "actividad"))

    fps = 30.0
    df_summary = df_evaluated.groupBy("session_id", "worker_id").agg(
        (F.count("frame_index") / fps).alias("duracion_total_seg"),
        F.round(F.percentile_approx("ang_tronco", 0.10), 1).alias("tronco_p10_deg"),
        F.round(F.percentile_approx("ang_tronco", 0.50), 1).alias("tronco_p50_deg"),
        F.round(F.percentile_approx("ang_tronco", 0.95), 1).alias("tronco_p95_deg"),
        F.round(F.percentile_approx("ang_cuello", 0.10), 1).alias("cuello_p10_deg"),
        F.round(F.percentile_approx("ang_cuello", 0.50), 1).alias("cuello_p50_deg"),
        F.round(F.percentile_approx("ang_cuello", 0.95), 1).alias("cuello_p95_deg"),
        F.round(F.percentile_approx("ang_brazo_der", 0.10), 1).alias("brazo_p10_deg"),
        F.round(F.percentile_approx("ang_brazo_der", 0.50), 1).alias("brazo_p50_deg"),
        F.round(F.percentile_approx("ang_brazo_der", 0.95), 1).alias("brazo_p95_deg"),
        F.max("SCORE_FINAL").alias("score_final")
    )

    df_summary.write.mode("overwrite").parquet(args.output_parquet)
    candidatos = extraer_candidatos_para_gemini(args.video_path) if os.path.exists(args.video_path) else []
    resumen_list = [row.asDict() for row in df_summary.collect()]

    # Convertir serie continua a Pandas para el Boxplot y la Base de Datos SQLite
    pdf_continuous = df_evaluated.select("session_id", "worker_id", "frame_index", "timestamp_ms", 
                                         "ang_tronco", "ang_cuello", "ang_brazo_der", "ang_muneca_der", "SCORE_FINAL").toPandas()

    for registro in resumen_list:
        w_id = registro["worker_id"]
        score_val = registro.get("score_final", 5)
        registro["metodo"] = metodo_seleccionado

        plan = planificar_evidencias(candidatos, metodo_seleccionado, score_val)
        out_img_dir = f"reportes/img/{w_id}"
        
        # 1. Renderizar Capturas
        if plan and os.path.exists(args.video_path):
            renderizar_imagenes_segun_instrucciones(args.video_path, plan, out_img_dir, w_id)

        # 2. Generar Boxplot Pericial
        boxplot_path = f"{out_img_dir}/boxplot_distribucion_postural.png"
        generar_boxplot_ergonomico(pdf_continuous, boxplot_path, w_id, metodo_seleccionado)

        # 3. Guardar en Base de Datos Relacional SQLite
        inicializar_y_guardar_bd(pdf_continuous, registro, "data/ergo_database.db")

        # 4. Generar Dictamen Pericial Markdown
        informe_md = generar_dictamen_ergonomico(registro, plan, metodo_seleccionado, f"img/{w_id}")
        archivo_reporte = f"reportes/Informe_{registro['session_id']}_{w_id}.md"
        
        with open(archivo_reporte, "w", encoding="utf-8") as f:
            f.write(informe_md)
        print(f"\n✅ Dictamen Integral generado en: {archivo_reporte}")

if __name__ == "__main__":
    main()
