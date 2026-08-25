# ERGO-SPARK ENGINE: BASE DE CONOCIMIENTO BIOMECÁNICA, ARQUITECTURA Y CÓDIGO

## 1. DESCRIPCIÓN GENERAL Y OBJETIVO
Suite de software para auditoría ergonómica pericial y biomecánica computacional (IH&T Services).
Procesa videos (30 FPS), extrae cinemática markerless (MediaPipe Pose), evalúa exposición continua (ISO 11226: P10, P50, P95), aplica matrices oficiales (ROSA, RULA, REBA), anonimiza rostros (LOPDP Ecuador) y emite dictámenes periciales forenses en PDF de 3 páginas (D.E. 255, Anexo 3 MDT, SISAT MSP Acuerdo 00004-2026, Res. C.D. 513 IESS).

---

## 2. ÁRBOL DE ARCHIVOS Y MÓDULOS
- `app.py`: Frontend Streamlit, selección de métodos, tabs de visualización y campo de texto editable (`st.text_area`) para notas de campo del perito.
- `src/kinematics.py`: Modelos vectoriales 2D/3D. Modelo cráneo-cervical combinando Plano de Frankfurt (tragus a comisura ocular/nasal) con vector vertebral C7. Matrices oficiales completas de ROSA (Tabla C), RULA (Tablas A, B, C) y REBA (Tablas A, B, C + modificador de actividad).
- `src/extractor.py`: Inferencia de landmarks anatómicos con MediaPipe Pose.
- `src/coherence_validator.py`: Spark Gatekeeper: filtrado de outliers de aceleración angular (<150°/s) y cálculo del índice de confiabilidad (>95%).
- `src/visualizer.py`: Renderizado con OpenCV. Anonimización facial obligatoria por pixelado/mosaico (LOPDP) previo al trazo óseo y superposición de badges angulares y banners.
- `src/analytics.py`: Base de datos SQLite (`ergo_database.db`), cálculo de percentiles continuos (P10, P50, P95) y diagrama Box Plot segmentado por articulación con límites normativos específicos:
  * Tronco: < 20° (Verde), 20°-45° (Amarillo), > 45° (Rojo)
  * Cuello/Cara: < 25° (Verde), 25°-35° (Amarillo), > 35° (Rojo)
  * Brazo: < 20° (Verde), 20°-45° (Amarillo), > 45° (Rojo)
  * Muñeca: < 15° (Verde), 15°-25° (Amarillo), > 25° (Rojo)
- `src/science_engine.py`: Diagnóstico fisiopatológico de compresión discal, sobrecarga miofascial, prescripción jerarquizada de controles de ingeniería física y evaluación de exoesqueletos ocupacionales (ASTM F48 / ISO 13482).
- `src/reporter.py`: Generador de informe en Markdown con antecedentes, marco normativo ecuatoriano, capítulo metodológico Ergo-Spark, telemetría y prescripción.
- `src/pdf_generator.py`: Compilador PDF de 3 páginas (FPDF2) con antecedentes, marco legal, metodología Spark, matriz de telemetría, gráfico Box Plot, 4 capturas fotográficas anonimizadas, dictamen con observaciones personalizadas y bloque de firma pericial.

---

## 3. MARCO NORMATIVO Y PERICIAL DE ECUADOR
- **Decreto Ejecutivo 255:** Reglamento de Seguridad y Salud de los Trabajadores.
- **Anexo 3 MDT:** Norma Técnica de Seguridad e Higiene del Trabajo (Art. 3 Num. 21: Posturas Forzadas).
- **Acuerdo Ministerial MSP 00004-2026 (SISAT):** Art. 3 Num. 20 (Manifestaciones tempranas), Art. 3 Num. 25, Art. 43 (Investigación Ergonómica obligatoria).
- **Resolución C.D. 513 IESS:** Criterio higiénico-ergonómico de causalidad de enfermedades profesionales (TME).
- **LOPDP:** Protección de datos personales mediante anonimización/pixelado facial en evidencias.
