# Análisis de accidentabilidad laboral y clasificación de incidentes con modelos de lenguaje (OSHA ITA)

Plataforma técnica de analítica de datos, ciencia de datos y procesamiento de lenguaje natural (PLN) aplicada a los registros oficiales de la Administración de Seguridad y Salud Ocupacional de los Estados Unidos (**OSHA - Injury Tracking Application**, ciclo regulatorio anual 2024–2025). 

El proyecto resuelve tres retos operativos fundamentales en seguridad industrial y medicina del trabajo:
1. **Cálculo y normalización de tasas oficiales de accidentabilidad (TIR y DART)** a 200,000 horas hombre trabajadas sobre una base empírica de **1.06 billones de horas** y **352,500 centros de trabajo**, desglosadas por sector económico (NAICS) y puesto operativo (SOC).
2. **Clasificación supervisada y explicable (XAI) de narrativas de incidentes** (redactadas en texto libre en inglés) en 6 categorías de riesgo crítico, comparando un modelo léxico clásico (TF-IDF + Regresión Logística, 91.97% exactitud) contra representaciones semánticas densas (Sentence Transformers `all-MiniLM-L6-v2`, 87.97% exactitud).
3. **Pronóstico clínico temprano de incapacidad prolongada (>30 días de baja médica, ROC-AUC 0.7192)** mediante minería de texto patológico durante las primeras 24 horas del reporte, asociando prescripciones técnicas bajo la jerarquía de controles de OSHA.

---

## 1. Resumen técnico del proyecto

### Analítica de accidentabilidad e indicadores de frecuencia
- **Integración relacional masiva (606 MB):** Cruce a nivel de establecimiento entre el resumen anual de lesiones (*Summary Data - Formulario 300A*, 352,500 centros) y el detalle circunstancial de cada incidente (*Case Detail Data - Formularios 300/301*, 688,649 casos individuales) mediante el identificador federal unívoco `establishment_id`.
- **Estandarización federal (29 CFR 1904):** Normalización basada en la constante reglamentaria de 200,000 horas de exposición (equivalente a 100 trabajadores a tiempo completo por año):
  - **TIR (*Total Incident Rate* / TRIR):** Casos registrables totales por 200,000 horas (0.278 nacional).
  - **DART (*Days Away, Restricted or Transferred*):** Casos con baja médica o restricción laboral por 200,000 horas (0.178 nacional).
  - **Severidad operativa acumulada:** Días perdidos (DAFW) y días en tareas restringidas (DJTR) por ocupación estándar (SOC).

### Modelado predictivo de lenguaje natural (PLN) y Explicabilidad (XAI)
- **Taxonomía de riesgo crítico:**
  - *Caída a distinto nivel (trabajo en altura)*
  - *Atrapamiento en maquinaria o elementos rotativos*
  - *Golpeado por caída o desprendimiento de carga*
  - *Incidente vehicular o vuelco de montacargas*
  - *Exposición a sustancias químicas peligrosas*
  - *Sobreesfuerzo y lesión musculoesquelética*
- **Evaluación experimental:**
  - **Modelo base:** Extracción de n-gramas léxicos con TF-IDF (1-2 gramos) y clasificador lineal de Regresión Logística (91.97% exactitud, 0.47s inferencia sobre 3,000 casos).
  - **Modelo semántico:** Embeddings densos de 384 dimensiones mediante `SentenceTransformer('all-MiniLM-L6-v2')` con clasificador supervisado (87.97% exactitud).
  - **Explicabilidad local (XAI):** Descomposición lineal de la frontera de decisión ($\text{TF-IDF} \times \beta$) que atribuye a cada palabra del reporte su peso matemático específico en la predicción.
- **Pronóstico temprano de severidad clínica (>30 días):**
  - Modelo logístico sobre patologías clave (ROC-AUC 0.7192, Sensibilidad 69.3%) contrastando términos de alto riesgo (`fracture`, `amputation`, `hernia`) frente a términos ambulatorios atenuantes (`needle`, `exposure`, `cut`).
- **Prescripción técnica inmediata:** Generación automática de controles de ingeniería, procedimientos administrativos (LOTO) y EPP según la jerarquía reglamentaria de OSHA.

---

## 2. Estructura del repositorio

```text
analisis-accidentes-osha/
│
├── datos/
│   ├── ITA_300A_Summary_Data_2024_through_12-31-2025_v2.csv   # Resumen anual 300A (87 MB)
│   ├── ITA_Case_Detail_Data_2024_through_12-31-2025.csv       # Casos y narrativas 300/301 (520 MB)
│   └── procesados/
│       ├── tasas_por_sector_naics.csv                        # Tasas TIR y DART por sector económico
│       ├── tasas_por_ocupacion_soc.csv                       # Días DAFW y DJTR por ocupación estándar
│       ├── dataset_narrativas_entrenamiento.csv              # 11,998 narrativas balanceadas
│       ├── dataset_narrativas_prueba.csv                     # 3,000 narrativas de prueba ciega
│       ├── predicciones_comparativas_test.csv                # 3,000 casos con ID oficial y predicciones
│       ├── metricas_comparativas.json                        # Métricas de benchmark de modelos
│       ├── matriz_confusion_comparativa.json                 # Matrices de confusión normalizadas
│       └── investigaciones_avanzadas.json                    # Fatiga horaria, Bird, subreporte y prognosis
│
├── cuadernos/
│   └── 01_analisis_exploratorio_y_modelado_osha.ipynb        # Cuaderno de análisis reproducible
│
├── src/
│   ├── inspeccionar_datos.py                                 # Validación de esquemas y completitud
│   ├── analisis_tasas_accidentabilidad.py                    # Motor de cálculo de tasas oficiales
│   ├── preparar_dataset_nlp.py                               # Limpieza, etiquetado y partición estratificada
│   ├── entrenar_modelos_nlp.py                               # Entrenamiento y benchmark de modelos NLP
│   ├── investigaciones_avanzadas.py                          # Análisis de fatiga, pirámide y prognosis
│   └── modelos/
│       ├── modelo_tfidf_pipeline.joblib                      # Pipeline entrenado TF-IDF + Clasificador
│       ├── clasificador_embeddings.joblib                    # Clasificador sobre embeddings de 384d
│       └── modelo_prognosis_severidad.joblib                 # Modelo de pronóstico temprano de severidad
│
├── app/
│   ├── app.py                                                # Plataforma continua Streamlit (5 capítulos)
│   └── pages/
│       └── 1_Ficha_Tecnica.py                                # Ficha técnica y arquitectura de ciencia de datos
│
├── informe/
│   └── index.html                                            # Informe técnico interactivo
│
├── requirements.txt                                          # Dependencias del proyecto
├── DOCUMENTACION.md                                          # Memoria técnica completa y formal
└── README.md                                                 # Guía de inicio rápido y arquitectura
```

---

## 3. Resultados experimentales

### A. Indicadores nacionales agregados (1.06 billones de horas trabajadas)

| Indicador | Valor ponderado | Interpretación técnica |
|---|---|---|
| **Centros de trabajo analizados** | **352,500** | Establecimientos con al menos 10,000 horas anuales |
| **Horas hombre trabajadas** | **1,064,739,946,443** | Base empírica de exposición laboral (1.06 B) |
| **Casos registrables evaluados** | **688,649** | Incidentes individuales conforme al estándar 29 CFR 1904 |
| **TIR nacional ponderado** | **0.278** / 200k hrs | Frecuencia de casos registrables totales |
| **DART nacional ponderado** | **0.178** / 200k hrs | El 65.9% de incidentes implicó baja médica o restricción |
| **Sector de mayor severidad** | **Almacenamiento (5.04 DART)** | Alta incidencia de interacción peatón-montacargas |

### B. Rendimiento comparativo de modelos de PLN (Muestra ciega: 3,000 casos independientes)

| Métrica | TF-IDF + Regresión Logística | Sentence Transformers (`all-MiniLM-L6-v2`) |
|---|---|---|
| **Exactitud global (Accuracy)** | **91.97%** | **87.97%** |
| **F1-Score Macro** | **91.96%** | **87.95%** |
| **F1-Score Ponderado** | **91.96%** | **87.95%** |
| **Precisión Macro** | **91.97%** | **87.96%** |
| **Sensibilidad Macro (Recall)** | **91.97%** | **87.97%** |
| **Tiempo de inferencia (3,000 casos)** | **0.470 s** (0.15 ms / caso en CPU) | **28.425 s** (9.47 ms / caso en CPU) |
| **Concordancia entre modelos** | **90.1% (2,704 casos)** | 296 discrepancias en eventos compuestos |
| **Uso óptimo** | Procesamiento masivo en lote (batch) en CPU | Búsqueda semántica y redacciones informales |

---

## 4. Estructura de la aplicación interactiva (Storytelling en 5 capítulos)

La interfaz en Streamlit (`app/app.py`) está diseñada con tipografía Inter, jerarquía visual calibrada (máximo peso tipográfico 600) y navegación continua fluida:

1. **Capítulo 1 - Diagnóstico macro y universo de exposición:** 4 KPIs principales, donut de severidad clínica real, barra de mecanismos causales y arquitectura del pipeline ETL de 606 MB con datos descargables.
2. **Capítulo 2 - Paradoja sectorial (Frecuencia vs Severidad):** Comparativa de tasas TIR vs DART en sectores NAICS reales (liderado por Almacenamiento con 5.04), tiempos de retorno al trabajo por ocupación SOC y cadencia cronológica semanal/estacional.
3. **Capítulo 3 - Evidencia empírica y contrastación de teorías clásicas:**
   - *Curva de fatiga horaria (543,867 casos):* Desmitifica el mito del fin de turno: 42.5% de accidentes ocurre en las horas 1 a 4.
   - *Pirámide de Bird frente a 29 CFR 1904:* Estructura empírica observada (1 muerte : 1,224 DAFW : 1,035 DJTR : 1,166 ambulatorios).
   - *Auditoría de subreporte:* Ratio de 1.8x grave/leve en microempresas frente a 0.85x en corporaciones.
   - *Riesgos emergentes:* Plataformas elevadoras (PEMP: 18.5 días de baja) y violencia laboral.
4. **Capítulo 4 - Pronóstico temprano y modelos predictivos:**
   - Pronóstico de baja >30 días (ROC-AUC 0.7192, Sensibilidad 69.3%) con doble panel de términos patológicos inductores vs atenuantes.
   - Benchmark experimental exhaustivo con matrices de confusión normalizadas.
   - Inspector de inferencia sobre 3,000 casos reales de prueba con filtros por discrepancia y descargas CSV.
5. **Capítulo 5 - Triaje en tiempo real y jerarquía de controles:**
   - Simulador interactivo con medición de latencia en milisegundos.
   - Explicabilidad algorítmica (XAI) con descomposición local de pesos TF-IDF.
   - Detección de palabras patológicas y alertas clínicas preventivas.
   - Prescripción técnica de ingeniería, administrativa (LOTO) y EPP bajo el estándar OSHA.

---

## 5. Instrucciones de ejecución

### Instalación de dependencias
```bash
pip install -r requirements.txt
```

### Ejecución de los scripts de análisis y modelado
```bash
# 1. Validación de esquemas y tipos de datos:
python3 src/inspeccionar_datos.py

# 2. Cálculo de tasas de accidentabilidad por sector y puesto:
python3 src/analisis_tasas_accidentabilidad.py

# 3. Preparación del corpus y partición estratificada:
python3 src/preparar_dataset_nlp.py

# 4. Entrenamiento y benchmark de modelos NLP:
python3 src/entrenar_modelos_nlp.py

# 5. Modelado de fatiga, pirámide de Bird y severidad:
python3 src/investigaciones_avanzadas.py
```

### Ejecución de la aplicación interactiva
```bash
streamlit run app/app.py
```
Disponible de forma local en `http://localhost:8501`.

---

## 6. Despliegue en la nube

1. **Streamlit Community Cloud:**
   - Conectar el repositorio de GitHub en [share.streamlit.io](https://share.streamlit.io).
   - Establecer el archivo principal en `app/app.py`.
2. **GitHub Pages:**
   - En el repositorio de GitHub, acceder a **Settings > Pages**.
   - Configurar la rama `main` y el directorio `/informe`.
   - El informe técnico queda publicado automáticamente con soporte interactivo mediante Plotly.js.
