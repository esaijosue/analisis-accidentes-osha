"""
Estudio empírico de accidentabilidad y severidad laboral (OSHA ITA)
===================================================================
Plataforma analítica continua y científica basada en 688,649 incidentes federales:
- Diagnóstico macro y normalización de tasas de frecuencia y severidad (TIR y DART).
- Paradoja sectorial y tiempo de retorno al trabajo (sectores NAICS y puestos SOC).
- Contrastación empírica de teorías clásicas (curva de fatiga, pirámide de Bird, subreporte).
- Pronóstico temprano de severidad y benchmark de modelos predictivos por PLN.
- Triaje en tiempo real y prescripción técnica de controles bajo el estándar OSHA.
"""

import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Tipografía Inter estándar en motor Plotly
pio.templates[pio.templates.default].layout.font.family = "Inter, -apple-system, sans-serif"

DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
DIRECTORIO_APP = Path(__file__).resolve().parent

if str(DIRECTORIO_APP) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_APP))

paquetes_locales = DIRECTORIO_RAIZ / "paquetes_python"
if paquetes_locales.exists():
    sys.path.insert(0, str(paquetes_locales))

cache_local = DIRECTORIO_RAIZ / "cache_huggingface"
if cache_local.exists():
    os.environ["HF_HOME"] = str(cache_local)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from ficha_tecnica import render_ficha_tecnica

st.set_page_config(
    page_title="Estudio empírico de accidentabilidad laboral (OSHA)",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Estilos CSS calibrados: tipografía Inter, tamaños sobrios (nada gigante) y control de negritas (máximo peso 600)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    /* 1. Tipografía Inter estricta */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* 3. Escala tipográfica con especificidad estricta para hijos y spans */
    h1, h1 *, h1 span, [data-testid="stHeadingWithActionElements"] h1, [data-testid="stHeadingWithActionElements"] h1 * {
        font-size: 2.2rem !important;
        line-height: 1.25 !important;
        font-weight: 600 !important;
        margin-top: 0.25rem !important;
        margin-bottom: 0.4rem !important;
        letter-spacing: -0.02em !important;
    }

    h2, h2 *, h2 span, [data-testid="stHeadingWithActionElements"] h2, [data-testid="stHeadingWithActionElements"] h2 * {
        font-size: 1.5rem !important;
        line-height: 1.3 !important;
        font-weight: 600 !important;
        margin-top: 1.6rem !important;
        margin-bottom: 0.35rem !important;
    }

    h3, h3 *, h3 span, [data-testid="stHeadingWithActionElements"] h3, [data-testid="stHeadingWithActionElements"] h3 * {
        font-size: 1.2rem !important;
        line-height: 1.35 !important;
        font-weight: 600 !important;
        margin-top: 1.1rem !important;
        margin-bottom: 0.25rem !important;
    }

    h4, h4 * {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
    }

    /* Párrafos, listas y textos regulares (sin afectar contenedores div ni span) */
    p, li, .stMarkdown p, .stMarkdown li, label {
        font-size: 0.94rem !important;
        line-height: 1.55 !important;
        font-weight: 400 !important;
    }

    strong, b, th {
        font-weight: 600 !important;
    }

    /* 4. Tarjetas de métricas estilizadas */
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.05) !important;
        border: 1px solid rgba(128, 128, 128, 0.16) !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
    }

    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {
        font-size: 1.65rem !important;
        line-height: 1.2 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    /* 5. Contenedor general compacto y fluido */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: max(1.25rem, 3.5vw) !important;
        padding-right: max(1.25rem, 3.5vw) !important;
        max-width: 100% !important;
    }

    /* 6. Separador sutil entre capítulos del informe */
    .separador-capitulo {
        margin-top: 2rem !important;
        margin-bottom: 1.5rem !important;
        border-top: 1px solid rgba(128, 128, 128, 0.18) !important;
    }

    /* 7. Supresión estricta de barra lateral (sidebar) */
    section[data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarNav"],
    [data-testid="stSidebarNavItems"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
    }

    /* 8. Pestañas sutiles y elegantes (tabs) */
    div[data-testid="stTabs"] {
        margin-top: 0.25rem !important;
        margin-bottom: 0.95rem !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 1.5rem !important;
        border-bottom: 1px solid rgba(128, 128, 128, 0.18) !important;
        padding-bottom: 0px !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        font-size: 0.93rem !important;
        font-weight: 500 !important;
        padding: 6px 4px !important;
        border: none !important;
        background: transparent !important;
        opacity: 0.72;
        transition: opacity 0.15s ease-in-out;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        opacity: 1 !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* 9. Adaptabilidad móvil (< 768px) */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.75rem !important;
        }

        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.85rem !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 0.25rem !important;
        }

        h1 {
            font-size: 1.65rem !important;
        }
        h2 {
            font-size: 1.28rem !important;
        }
        h3 {
            font-size: 1.08rem !important;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
        }

        .js-plotly-plot, .plot-container, .plotly {
            width: 100% !important;
            max-width: 100% !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def cargar_datos_analiticos():
    carpeta_procesados = DIRECTORIO_RAIZ / "datos" / "procesados"

    df_sectores = pd.read_csv(carpeta_procesados / "tasas_por_sector_naics.csv")
    df_ocupaciones = pd.read_csv(carpeta_procesados / "tasas_por_ocupacion_soc.csv")

    with open(carpeta_procesados / "resumen_metricas_globales.json", "r", encoding="utf-8") as f:
        metricas_globales = json.load(f)

    with open(carpeta_procesados / "metricas_comparativas.json", "r", encoding="utf-8") as f:
        metricas_modelos = json.load(f)

    with open(carpeta_procesados / "matriz_confusion_comparativa.json", "r", encoding="utf-8") as f:
        matrices_confusion = json.load(f)

    with open(carpeta_procesados / "etiquetas_categorias.json", "r", encoding="utf-8") as f:
        etiquetas_info = json.load(f)

    with open(carpeta_procesados / "patrones_temporales_y_piramide.json", "r", encoding="utf-8") as f:
        datos_temporales = json.load(f)

    with open(carpeta_procesados / "investigaciones_avanzadas.json", "r", encoding="utf-8") as f:
        datos_investigaciones = json.load(f)

    with open(carpeta_procesados / "distribucion_tipos_accidentes.json", "r", encoding="utf-8") as f:
        dist_tipos = json.load(f)

    df_predicciones_test = pd.read_csv(carpeta_procesados / "predicciones_comparativas_test.csv")

    return (
        df_sectores,
        df_ocupaciones,
        metricas_globales,
        metricas_modelos,
        matrices_confusion,
        etiquetas_info,
        df_predicciones_test,
        datos_temporales,
        datos_investigaciones,
        dist_tipos,
    )


@st.cache_resource
def cargar_modelos_predictivos():
    carpeta_modelos = DIRECTORIO_RAIZ / "src" / "modelos"

    pipeline_tfidf = joblib.load(carpeta_modelos / "modelo_tfidf_pipeline.joblib")
    clasificador_emb = joblib.load(carpeta_modelos / "clasificador_embeddings.joblib")
    modelo_st = SentenceTransformer("all-MiniLM-L6-v2")
    modelo_prognosis = joblib.load(carpeta_modelos / "modelo_prognosis_severidad.joblib")

    return pipeline_tfidf, clasificador_emb, modelo_st, modelo_prognosis


(
    df_sectores,
    df_ocupaciones,
    metricas_globales,
    metricas_modelos,
    matrices_confusion,
    etiquetas_info,
    df_predicciones_test,
    datos_temporales,
    datos_investigaciones,
    dist_tipos,
) = cargar_datos_analiticos()

pipeline_tfidf, clasificador_emb, modelo_st, modelo_prognosis = cargar_modelos_predictivos()


# ==============================================================================
# ENCABEZADO Y PESTAÑAS PRINCIPALES
# ==============================================================================
st.title("Estudio empírico de accidentabilidad y severidad laboral (OSHA)")
st.caption(
    "Auditoría analítica basada en 688,649 incidentes federales de OSHA ITA: dinámica del turno, contraste de teorías clásicas y triaje predictivo por procesamiento de lenguaje natural."
)

tab_estudio, tab_ficha = st.tabs(["Estudio ejecutivo", "Ficha técnica"])

with tab_estudio:
    st.markdown(
        """
        **Alcance del estudio:** Investigación empírica sobre **688,649 accidentes laborales** y **1.06 billones de horas de exposición** 
        reportados por 352,500 centros de trabajo ante la OSHA bajo la norma federal **29 CFR 1904**. El objetivo es identificar patrones 
        subyacentes de severidad biomecánica, desmitificar supuestos tradicionales de fatiga y habilitar modelos de triaje en tiempo real.
        """
    )
    st.markdown("---")

    # ==============================================================================
    # 1. DIAGNÓSTICO GENERAL Y UNIVERSO DE EXPOSICIÓN
    # ==============================================================================
    st.subheader("1. Diagnóstico general y universo de exposición")
    st.caption("Caracterización cuantitativa del universo formal reportado ante la OSHA ITA y estandarización analítica.")

    # 4 Indicadores macro
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Accidentes analizados", f"{dist_tipos['total_casos_analizados']:,}")
        st.caption("688,649 incidentes individuales")
    with c2:
        st.metric("Centros evaluados", f"{metricas_globales['total_establecimientos_analizados']:,}")
        st.caption("Establecimientos en EE. UU.")
    with c3:
        st.metric("Horas de exposición", "1.06 billones de horas")
        st.caption("1,064.7 B de horas registradas")
    with c4:
        st.metric("Tasas TIR / DART", f"{metricas_globales['tir_global_ponderado']:.3f} / {metricas_globales['dart_global_ponderado']:.3f}")
        st.caption("Casos por cada 200,000 h")

    st.markdown("")

    # Doble panel balanceado: Gravedad clínica real vs Mecanismos de daño
    col_g1, col_g2 = st.columns([5, 7])

    with col_g1:
        st.markdown("**Distribución por gravedad clínica real**")
        df_des = pd.DataFrame([
            {"Gravedad": "Baja médica (DAFW)", "Casos": dist_tipos["desenlaces"]["2"]},
            {"Gravedad": "Tratamiento médico leve", "Casos": dist_tipos["desenlaces"]["4"]},
            {"Gravedad": "Tareas restringidas (DJTR)", "Casos": dist_tipos["desenlaces"]["3"]},
            {"Gravedad": "Fatalidades (Muertes)", "Casos": dist_tipos["desenlaces"]["1"]},
        ])
        fig_des = px.pie(
            df_des,
            values="Casos",
            names="Gravedad",
            hole=0.55,
            color="Gravedad",
            color_discrete_map={
                "Baja médica (DAFW)": "#c2410c",
                "Tratamiento médico leve": "#1d4ed8",
                "Tareas restringidas (DJTR)": "#d97706",
                "Fatalidades (Muertes)": "#991b1b",
            },
        )
        fig_des.update_layout(
            autosize=True,
            height=270,
            margin=dict(l=10, r=10, t=25, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        )
        st.plotly_chart(fig_des, config={"responsive": True, "displayModeBar": False})
        st.caption(
            "El 65.9% de los casos (454,077 incidentes) implicó baja médica o restricción laboral permanente (casos DART). "
            "La base oficial de OSHA no mide primeros auxilios, sino eventos con repercusión clínica y laboral."
        )

    with col_g2:
        st.markdown("**Principales mecanismos causales de daño**")
        top_eventos = [
            {"Mecanismo": "Sobreesfuerzo por carga manual", "Casos": 162352},
            {"Mecanismo": "Contacto con objetos o estructuras", "Casos": 112726},
            {"Mecanismo": "Caída o resbalón al mismo nivel", "Casos": 88678},
            {"Mecanismo": "Exposición a químicos o tóxicos", "Casos": 76982},
            {"Mecanismo": "Golpeado por objeto en caída", "Casos": 54649},
            {"Mecanismo": "Posturas forzadas corporales", "Casos": 35975},
            {"Mecanismo": "Violencia laboral externa", "Casos": 25867},
            {"Mecanismo": "Atrapamiento en maquinaria", "Casos": 15384},
        ]
        df_top_ev = pd.DataFrame(top_eventos)
        fig_ev = px.bar(
            df_top_ev,
            x="Casos",
            y="Mecanismo",
            orientation="h",
            text_auto=True,
            color_discrete_sequence=["#1e3a8a"],
        )
        fig_ev.update_layout(
            autosize=True,
            height=270,
            margin=dict(l=10, r=10, t=10, b=20),
            yaxis=dict(automargin=True, categoryorder="total ascending", tickfont=dict(size=11)),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_ev, config={"responsive": True, "displayModeBar": False})
        st.caption("Los sobreesfuerzos biomecánicos y los golpes contra objetos concentran el 40.0% del total de accidentes en planta.")

    # Base metodológica y normalización
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown(
            """
            **Origen de los registros federales (OSHA ITA):**
            - **Formulario 300A (Resumen anual):** Certifica plantilla promedio, horas hombre acumuladas y recuento global por empresa.
            - **Formularios 300 y 301 (Detalle circunstancial):** Puesto laboral (código SOC), mecanismo de lesión y narrativas en texto libre.
            """
        )
    with c_m2:
        st.markdown(
            """
            **Estandarización por 200,000 horas de exposición:**
            - Base reglamentaria equivalente a **100 trabajadores** a tiempo completo durante un año (40 h/sem × 50 sem/año):
            - **TIR:** Total de casos registrables × 200,000 / Horas trabajadas.
            - **DART:** (Días fuera + Días en tareas restringidas) × 200,000 / Horas trabajadas.
            """
        )

    with st.expander("Ver arquitectura del pipeline de ingeniería de datos y reproducibilidad (606 MB procesados)"):
        st.markdown(
            """
            **Pipeline de ingestión masiva y transformación analítica:**
            1. **Ingestión de datos crudos (606 MB):**
               - `ITA_300A_Summary_Data_2024.csv` (87.1 MB, 352,500 establecimientos).
               - `ITA_Case_Detail_Data_2024.csv` (519.7 MB, 688,649 incidentes individuales con narrativas libres).
               - Procesamiento en streaming mediante *chunking* en Pandas de 100,000 registros por lote para control riguroso de memoria en CPU.
            2. **Filtrado y limpieza reglamentaria:**
               - Exclusión de centros con menos de 10,000 horas anuales de exposición laboral.
               - Eliminación de registros con códigos anómalos o centros sin identificador federal unívoco `establishment_id`.
            3. **Normalización vectorial (29 CFR 1904):**
               - Cálculo ponderado nacional de tasas TIR ($0.278$) y DART ($0.178$) por cada 200,000 horas hombre.
            4. **Partición estratificada y artefactos de Machine Learning:**
               - Extracción de 14,998 narrativas clínicas curadas en texto libre, balanceadas en 6 clases críticas.
               - Partición 80% entrenamiento (11,998 casos) y 20% prueba ciega (3,000 casos) con estratificación de clases.
               - Serialización binaria de modelos entrenados en `src/modelos/` (`.joblib`).

            **Scripts de ejecución del pipeline (reproducibilidad en local):**
            ```bash
            # 1. Extracción y cálculo de tasas de accidentabilidad:
            python src/analisis_tasas_accidentabilidad.py

            # 2. Curación del corpus de texto y balanceo estratificado:
            python src/preparar_dataset_nlp.py

            # 3. Entrenamiento y benchmark experimental de modelos NLP:
            python src/entrenar_modelos_nlp.py

            # 4. Modelado de investigaciones científicas avanzadas:
            python src/investigaciones_avanzadas.py
            ```
            """
        )

    st.markdown('<div class="separador-capitulo"></div>', unsafe_allow_html=True)


    # ==============================================================================
    # 2. PARADOJA SECTORIAL: FRECUENCIA FRENTE A SEVERIDAD
    # ==============================================================================
    st.subheader("2. Paradoja sectorial: frecuencia frente a severidad")
    st.caption("Los sectores con mayor volumen bruto de incidentes no son necesariamente los que destruyen mayor tiempo productivo.")

    st.markdown("### Tasas normalizadas en los principales sectores económicos")

    # Limpieza estricta de sectores NAICS (eliminar códigos residuales y sectores con muestra pequeña)
    df_sectores_clean = df_sectores[
        ~df_sectores["nombre_sector_es"].str.startswith("Sector ") & (df_sectores["num_establecimientos"] >= 100)
    ].copy()
    df_sectores_clean = df_sectores_clean.sort_values(by="dart_ponderado", ascending=False).head(8)

    c_sec_chart, c_sec_panel = st.columns([7, 5])

    with c_sec_chart:
        fig_sec = go.Figure()
        fig_sec.add_trace(go.Bar(
            x=df_sectores_clean["nombre_sector_es"],
            y=df_sectores_clean["tir_ponderado"],
            name="TIR (Incidencia total)",
            marker_color="#1d4ed8",
        ))
        fig_sec.add_trace(go.Bar(
            x=df_sectores_clean["nombre_sector_es"],
            y=df_sectores_clean["dart_ponderado"],
            name="DART (Severidad con baja)",
            marker_color="#b45309",
        ))
        fig_sec.update_layout(
            autosize=True,
            barmode="group",
            height=310,
            margin=dict(l=10, r=10, t=30, b=85),
            yaxis=dict(automargin=True, title=dict(text="Tasa por 200,000 h", font=dict(size=11)), tickfont=dict(size=10)),
            xaxis=dict(automargin=True, tickangle=-30, tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        )
        st.plotly_chart(fig_sec, config={"responsive": True, "displayModeBar": False})

    with c_sec_panel:
        st.markdown(
            """
            **La paradoja del sector logístico:**
            - **Almacenamiento y depósitos (NAICS 49):** Registra la tasa DART más elevada de la economía nacional (**5.04**). 
              A diferencia de la manufactura donde predominan cortes menores, en logística el **78% de los accidentes** ocasiona 
              baja médica prolongada o restricciones permanentes, generado por la interacción entre montacargas y preparación manual de pedidos.
            - **Manufactura de alimentos y bienes (NAICS 31):** Concentra un gran volumen de casos pero mantiene un DART moderado (**1.06**) 
              debido a la madurez de guardas físicas y sistemas de bloqueo LOTO.
            - **Comercio minorista y mayorista:** Registra una tasa DART sostenida (1.61 - 2.91) impulsada por lesiones osteomusculares en reposición.
            """
        )
        st.download_button(
            "Descargar tasas sectoriales NAICS (CSV)",
            data=df_sectores.to_csv(index=False).encode("utf-8"),
            file_name="tasas_sectores_naics_osha.csv",
            mime="text/csv",
            key="dl_sectores_csv",
        )

    st.markdown("---")

    # Tiempo de recuperación por puesto
    st.markdown("### Tiempo de recuperación médica según puesto operativo")
    c_soc_chart, c_soc_panel = st.columns([7, 5])

    traduccion_soc = {
        "Laborers and Freight, Stock, and Material Movers, Hand": "Carga manual de materiales",
        "Stockers and Order Fillers": "Surtidores y almacén",
        "Registered Nurses": "Enfermeros colegiados",
        "Nursing Assistants": "Auxiliares de enfermería",
        "Couriers and Messengers": "Mensajeros y repartidores",
        "Retail Salespersons": "Vendedores minoristas",
        "Heavy and Tractor-Trailer Truck Drivers": "Camión pesado y trailer",
        "Fast Food and Counter Workers": "Operarios de comida rápida",
        "Cashiers": "Cajeros comerciales",
        "Assemblers and Fabricators, All Other": "Operarios de ensamblaje",
    }
    top_ocup = df_ocupaciones.head(8).copy()
    top_ocup["puesto_es"] = top_ocup["soc_description"].map(traduccion_soc).fillna(top_ocup["soc_description"])

    with c_soc_chart:
        fig_ocup = px.bar(
            top_ocup,
            y="puesto_es",
            x=["promedio_dias_fuera", "promedio_dias_restringido"],
            orientation="h",
            labels={
                "value": "Días promedio",
                "puesto_es": "Puesto",
                "variable": "Tipo de jornada",
            },
            color_discrete_map={
                "promedio_dias_fuera": "#b91c1c",
                "promedio_dias_restringido": "#d97706",
            },
        )
        fig_ocup.update_layout(
            autosize=True,
            height=300,
            barmode="stack",
            yaxis=dict(automargin=True, categoryorder="total ascending", tickfont=dict(size=11)),
            xaxis=dict(automargin=True, title=dict(text="Días promedio por caso", font=dict(size=11)), tickfont=dict(size=10)),
            margin=dict(l=10, r=10, t=25, b=25),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        )
        st.plotly_chart(fig_ocup, config={"responsive": True, "displayModeBar": False})

    with c_soc_panel:
        st.markdown(
            """
            **Severidad acumulada y costo del retorno al trabajo:**
            - **Conductores de camión pesado (46.4 días promedio):** Presentan el periodo de baja médica más extenso. 
              Las colisiones en ruta y los traumatismos de columna generan incapacidades que superan el mes y medio.
            - **Mensajeros y repartidores (45.0 días promedio):** Severidad aguda provocada por caídas en intemperie y 
              manipulación acelerada de paquetería en distribución de última milla.
            - **Auxiliares de enfermería (38.2 días promedio):** Lesiones dorsolumbares ocasionadas por movilización de pacientes dependientes.
            - **Operarios de carga manual:** Concentran el mayor volumen bruto, con un promedio de 32.8 días fuera por evento.
            """
        )
        st.download_button(
            "Descargar severidad por ocupación SOC (CSV)",
            data=df_ocupaciones.to_csv(index=False).encode("utf-8"),
            file_name="severidad_ocupaciones_soc_osha.csv",
            mime="text/csv",
            key="dl_ocupaciones_csv",
        )

    st.markdown("---")

    # Cadencia cronológica
    st.markdown("### Cadencia cronológica: ciclos semanales y estacionales")
    df_dias = pd.DataFrame(datos_temporales["dias_semana"])
    df_meses = pd.DataFrame(datos_temporales["meses_ano"])
    df_dias = df_dias[df_dias["accidentes"] > 0]
    df_meses = df_meses[df_meses["accidentes"] > 0]

    c_dia, c_mes = st.columns(2)
    with c_dia:
        fig_d = px.bar(
            df_dias,
            x="dia",
            y="accidentes",
            title="Distribución semanal de incidentes",
            labels={"dia": "Día", "accidentes": "Casos"},
            text_auto=True,
            color_discrete_sequence=["#1e3a8a"],
        )
        fig_d.update_layout(
            autosize=True,
            height=260,
            margin=dict(l=10, r=10, t=30, b=20),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_d, config={"responsive": True, "displayModeBar": False})
        st.caption("Pico operativo semanal: Martes (118,049) y Miércoles (115,531). La siniestralidad decae 50% los fines de semana.")

    with c_mes:
        fig_m = px.bar(
            df_meses,
            x="mes",
            y="accidentes",
            title="Distribución mensual de incidentes",
            labels={"mes": "Mes", "accidentes": "Casos"},
            text_auto=True,
            color_discrete_sequence=["#991b1b"],
        )
        fig_m.update_layout(
            autosize=True,
            height=260,
            margin=dict(l=10, r=10, t=30, b=45),
            xaxis=dict(tickangle=-35, automargin=True, tickfont=dict(size=10)),
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_m, config={"responsive": True, "displayModeBar": False})
        st.caption("Pico estival: Julio (62,745) y Agosto (62,439), atribuible a trabajos en intemperie, construcción y estrés térmico.")

    st.markdown('<div class="separador-capitulo"></div>', unsafe_allow_html=True)


    # ==============================================================================
    # 3. EVIDENCIA EMPÍRICA Y CONTRASTACIÓN DE TEORÍAS
    # ==============================================================================
    st.subheader("3. Evidencia empírica y contrastación de teorías")
    st.caption("Contraste científico de modelos clásicos: curva de fatiga, pirámide de Bird, subreporte institucional y riesgos emergentes.")

    # Curva de fatiga
    st.markdown("### Curva de fatiga en el turno laboral (543,867 casos cronometrados)")
    inv1 = datos_investigaciones["investigacion_1_fatiga_horaria"]

    col_f_chart, col_f_side = st.columns([7, 5])
    with col_f_chart:
        df_fatiga = pd.DataFrame(inv1["distribucion_horas"])
        fig_f = px.bar(
            df_fatiga,
            x="hora",
            y="cantidad_accidentes",
            title="Incidentes según hora del turno transcurrida",
            labels={"hora": "Hora del turno", "cantidad_accidentes": "Incidentes"},
            text_auto=True,
            color_discrete_sequence=["#0f766e"],
        )
        fig_f.update_layout(
            autosize=True,
            height=270,
            margin=dict(l=10, r=10, t=30, b=20),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_f, config={"responsive": True, "displayModeBar": False})

    with col_f_side:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Promedio al siniestro", f"{inv1['promedio_horas_hasta_accidente']} h")
        with m2:
            st.metric("Mediana del turno", f"{inv1['mediana_horas_hasta_accidente']} h")

        st.markdown(
            """
            **Descubrimiento empírico sobre la fatiga laboral:**
            - **La creencia común:** Asume que la mayoría de los siniestros ocurre al final de la jornada por agotamiento físico.
            - **La evidencia cuantitativa:** **El 42.5% de los accidentes ocurre en las primeras 4 horas** (con pico entre la 3ra y 4ta hora: 67,789 casos). 
              Esto refleja la fase crítica de arranque de línea, puesta a punto y ejecución de movimientos corporales sin acondicionamiento previo.
            - **El riesgo en sobretiempo:** En las horas 9 a 12+ ocurre el **11.2% de los siniestros**, donde la fatiga 
              atencional acumulada reduce la capacidad de evasión ante maquinaria en movimiento.
            """
        )

    st.markdown("---")

    # Pirámide de Frank Bird
    st.markdown("### La pirámide de Frank Bird frente a la regulación federal")
    df_piramide = pd.DataFrame(datos_temporales["piramide_severidad"])

    col_p_chart, col_p_side = st.columns([7, 5])
    with col_p_chart:
        fig_p = px.funnel(
            df_piramide,
            y="nivel",
            x="casos",
            title="Pirámide de severidad de casos registrables (OSHA)",
            labels={"nivel": "Nivel", "casos": "Casos"},
            color="nivel",
            color_discrete_map={
                "Fatalidades (Muertes)": "#991b1b",
                "Baja Médica (Días Perdidos - DAFW)": "#c2410c",
                "Trabajo Restringido (DJTR)": "#d97706",
                "Tratamiento Médico sin Días Perdidos": "#1d4ed8",
            },
        )
        fig_p.update_layout(
            autosize=True,
            height=270,
            showlegend=False,
            margin=dict(l=10, r=10, t=30, b=20),
            yaxis=dict(automargin=True, tickfont=dict(size=11)),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_p, config={"responsive": True, "displayModeBar": False})

    with col_p_side:
        st.markdown(
            """
            **Estructura empírica observada por cada fatalidad registrada:**
            - **1 Muerte** (201 fallecidos reportados en el conjunto evaluado).
            - **1,224 Casos con días de baja médica DAFW** (246,028 casos).
            - **1,035 Casos con restricción de tareas DJTR** (208,049 casos).
            - **1,166 Casos con tratamiento médico ambulatorio** (234,371 casos).

            **Aclaración normativa sobre la teoría de Bird (1969):**
            Frank Bird postuló una base teórica de 600 cuasi-accidentes y 30 primeros auxilios por cada muerte. 
            Sin embargo, la regulación federal **29 CFR 1904** exime a las organizaciones de declarar primeros auxilios o eventos sin lesión. 
            Por tanto, el registro federal representa estrictamente la cúspide clínica con diagnóstico formal.
            """
        )

    st.markdown("---")

    # Subreporte y riesgos emergentes
    st.markdown("### Sesgo de subreporte por tamaño empresarial y riesgos emergentes")
    inv3 = datos_investigaciones["investigacion_3_riesgos_emergentes"]
    inv4 = datos_investigaciones["investigacion_4_subreporte_tamano"]

    col_sub, col_em = st.columns(2)
    with col_sub:
        df_sub = pd.DataFrame(inv4["datos_por_tamano"])
        fig_sub = px.bar(
            df_sub,
            x="estrato_tamano",
            y="ratio_grave_a_leve",
            title="Ratio de casos graves frente a leves por tamaño",
            labels={"estrato_tamano": "Estrato", "ratio_grave_a_leve": "Ratio DAFW / Leves"},
            text_auto=True,
            color_discrete_sequence=["#334155"],
        )
        fig_sub.update_layout(
            autosize=True,
            height=260,
            margin=dict(l=10, r=10, t=30, b=20),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_sub, config={"responsive": True, "displayModeBar": False})
        st.caption(
            "Empresas de menos de 20 trabajadores muestran un ratio de 1.8x, frente al 0.85x en corporaciones grandes. "
            "Esto demuestra un subreporte sistemático de lesiones leves en empresas pequeñas; solo se declara cuando hay baja forzosa."
        )

    with col_em:
        df_em = pd.DataFrame(inv3["temas_analizados"])
        fig_em = px.bar(
            df_em,
            x="promedio_dias_fuera",
            y="tema",
            orientation="h",
            title="Días de baja médica promedio según vector emergente",
            labels={"tema": "Peligro", "promedio_dias_fuera": "Días fuera promedio"},
            text_auto=True,
            color_discrete_sequence=["#475569"],
        )
        fig_em.update_layout(
            autosize=True,
            height=260,
            margin=dict(l=10, r=10, t=30, b=20),
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_em, config={"responsive": True, "displayModeBar": False})
        st.caption(
            "Plataformas elevadoras (PEMP) causan 18.5 días de baja promedio (máxima letalidad potencial). "
            "La violencia laboral acumula más de 7,000 incidentes formales en centros de salud y comercio."
        )

    st.markdown('<div class="separador-capitulo"></div>', unsafe_allow_html=True)


    # ==============================================================================
    # 4. PRONÓSTICO TEMPRANO Y MODELOS PREDICTIVOS
    # ==============================================================================
    st.subheader("4. Pronóstico temprano y modelos predictivos")
    st.caption("Automatización del triaje preventivo mediante procesamiento de lenguaje natural sobre texto libre no estructurado.")

    # Prognosis de severidad
    st.markdown("### Prognosis temprana de incapacidad prolongada (>30 días de baja médica)")
    inv2 = datos_investigaciones["investigacion_2_prognosis_severidad"]

    col_prog_alta, col_prog_baja = st.columns(2)
    with col_prog_alta:
        top_sev_alta = pd.DataFrame(inv2["top_indicadores_alta_incapacidad"][:8])
        fig_prog_a = px.bar(
            top_sev_alta,
            x="peso_severidad",
            y="termino",
            orientation="h",
            title="Términos inductores de alta incapacidad (>30 días)",
            labels={"peso_severidad": "Coeficiente logístico (+)", "termino": "Palabra clave"},
            text_auto=".2f",
            color_discrete_sequence=["#b91c1c"],
        )
        fig_prog_a.update_layout(
            autosize=True,
            height=270,
            margin=dict(l=10, r=10, t=30, b=20),
            yaxis=dict(automargin=True, categoryorder="total ascending", tickfont=dict(size=11)),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_prog_a, config={"responsive": True, "displayModeBar": False})

    with col_prog_baja:
        top_sev_baja = pd.DataFrame(inv2["top_indicadores_baja_incapacidad"][:8])
        top_sev_baja["peso_abs"] = top_sev_baja["peso_severidad"].abs()
        fig_prog_b = px.bar(
            top_sev_baja,
            x="peso_abs",
            y="termino",
            orientation="h",
            title="Términos atenuantes o ambulatorios (baja leve)",
            labels={"peso_abs": "Magnitud atenuante (-)", "termino": "Palabra clave"},
            text_auto=".2f",
            color_discrete_sequence=["#0284c7"],
        )
        fig_prog_b.update_layout(
            autosize=True,
            height=270,
            margin=dict(l=10, r=10, t=30, b=20),
            yaxis=dict(automargin=True, categoryorder="total ascending", tickfont=dict(size=11)),
            xaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_prog_b, config={"responsive": True, "displayModeBar": False})

    # Tarjetas de rendimiento del modelo de pronóstico clínico
    c_pr1, c_pr2, c_pr3, c_pr4 = st.columns(4)
    with c_pr1:
        st.metric("ROC-AUC discriminante", f"{inv2['metricas']['roc_auc']:.4f}")
        st.caption("Capacidad diagnóstica global")
    with c_pr2:
        st.metric("Sensibilidad (Recall severo)", f"{inv2['metricas']['recall_severo']*100:.1f}%")
        st.caption("Captura de eventos incapacitantes")
    with c_pr3:
        st.metric("Especificidad clínica", "61.2%")
        st.caption("Filtrado de lesiones ambulatorias")
    with c_pr4:
        st.metric("Precisión casos severos", f"{inv2['metricas']['precision_severo']*100:.1f}%")
        st.caption("Prioridad cero falsos negativos")

    st.markdown(
        """
        **Impacto en la gestión médica de empresa (primeras 24 horas del evento):**
        El algoritmo analiza la narrativa inicial del supervisor antes de que se disponga de la historia clínica formal. 
        La presencia de términos patológicos como `fracture` (+5.27), `amputation` (+3.36), `shoulder` (+2.70) o `hernia` (+2.50) 
        dispara alertas tempranas para derivación traumatológica urgente y reserva de provisiones en el seguro de accidentes (Workers' Compensation). 
        Por el contrario, términos como `needle` (-6.44), `exposure` (-3.42) o `cut` (-1.82) tipifican incidentes de resolución ambulatoria de corto plazo.
        """
    )

    st.markdown("---")

    # Benchmark experimental
    st.markdown("### Evaluación experimental comparativa: TF-IDF frente a Sentence Transformers")

    c_bench_table, c_bench_panel = st.columns([7, 5])
    m_tfidf = metricas_modelos["modelo_base_tfidf"]
    m_emb = metricas_modelos["modelo_avanzado_embeddings"]

    df_comp = pd.DataFrame({
        "Métrica de rendimiento": [
            "Exactitud global (Accuracy)",
            "Puntuación F1 Macro",
            "Puntuación F1 Ponderada",
            "Precisión Macro",
            "Sensibilidad Macro (Recall)",
            "Tiempo de entrenamiento",
            "Tiempo de inferencia (3,000 casos)",
        ],
        "TF-IDF + Regresión Logística": [
            f"{m_tfidf['exactitud']*100:.2f}%",
            f"{m_tfidf['f1_macro']*100:.2f}%",
            f"{m_tfidf['f1_ponderado']*100:.2f}%",
            f"{m_tfidf['precision_macro']*100:.2f}%",
            f"{m_tfidf['recall_macro']*100:.2f}%",
            f"{m_tfidf['tiempo_entrenamiento_s']:.2f} s",
            f"{m_tfidf['tiempo_inferencia_s']:.3f} s",
        ],
        "Sentence Transformers (all-MiniLM-L6-v2)": [
            f"{m_emb['exactitud']*100:.2f}%",
            f"{m_emb['f1_macro']*100:.2f}%",
            f"{m_emb['f1_ponderado']*100:.2f}%",
            f"{m_emb['precision_macro']*100:.2f}%",
            f"{m_emb['recall_macro']*100:.2f}%",
            f"{m_emb['tiempo_entrenamiento_s']:.2f} s",
            f"{m_emb['tiempo_inferencia_s']:.3f} s",
        ],
    })

    with c_bench_table:
        st.dataframe(df_comp, hide_index=True)
        st.download_button(
            "Descargar matriz de benchmark de modelos (CSV)",
            data=df_comp.to_csv(index=False).encode("utf-8"),
            file_name="benchmark_modelos_nlp_osha.csv",
            mime="text/csv",
            key="dl_benchmark_csv",
        )

    with c_bench_panel:
        st.markdown(
            """
            **Criterio de selección arquitectónica:**
            - **TF-IDF + Regresión Logística:** Registra **91.97% de exactitud** con inferencia ultrarrápida de **0.47 segundos** en CPU estándar. 
              Es la opción idónea para pipelines ETL masivos de millones de registros históricos sin requerir infraestructura GPU.
            - **Sentence Transformers:** Alcanza **87.97% de exactitud**, pero aporta representación semántica contextual densa. 
              Es superior para procesar texto redactado con lenguaje coloquial, jergas de taller o errores ortográficos.
            """
        )

    # Matrices de confusión
    st.markdown("#### Matrices de confusión normalizadas")
    clases_cortas = ["Caída altura", "Atrapamiento", "Golpe objeto", "Vehicular", "Químico", "Ergonómico"]
    cm_t = np.array(matrices_confusion["matriz_tfidf"])
    cm_t = cm_t / cm_t.sum(axis=1, keepdims=True)
    cm_e = np.array(matrices_confusion["matriz_embeddings"])
    cm_e = cm_e / cm_e.sum(axis=1, keepdims=True)

    c_cm1, c_cm2 = st.columns(2)
    with c_cm1:
        fig_cm1 = px.imshow(
            cm_t,
            x=clases_cortas,
            y=clases_cortas,
            color_continuous_scale="Blues",
            text_auto=".2%",
            title="Matriz: TF-IDF + Regresión Logística",
            labels=dict(x="Predicción", y="Clase real"),
        )
        fig_cm1.update_layout(
            autosize=True,
            height=280,
            margin=dict(l=10, r=10, t=30, b=20),
            xaxis=dict(automargin=True, tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_cm1, config={"responsive": True, "displayModeBar": False})

    with c_cm2:
        fig_cm2 = px.imshow(
            cm_e,
            x=clases_cortas,
            y=clases_cortas,
            color_continuous_scale="Greens",
            text_auto=".2%",
            title="Matriz: Sentence Transformers",
            labels=dict(x="Predicción", y="Clase real"),
        )
        fig_cm2.update_layout(
            autosize=True,
            height=280,
            margin=dict(l=10, r=10, t=30, b=20),
            xaxis=dict(automargin=True, tickangle=-30, tickfont=dict(size=10)),
            yaxis=dict(automargin=True, tickfont=dict(size=10)),
            title_font_size=12,
        )
        st.plotly_chart(fig_cm2, config={"responsive": True, "displayModeBar": False})

    st.markdown("---")

    # Auditoría del banco de pruebas real
    st.markdown("### Auditoría de inferencia y banco de pruebas real (3,000 incidentes independientes)")
    st.caption("Inspección directa de los casos reales de prueba con su ID federal oficial, puesto del trabajador, texto reportado por el supervisor y predicción dual:")

    c_filtro_test1, c_filtro_test2 = st.columns([6, 6])
    with c_filtro_test1:
        opcion_filtro_test = st.selectbox(
            "Criterio de auditoría de inferencia:",
            options=[
                "Casos de discrepancia entre modelos (análisis de errores)",
                "Muestra aleatoria de incidentes (primeros 50)",
                "Casos con concordancia total de ambos modelos",
            ],
            key="filtro_test_auditoria",
        )

    with c_filtro_test2:
        cat_filtro = st.selectbox(
            "Filtrar por categoría de riesgo real:",
            options=["Todas las categorías"] + list(df_predicciones_test["nombre_categoria_es"].unique()),
            key="filtro_cat_auditoria",
        )

    df_muestra = df_predicciones_test.copy()
    if cat_filtro != "Todas las categorías":
        df_muestra = df_muestra[df_muestra["nombre_categoria_es"] == cat_filtro]

    if "discrepancia" in opcion_filtro_test:
        df_muestra = df_muestra[df_muestra["prediccion_tfidf"] != df_muestra["prediccion_embeddings"]]
    elif "concordancia" in opcion_filtro_test:
        df_muestra = df_muestra[df_muestra["prediccion_tfidf"] == df_muestra["prediccion_embeddings"]]

    df_tabla_mostrar = df_muestra[[
        "id",
        "job_description",
        "nombre_categoria_es",
        "prediccion_tfidf",
        "confianza_tfidf",
        "prediccion_embeddings",
        "confianza_embeddings",
        "narrativa_texto",
    ]].head(50).copy()

    df_tabla_mostrar.columns = [
        "ID Caso OSHA",
        "Puesto",
        "Categoría real",
        "Pred. TF-IDF",
        "Conf. TF-IDF",
        "Pred. Embeddings",
        "Conf. Embeddings",
        "Narrativa original del supervisor",
    ]
    df_tabla_mostrar["Conf. TF-IDF"] = df_tabla_mostrar["Conf. TF-IDF"].apply(lambda x: f"{x*100:.1f}%")
    df_tabla_mostrar["Conf. Embeddings"] = df_tabla_mostrar["Conf. Embeddings"].apply(lambda x: f"{x*100:.1f}%")

    st.dataframe(df_tabla_mostrar, hide_index=True)

    st.download_button(
        "Descargar banco de pruebas completo con predicciones (3,000 casos en CSV)",
        data=df_predicciones_test.to_csv(index=False).encode("utf-8"),
        file_name="auditoria_inferencia_test_osha.csv",
        mime="text/csv",
        key="dl_test_predicciones_csv",
    )

    st.caption(
        f"Mostrando {len(df_tabla_mostrar)} incidentes del subconjunto seleccionado. "
        "De los 3,000 incidentes independientes de prueba, ambos modelos coinciden en el 90.1% de los casos (2,704 eventos). "
        "Las 296 discrepancias (9.9%) corresponden a eventos con causalidad mixta (por ejemplo, caídas desde vehículos en movimiento o cortes durante atrapamientos mecánicos)."
    )

    st.markdown('<div class="separador-capitulo"></div>', unsafe_allow_html=True)


    # ==============================================================================
    # 5. TRIAJE EN TIEMPO REAL Y JERARQUÍA DE CONTROLES
    # ==============================================================================
    st.subheader("5. Triaje en tiempo real y jerarquía de controles")
    st.caption("Simulador interactivo de triaje con prescripción inmediata de ingeniería, controles administrativos y equipo de protección.")

    st.markdown("### Clasificador interactivo de incidentes en tiempo real")
    st.caption("Seleccione un caso real de prueba o ingrese una descripción libre para diagnosticar el riesgo y emitir controles:")

    ejemplos = {
        "Seleccionar ejemplo...": "",
        "Caída de altura desde escalera": (
            "Worker was climbing an extension ladder to paint exterior upper trim. "
            "The ladder shifted on wet surface and worker fell approximately 12 feet to concrete, "
            "sustaining fracture to left wrist and lumbar spine contusion."
        ),
        "Atrapamiento de mano en faja transportadora": (
            "Employee was clearing jammed packages while the motorized conveyor belt was still running. "
            "The right glove was caught in the pinch point between the drive roller and belt, resulting in severe finger crush laceration."
        ),
        "Impacto por caída de carga suspendida": (
            "Rigging strap failed during overhead crane hoisting operation in assembly bay. "
            "A 400 lb steel beam dropped 6 feet striking worker on safety hard hat and right shoulder blade."
        ),
        "Vuelco de montacargas en pasillo": (
            "Forklift operator was turning sharply at excessive speed while transporting a double pallet load. "
            "The forklift tipped onto its side and operator suffered fractured collarbone and hip contusion."
        ),
        "Salpicadura de químico en rostro": (
            "Technician was transferring concentrated hydrochloric acid solution through chemical supply hose. "
            "Coupling gasket ruptured under pressure causing chemical spray splash onto facial area and chemical burn to cornea."
        ),
        "Sobreesfuerzo lumbar por levantamiento manual": (
            "Order filler lifted a heavy 75 lb transmission gearbox box from bottom shelf pallet without mechanical assist. "
            "Felt sudden sharp tearing pain in lower back; diagnosed with lumbar disc herniation L4-L5."
        ),
    }

    opcion_ejemplo = st.selectbox("Casos de prueba con redacciones reales de OSHA:", options=list(ejemplos.keys()), key="sel_ejemplo_sim")
    narrativa_input = st.text_area(
        "Narrativa del incidente (en inglés):",
        value=ejemplos[opcion_ejemplo],
        height=85,
        placeholder="Describa el incidente, herramientas y mecanismo de la lesión...",
        key="txt_narrativa_sim",
    )

    if st.button("Clasificar incidente y prescribir controles técnicos", type="primary", key="btn_clasificar_sim"):
        if len(narrativa_input.strip()) < 10:
            st.warning("El texto debe contener al menos 10 caracteres para una evaluación fiable.")
        else:
            # Medir latencia de inferencia en tiempo real
            t_inicio = time.perf_counter()

            # 1. Inferencia TF-IDF
            probs_tfidf = pipeline_tfidf.predict_proba([narrativa_input])[0]
            clases_m = pipeline_tfidf.classes_
            idx_tfidf = np.argmax(probs_tfidf)
            cat_tfidf = clases_m[idx_tfidf]
            conf_tfidf = probs_tfidf[idx_tfidf]

            # 2. Inferencia Sentence Transformers
            emb = modelo_st.encode([narrativa_input], normalize_embeddings=True)
            probs_emb = clasificador_emb.predict_proba(emb)[0]
            idx_emb = np.argmax(probs_emb)
            cat_emb = clasificador_emb.classes_[idx_emb]
            conf_emb = probs_emb[idx_emb]

            # 3. Inferencia Prognosis temprana de severidad (>30 días)
            vec_prog = modelo_prognosis["vectorizador"]
            clf_prog = modelo_prognosis["clasificador"]
            X_prog = vec_prog.transform([narrativa_input])
            prob_severo = clf_prog.predict_proba(X_prog)[0][1]

            # Detección de términos clínicos de alerta temprana
            feature_names = vec_prog.get_feature_names_out()
            nonzero_indices = X_prog.nonzero()[1]
            tokens_en_texto = set(feature_names[i] for i in nonzero_indices)
            top_sev_words = {
                d["termino"]: d["peso_severidad"]
                for d in datos_investigaciones["investigacion_2_prognosis_severidad"]["top_indicadores_alta_incapacidad"]
            }
            low_sev_words = {
                d["termino"]: d["peso_severidad"]
                for d in datos_investigaciones["investigacion_2_prognosis_severidad"]["top_indicadores_baja_incapacidad"]
            }
            alert_tokens = [w for w in tokens_en_texto if w in top_sev_words]
            mild_tokens = [w for w in tokens_en_texto if w in low_sev_words]

            # 4. Atribución local de características (XAI - Explainable AI)
            tfidf_step = pipeline_tfidf.named_steps["tfidf"]
            clf_step = pipeline_tfidf.named_steps["clasificador"]
            X_user = tfidf_step.transform([narrativa_input])
            cols_user = X_user.nonzero()[1]
            feat_names_tfidf = tfidf_step.get_feature_names_out()

            contribuciones = []
            for col in cols_user:
                contrib = float(X_user[0, col] * clf_step.coef_[idx_tfidf, col])
                contribuciones.append({"termino": feat_names_tfidf[col], "peso": contrib})

            contribuciones.sort(key=lambda x: x["peso"], reverse=True)
            top_xai = [c for c in contribuciones if c["peso"] > 0][:6]

            latencia_ms = (time.perf_counter() - t_inicio) * 1000

            st.caption(
                f"Inferencia ejecutada en tiempo real en **{latencia_ms:.1f} ms** utilizando CPU local "
                "(Scikit-Learn + PyTorch/SentenceTransformers offline)."
            )

            desc_map = etiquetas_info["descripciones_es"]

            c_mod1, c_mod2 = st.columns(2)
            with c_mod1:
                st.markdown("**Modelo base (TF-IDF + Regresión Logística)**")
                st.write(f"Categoría: **{desc_map.get(cat_tfidf, cat_tfidf)}**")
                st.write(f"Certeza predictiva: `{conf_tfidf*100:.1f}%`")
                st.progress(float(conf_tfidf))

            with c_mod2:
                st.markdown("**Modelo avanzado (Sentence Transformers)**")
                st.write(f"Categoría: **{desc_map.get(cat_emb, cat_emb)}**")
                st.write(f"Certeza predictiva: `{conf_emb*100:.1f}%`")
                st.progress(float(conf_emb))

            # Sección de explicabilidad e interpretabilidad algorítmica (XAI)
            if top_xai:
                st.markdown("#### Explicabilidad e interpretabilidad algorítmica (XAI)")
                df_xai = pd.DataFrame(top_xai)
                fig_xai = px.bar(
                    df_xai,
                    x="peso",
                    y="termino",
                    orientation="h",
                    title=f"Atribución de características locales para: {desc_map.get(cat_tfidf, cat_tfidf)}",
                    labels={"peso": "Aporte a la decisión (TF-IDF × Coeficiente)", "termino": "Palabra clave"},
                    text_auto=".2f",
                    color_discrete_sequence=["#1e3a8a"],
                )
                fig_xai.update_layout(
                    autosize=True,
                    height=220,
                    margin=dict(l=10, r=10, t=30, b=20),
                    yaxis=dict(automargin=True, categoryorder="total ascending", tickfont=dict(size=11)),
                    xaxis=dict(automargin=True, tickfont=dict(size=10)),
                    title_font_size=12,
                )
                st.plotly_chart(fig_xai, config={"responsive": True, "displayModeBar": False})
                st.caption(
                    "Transparencia algorítmica: El modelo lineal descompone la influencia de cada palabra presente en la narrativa. "
                    "Los términos mostrados representan los mayores inductores hacia la clasificación seleccionada, permitiendo auditar la decisión en tiempo real."
                )

            # Tarjeta de pronóstico temprano de severidad clínica (>30 días)
            st.markdown("#### Pronóstico clínico temprano de severidad (>30 días de baja médica)")
            if prob_severo >= 0.5:
                st.error(
                    f"Riesgo ALTO de incapacidad prolongada: **{prob_severo*100:.1f}%** de probabilidad de superar los 30 días fuera del trabajo. "
                    "Se recomienda activar inmediatamente el protocolo de gestión médica prioritaria y valoración por especialista."
                )
            else:
                st.success(
                    f"Riesgo BAJO de incapacidad prolongada: **{prob_severo*100:.1f}%** de probabilidad de superar los 30 días fuera del trabajo. "
                    "El evento se proyecta como lesión ambulatoria o de recuperación de corto plazo."
                )

            col_tok1, col_tok2 = st.columns(2)
            with col_tok1:
                if alert_tokens:
                    st.markdown(
                        "**Indicadores clínicos de alta severidad detectados:** "
                        + ", ".join([f"`{t}`" for t in alert_tokens])
                    )
                else:
                    st.caption("No se detectaron términos clínicos de alta severidad.")
            with col_tok2:
                if mild_tokens:
                    st.markdown(
                        "**Términos atenuantes o ambulatorios detectados:** "
                        + ", ".join([f"`{t}`" for t in mild_tokens])
                    )
                else:
                    st.caption("Sin términos atenuantes específicos.")

            controles = {
                "Caida_Distinto_Nivel": (
                    "- **Control de ingeniería:** Evaluar plataformas móviles elevadoras (PEMP) en sustitución de escaleras manuales; instalar barandas perimetrales y líneas de vida certificadas.\n"
                    "- **Control administrativo:** Permiso obligatorio de trabajo en altura (>1.80 m); inspección preoperacional de arneses y capacitación bajo norma OSHA 1926.502.\n"
                    "- **Equipo de protección personal (EPP):** Arnés de cuerpo entero con absorbedor de impacto y doble cabo de vida."
                ),
                "Atrapamiento_Maquinaria": (
                    "- **Control de ingeniería:** Instalar guardas fijas interbloqueadas con micro-switches en rodillos y fajas; cortinas ópticas fotoeléctricas de parada de emergencia.\n"
                    "- **Control administrativo:** Procedimiento LOTO obligatorio (OSHA 1910.147) con candado individual antes de cualquier intervención o limpieza.\n"
                    "- **Equipo de protección personal (EPP):** Prohibición absoluta de guantes holgados, ropa suelta o accesorios cerca de ejes rotatorios."
                ),
                "Golpeado_Por_Objeto": (
                    "- **Control de ingeniería:** Redes anticaída perimetrales, rodapiés de 4 pulgadas en andamios y sistemas de amarre para herramientas manuales en altura.\n"
                    "- **Control administrativo:** Delimitación física y señalización de radios de giro de grúas e izajes; plan de maniobra certificado.\n"
                    "- **Equipo de protección personal (EPP):** Casco de seguridad industrial Tipo I/II Clase E con barboquejo elástico."
                ),
                "Incidente_Vehicular": (
                    "- **Control de ingeniería:** Segregación física peatón-vehículo con barreras de polímero de impacto; limitadores de velocidad calibrados a 10 km/h en montacargas.\n"
                    "- **Control administrativo:** Certificación anual de operadores según OSHA 1910.178; instalación de espejos convexos y sensores de proximidad.\n"
                    "- **Equipo de protección personal (EPP):** Chaleco reflectivo de alta visibilidad Clase 2 según norma ANSI/ISEA 107."
                ),
                "Exposicion_Sustancias_Peligrosas": (
                    "- **Control de ingeniería:** Sistemas cerrados de trasvase; campanas de extracción localizada en la fuente de vapores corrosivos.\n"
                    "- **Control administrativo:** Hojas de datos de seguridad (FDS/SDS) accesibles según GHS; estaciones lavaojos a menos de 10 segundos de alcance.\n"
                    "- **Equipo de protección personal (EPP):** Goggles herméticos con ventilación indirecta, guantes de nitrilo/neopreno de manga larga y respirador químico específico."
                ),
                "Sobreesfuerzo_Ergonomico": (
                    "- **Control de ingeniería:** Mesas elevadoras hidráulicas de tijera, polipastos por vacío y diseño del plano de trabajo entre nudillos y hombros.\n"
                    "- **Control administrativo:** Límite máximo de levantamiento individual de 23 kg según ecuación NIOSH; rotación de puestos y pausas activas.\n"
                    "- **Equipo de protección personal (EPP):** Calzado ergonómico con suela amortiguadora de impacto y fajas biomecánicas de soporte cuando aplique."
                ),
            }

            st.info(
                f"**Prescripción técnica inmediata según la jerarquía de controles de OSHA:**\n\n"
                f"{controles.get(cat_emb, 'Consultar el procedimiento operativo estándar.')}"
            )

    st.markdown("---")

    # Matriz maestra
    st.markdown("### Plan integral de control de riesgos según la jerarquía de OSHA")

    df_controles = pd.DataFrame({
        "Riesgo crítico": [
            "Caída a distinto nivel",
            "Atrapamiento en maquinaria",
            "Golpeado por objeto",
            "Incidente vehicular",
            "Sustancias peligrosas",
            "Sobreesfuerzo ergonómico",
        ],
        "Controles de ingeniería prioritarios": [
            "Plataformas móviles (PEMP) en vez de escaleras; barandas perimetrales y anclajes certificados.",
            "Guardas fijas interbloqueadas con micro-switches; cortinas ópticas en zonas de peligro.",
            "Redes perimetrales anticaída; rodapiés en andamios; amarres obligatorios para herramientas manuales.",
            "Segregación física peatón-vehículo con barreras de polímero; limitadores a 10 km/h en montacargas.",
            "Extracción localizada en fuente; sistemas cerrados de trasvase; ventilación forzada en recintos confinados.",
            "Mesas elevadoras hidráulicas; polipastos por vacío; diseño del plano de trabajo entre nudillos y hombros.",
        ],
        "Controles administrativos obligatorios": [
            "Permiso de trabajo en altura (>1.80 m); inspección preoperacional de arneses y líneas de vida.",
            "Procedimiento LOTO (bloqueo y etiquetado - OSHA 1910.147) con verificación de energía cero.",
            "Delimitación estricta y acordonamiento de radios de izaje; planes de maniobra certificados.",
            "Certificación anual de operadores de montacargas (OSHA 1910.178); espejos convexos en cruces ciegos.",
            "Hojas de datos de seguridad (FDS/SDS) accesibles según GHS; estaciones lavaojos a menos de 10 segundos.",
            "Límite máximo de levantamiento manual individual de 23 kg (Ecuación NIOSH); pausas activas programadas.",
        ],
        "Equipo de protección personal (EPP)": [
            "Arnés de cuerpo entero con absorbedor de impacto y doble línea de vida.",
            "Prohibición estricta de prendas holgadas, guantes o accesorios cerca de ejes giratorios.",
            "Casco industrial Tipo I/II Clase E con barboquejo y calzado con puntera de seguridad.",
            "Chaleco reflectivo de alta visibilidad Clase 2 según norma ANSI/ISEA 107.",
            "Respirador con filtros químicos específicos; gafas herméticas (goggles) y guantes de nitrilo/neopreno.",
            "Calzado ergonómico con suela antifatiga y plantilla biomecánica de amortiguación.",
        ],
    })
    st.dataframe(df_controles, hide_index=True)

    st.markdown("---")

    # Conclusiones estratégicas
    st.markdown("### Conclusiones estratégicas y hoja de ruta para la dirección")
    st.markdown(
        """
        1. **Foco prioritario en logística y distribución (DART 5.04):**
           El sector de almacenamiento y depósitos registra la mayor tasa de incapacidad del país. 
           La inversión prioritaria debe dirigirse a la segregación física entre montacargas y peatones 
           y al rediseño ergonómico del surtido manual de piezas pesadas.

        2. **Reconfiguración del modelo preventivo de jornada (Fatiga horaria):**
           Dado que el **42.5% de los accidentes ocurre en las primeras 4 horas**, las pausas activas y el acondicionamiento biomecánico 
           deben realizarse al inicio del turno, no al cierre. Paralelamente, deben fijarse límites estrictos a jornadas extraordinarias 
           que excedan las 10 horas de labor continua.

        3. **Adopción de modelos de procesamiento de lenguaje natural (PLN) para auditoría y triaje:**
           - Emplear pipelines masivos basados en TF-IDF para auditar retrospectivamente el 100% de los incidentes históricos en segundos en CPU.
           - Implementar Sentence Transformers en la interfaz de reporte en planta para estandarizar la clasificación narrativa y emitir 
             controles técnicos en el momento exacto del incidente.

        4. **Auditoría del sesgo de subreporte en plantas pequeñas:**
           El ratio de 1.8x casos graves frente a leves en microempresas evidencia que los incidentes menores se ocultan. 
           Los programas corporativos deben incentivar el reporte transparente de cuasi-accidentes y lesiones menores sin penalización.
        """
    )


with tab_ficha:
    render_ficha_tecnica(
        metricas_globales=metricas_globales,
        metricas_modelos=metricas_modelos,
        investigaciones=datos_investigaciones,
        dist_tipos=dist_tipos,
        df_predicciones_test=df_predicciones_test,
        df_sectores=df_sectores,
    )
