"""
Módulo de Entrenamiento y Evaluación Comparativa de Modelos de PLN
===================================================================
Este script entrena, evalúa y compara dos arquitecturas de
procesamiento de lenguaje natural (PLN) para la clasificación de narrativas de accidentes OSHA:

1. Modelo Base:
   - Vectorización léxica: TF-IDF (Término Frecuencia - Frecuencia Inversa de Documento)
     con unigramas y bigramas (1, 2) y ponderación sublineal.
   - Clasificador: Regresión Logística Multiclase con regularización L2 y balanceo de pesos.

2. Modelo Avanzado de Aprendizaje Profundo:
   - Embeddings Semánticos Densos: Sentence Transformers ('all-MiniLM-L6-v2', 384 dimensiones).
   - Clasificador: Regresión Logística supervisada sobre las representaciones densas contextuales.

Métricas de Evaluación calculadas sobre el conjunto independiente de prueba (3,000 casos):
- Exactitud (Accuracy)
- Precisión Macro y Ponderada (Precision)
- Exhaustividad / Sensibilidad Macro y Ponderada (Recall)
- Puntuación F1 Macro y Ponderada (F1-Score)
- Matriz de Confusión normalizada
- Tiempo de entrenamiento e inferencia
"""

import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# Configuración de Hugging Face
DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
cache_local = DIRECTORIO_RAIZ / "cache_huggingface"
if cache_local.exists():
    os.environ["HF_HOME"] = str(cache_local)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer


def cargar_datos_procesados():
    """Carga los conjuntos balanceados de entrenamiento y prueba."""
    ruta_train = DIRECTORIO_RAIZ / "datos" / "procesados" / "dataset_narrativas_entrenamiento.csv"
    ruta_test = DIRECTORIO_RAIZ / "datos" / "procesados" / "dataset_narrativas_prueba.csv"

    print(f"Cargando conjunto de entrenamiento: {ruta_train}")
    df_train = pd.read_csv(ruta_train)

    print(f"Cargando conjunto de prueba:        {ruta_test}")
    df_test = pd.read_csv(ruta_test)

    # Filtrar nulos si hubieran
    df_train = df_train.dropna(subset=["narrativa_texto", "categoria_riesgo"]).reset_index(drop=True)
    df_test = df_test.dropna(subset=["narrativa_texto", "categoria_riesgo"]).reset_index(drop=True)

    return df_train, df_test


def entrenar_modelo_tfidf(X_train, y_train, X_test, y_test, clases):
    """
    Entrena el Modelo Base (TF-IDF + Regresión Logística) y evalúa su desempeño.
    """
    print("\n" + "=" * 70)
    print("1. ENTRENANDO MODELO BASE: TF-IDF + REGRESIÓN LOGÍSTICA")
    print("=" * 70)

    pipeline_tfidf = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=6000,
                sublinear_tf=True,
                stop_words="english",
            ),
        ),
        (
            "clasificador",
            LogisticRegression(
                C=1.5,
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])

    # Medir tiempo de entrenamiento
    inicio_entrenamiento = time.time()
    pipeline_tfidf.fit(X_train, y_train)
    tiempo_entrenamiento = time.time() - inicio_entrenamiento

    # Medir tiempo de inferencia
    inicio_inferencia = time.time()
    predicciones = pipeline_tfidf.predict(X_test)
    probabilidades = pipeline_tfidf.predict_proba(X_test)
    tiempo_inferencia = time.time() - inicio_inferencia

    # Métricas
    exactitud = accuracy_score(y_test, predicciones)
    f1_macro = f1_score(y_test, predicciones, average="macro")
    f1_ponderado = f1_score(y_test, predicciones, average="weighted")
    precision_macro = precision_score(y_test, predicciones, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, predicciones, average="macro", zero_division=0)

    reporte_detallado = classification_report(y_test, predicciones, output_dict=True, zero_division=0)
    matriz_conf = confusion_matrix(y_test, predicciones, labels=clases)

    print(f"• Tiempo de entrenamiento: {tiempo_entrenamiento:.2f} s")
    print(f"• Tiempo de inferencia ({len(X_test):,} muestras): {tiempo_inferencia:.3f} s")
    print(f"• Exactitud (Accuracy):    {exactitud * 100:.2f}%")
    print(f"• F1-Score Macro:          {f1_macro * 100:.2f}%")
    print(f"• F1-Score Ponderado:      {f1_ponderado * 100:.2f}%")

    metricas = {
        "nombre_modelo": "TF-IDF + Regresión Logística (Línea Base)",
        "exactitud": round(exactitud, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_ponderado": round(f1_ponderado, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "tiempo_entrenamiento_s": round(tiempo_entrenamiento, 2),
        "tiempo_inferencia_s": round(tiempo_inferencia, 3),
        "reporte_por_clase": reporte_detallado,
    }

    return pipeline_tfidf, predicciones, probabilidades, metricas, matriz_conf


def entrenar_modelo_embeddings(X_train, y_train, X_test, y_test, clases):
    """
    Entrena el Modelo Avanzado (Sentence Transformers + Regresión Logística) y evalúa su desempeño.
    """
    print("\n" + "=" * 70)
    print("2. ENTRENANDO MODELO AVANZADO: SENTENCE TRANSFORMERS (EMBEDDINGS) + CLASIFICADOR")
    print("=" * 70)

    print("Cargando modelo de embeddings (all-MiniLM-L6-v2) en memoria local...")
    modelo_st = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Generando embeddings para entrenamiento ({len(X_train):,} narrativas)...")
    inicio_vectorizacion = time.time()
    embeddings_train = modelo_st.encode(
        X_train.tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    tiempo_vectorizacion_train = time.time() - inicio_vectorizacion
    print(f"-> Embeddings generados en {tiempo_vectorizacion_train:.2f} s. Matriz: {embeddings_train.shape}")

    print(f"Generando embeddings para prueba ({len(X_test):,} narrativas)...")
    inicio_vectorizacion_test = time.time()
    embeddings_test = modelo_st.encode(
        X_test.tolist(),
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    tiempo_vectorizacion_test = time.time() - inicio_vectorizacion_test

    print("Entrenando clasificador supervisado sobre el espacio de embeddings...")
    clasificador = LogisticRegression(
        C=2.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    inicio_entrenamiento = time.time()
    clasificador.fit(embeddings_train, y_train)
    tiempo_entrenamiento_clasif = time.time() - inicio_entrenamiento

    # Inferencia
    inicio_inferencia = time.time()
    predicciones = clasificador.predict(embeddings_test)
    probabilidades = clasificador.predict_proba(embeddings_test)
    tiempo_inferencia_total = (time.time() - inicio_inferencia) + tiempo_vectorizacion_test

    # Métricas
    exactitud = accuracy_score(y_test, predicciones)
    f1_macro = f1_score(y_test, predicciones, average="macro")
    f1_ponderado = f1_score(y_test, predicciones, average="weighted")
    precision_macro = precision_score(y_test, predicciones, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, predicciones, average="macro", zero_division=0)

    reporte_detallado = classification_report(y_test, predicciones, output_dict=True, zero_division=0)
    matriz_conf = confusion_matrix(y_test, predicciones, labels=clases)

    tiempo_total_proceso = tiempo_vectorizacion_train + tiempo_entrenamiento_clasif

    print(f"• Tiempo total (embeddings + clasif): {tiempo_total_proceso:.2f} s")
    print(f"• Tiempo de inferencia ({len(X_test):,} muestras): {tiempo_inferencia_total:.3f} s")
    print(f"• Exactitud (Accuracy):               {exactitud * 100:.2f}%")
    print(f"• F1-Score Macro:                     {f1_macro * 100:.2f}%")
    print(f"• F1-Score Ponderado:                 {f1_ponderado * 100:.2f}%")

    metricas = {
        "nombre_modelo": "Sentence Transformers (Embeddings) + Regresión Logística",
        "exactitud": round(exactitud, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_ponderado": round(f1_ponderado, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "tiempo_entrenamiento_s": round(tiempo_total_proceso, 2),
        "tiempo_inferencia_s": round(tiempo_inferencia_total, 3),
        "reporte_por_clase": reporte_detallado,
    }

    return clasificador, modelo_st, predicciones, probabilidades, metricas, matriz_conf


def ejecutar_experimento_completo():
    """Función orquestadora de los modelos y guardado de artefactos."""
    df_train, df_test = cargar_datos_procesados()

    X_train = df_train["narrativa_texto"]
    y_train = df_train["categoria_riesgo"]

    X_test = df_test["narrativa_texto"]
    y_test = df_test["categoria_riesgo"]

    clases = sorted(y_train.unique().tolist())

    # 1. Modelo Base
    pipeline_tfidf, preds_tfidf, probs_tfidf, metrics_tfidf, conf_tfidf = entrenar_modelo_tfidf(
        X_train, y_train, X_test, y_test, clases
    )

    # 2. Modelo Embeddings
    clasif_emb, modelo_st, preds_emb, probs_emb, metrics_emb, conf_emb = entrenar_modelo_embeddings(
        X_train, y_train, X_test, y_test, clases
    )

    # Guardar artefactos
    carpeta_modelos = DIRECTORIO_RAIZ / "src" / "modelos"
    carpeta_modelos.mkdir(parents=True, exist_ok=True)
    carpeta_procesados = DIRECTORIO_RAIZ / "datos" / "procesados"

    print("\nGuardando modelos y métricas...")
    joblib.dump(pipeline_tfidf, carpeta_modelos / "modelo_tfidf_pipeline.joblib")
    joblib.dump(clasif_emb, carpeta_modelos / "clasificador_embeddings.joblib")

    # Guardar resumen de métricas comparativas
    comparativa = {
        "clases": clases,
        "modelo_base_tfidf": metrics_tfidf,
        "modelo_avanzado_embeddings": metrics_emb,
        "diferencia_f1_macro_pct": round((metrics_emb["f1_macro"] - metrics_tfidf["f1_macro"]) * 100, 2),
        "diferencia_exactitud_pct": round((metrics_emb["exactitud"] - metrics_tfidf["exactitud"]) * 100, 2),
    }

    with open(carpeta_procesados / "metricas_comparativas.json", "w", encoding="utf-8") as f:
        json.dump(comparativa, f, indent=4, ensure_ascii=False)

    # Guardar matrices de confusión
    matrices_dict = {
        "clases": clases,
        "matriz_tfidf": conf_tfidf.tolist(),
        "matriz_embeddings": conf_emb.tolist(),
    }
    with open(carpeta_procesados / "matriz_confusion_comparativa.json", "w", encoding="utf-8") as f:
        json.dump(matrices_dict, f, indent=4, ensure_ascii=False)

    # Guardar predicciones de prueba para visualización interactiva en Streamlit
    df_predicciones = df_test.copy()
    df_predicciones["prediccion_tfidf"] = preds_tfidf
    df_predicciones["confianza_tfidf"] = np.max(probs_tfidf, axis=1).round(4)
    df_predicciones["prediccion_embeddings"] = preds_emb
    df_predicciones["confianza_embeddings"] = np.max(probs_emb, axis=1).round(4)
    df_predicciones["acierto_tfidf"] = df_predicciones["categoria_riesgo"] == df_predicciones["prediccion_tfidf"]
    df_predicciones["acierto_embeddings"] = df_predicciones["categoria_riesgo"] == df_predicciones["prediccion_embeddings"]

    ruta_predicciones = carpeta_procesados / "predicciones_comparativas_test.csv"
    df_predicciones.to_csv(ruta_predicciones, index=False)
    print(f"-> Predicciones de prueba guardadas: {ruta_predicciones}")

    # Tabla resumen en consola
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO FINAL DE DESEMPEÑO:")
    print("=" * 70)
    print(f"{'Métrica':<25} {'TF-IDF (Base)':<20} {'Sentence Transformers':<25}")
    print("-" * 70)
    print(f"{'Exactitud (Accuracy)':<25} {metrics_tfidf['exactitud']*100:.2f}%{'':<14} {metrics_emb['exactitud']*100:.2f}%")
    print(f"{'F1-Score Macro':<25} {metrics_tfidf['f1_macro']*100:.2f}%{'':<14} {metrics_emb['f1_macro']*100:.2f}%")
    print(f"{'F1-Score Ponderado':<25} {metrics_tfidf['f1_ponderado']*100:.2f}%{'':<14} {metrics_emb['f1_ponderado']*100:.2f}%")
    print(f"{'Precisión Macro':<25} {metrics_tfidf['precision_macro']*100:.2f}%{'':<14} {metrics_emb['precision_macro']*100:.2f}%")
    print(f"{'Exhaustividad Macro':<25} {metrics_tfidf['recall_macro']*100:.2f}%{'':<14} {metrics_emb['recall_macro']*100:.2f}%")
    print(f"{'Tiempo Inferencia':<25} {metrics_tfidf['tiempo_inferencia_s']:.3f} s{'':<13} {metrics_emb['tiempo_inferencia_s']:.3f} s")
    print("=" * 70)


if __name__ == "__main__":
    ejecutar_experimento_completo()

