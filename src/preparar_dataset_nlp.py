"""
Módulo de Preparación del Conjunto de Datos de Procesamiento de Lenguaje Natural (PLN)
====================================================================================
Este script extrae, limpia y estandariza las narrativas de incidentes laborales de
OSHA Case Detail Data para construir los conjuntos de entrenamiento y prueba.

Taxonomía de Categorías de Riesgo Crítico en Seguridad Industrial:
-----------------------------------------------------------------
1. Caida_Distinto_Nivel: Caídas de altura (escaleras, andamios, techos, plataformas).
2. Atrapamiento_Maquinaria: Atrapamiento o contacto con partes móviles o maquinaria en marcha.
3. Golpeado_Por_Objeto: Impacto por objetos en suspensión, desprendimiento de carga o herramientas.
4. Incidente_Vehicular: Colisiones o atropellos con montacargas, camiones o vehículos industriales.
5. Exposicion_Sustancias_Peligrosas: Contacto o inhalación de sustancias químicas, vapores o extremos térmicos.
6. Sobreesfuerzo_Ergonomico: Trastornos musculoesqueléticos por levantamiento manual o sobreesfuerzo.
"""

import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Mapeo de eventos OSHA OIICS a nuestras 6 categorías de Riesgo Crítico Industrial
MAPEO_CATEGORIAS_RIESGO = {
    # 1. Caídas a distinto nivel (Trabajo en altura)
    "Fall to lower level": "Caida_Distinto_Nivel",
    # 2. Atrapamiento en maquinaria
    "Struck, caught, or compressed by running powered equipment": "Atrapamiento_Maquinaria",
    # 3. Golpeado por objeto en movimiento o caída
    "Struck by propelled, falling, or suspended object": "Golpeado_Por_Objeto",
    # 4. Incidentes vehiculares / maquinaria rodante
    "Roadway incidents involving motorized land vehicle": "Incidente_Vehicular",
    "Nonroadway incidents involving motorized land vehicles": "Incidente_Vehicular",
    "Pedestrian incidents involving motorized land vehicles": "Incidente_Vehicular",
    # 5. Sustancias peligrosas y temperaturas extremas
    "Exposure to harmful substances": "Exposicion_Sustancias_Peligrosas",
    "Exposure to temperature extremes": "Exposicion_Sustancias_Peligrosas",
    # 6. Sobreesfuerzo / Ergonómico
    "Overexertion while moving or manipulating external object(s)": "Sobreesfuerzo_Ergonomico",
    "Bodily position and motion": "Sobreesfuerzo_Ergonomico",
}

# Nombres legibles y descripciones en español para visualización en la app y reportes
DESCRIPCIONES_RIESGO_ES = {
    "Caida_Distinto_Nivel": "Caída a Distinto Nivel (Trabajo en Altura)",
    "Atrapamiento_Maquinaria": "Atrapamiento en Maquinaria / Equipos en Movimiento",
    "Golpeado_Por_Objeto": "Golpeado por Objeto en Caída o Movimiento",
    "Incidente_Vehicular": "Incidente Vehicular / Equipos Móviles Industriales",
    "Exposicion_Sustancias_Peligrosas": "Exposición a Sustancias Peligrosas / Químicos",
    "Sobreesfuerzo_Ergonomico": "Sobreesfuerzo y Lesión Ergonómica",
}


def limpiar_texto(texto: str) -> str:
    """
    Limpia y normaliza el texto de las narrativas en inglés:
    - Remueve saltos de línea repetidos, tabulaciones y caracteres extraños.
    - Estandariza espacios múltiples.
    """
    if pd.isna(texto) or not isinstance(texto, str):
        return ""
    # Reemplazar caracteres no imprimibles o saltos de línea
    t = re.sub(r"[\r\n\t]+", " ", texto)
    # Reemplazar múltiples espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t


def construir_conjuntos_pln(limite_muestras_por_clase: int = 2500):
    """
    Procesa el archivo de detalle de casos y genera los datasets de entrenamiento y prueba.
    """
    directorio_raiz = Path(__file__).resolve().parent.parent
    ruta_casos = directorio_raiz / "datos" / "ITA_Case_Detail_Data_2024_through_12-31-2025.csv"
    carpeta_salida = directorio_raiz / "datos" / "procesados"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PREPARACIÓN DEL DATASET PARA MODELOS DE PROCESAMIENTO DE LENGUAJE NATURAL (PLN)")
    print("=" * 80)
    print(f"Leyendo casos desde: {ruta_casos} ...")

    columnas_requeridas = [
        "id",
        "establishment_id",
        "event_title_pred",
        "job_description",
        "NEW_NAR_WHAT_HAPPENED",
        "NEW_NAR_BEFORE_INCIDENT",
        "NEW_INCIDENT_DESCRIPTION",
        "NEW_NAR_OBJECT_SUBSTANCE",
    ]

    # Leemos en bloques para optimizar velocidad y memoria
    tamano_bloque = 100000
    bloques_filtrados = []

    for bloque in pd.read_csv(
        ruta_casos,
        usecols=columnas_requeridas,
        chunksize=tamano_bloque,
        encoding="latin-1",
        low_memory=False,
    ):
        # Filtrar solo aquellos que pertenezcan a nuestro mapeo de categorías
        bloque["categoria_riesgo"] = bloque["event_title_pred"].map(MAPEO_CATEGORIAS_RIESGO)
        bloque_valido = bloque[bloque["categoria_riesgo"].notna()].copy()

        if not bloque_valido.empty:
            bloques_filtrados.append(bloque_valido)

    df_total = pd.concat(bloques_filtrados, ignore_index=True)
    print(f"Total de registros clasificados en categorías de riesgo crítico: {len(df_total):,}")

    print("\nDistribución inicial por categoría:")
    for cat, cantidad in df_total["categoria_riesgo"].value_counts().items():
        print(f"  • {cat:<32}: {cantidad:,} registros")

    # Muestreo balanceado por clase para evitar sesgos de entrenamiento
    print(f"\nBalanceando a máximo {limite_muestras_por_clase:,} registros por clase...")
    muestras_balanceadas = []
    for cat, grupo in df_total.groupby("categoria_riesgo"):
        n_muestras = min(len(grupo), limite_muestras_por_clase)
        muestras_balanceadas.append(grupo.sample(n=n_muestras, random_state=42))

    df_balanceado = pd.concat(muestras_balanceadas, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Construir campo de texto unificado
    print("Construyendo y limpiando narrativas de incidentes...")
    narrativas_limpias = []
    for _, fila in df_balanceado.iterrows():
        p1 = limpiar_texto(fila["NEW_NAR_WHAT_HAPPENED"])
        p2 = limpiar_texto(fila["NEW_NAR_BEFORE_INCIDENT"])
        p3 = limpiar_texto(fila["NEW_INCIDENT_DESCRIPTION"])
        p4 = limpiar_texto(fila["NEW_NAR_OBJECT_SUBSTANCE"])

        partes = [p for p in [p1, p2, p3, p4] if len(p) > 2]
        texto_unificado = " | ".join(partes) if partes else p1
        narrativas_limpias.append(texto_unificado)

    df_balanceado["narrativa_texto"] = narrativas_limpias
    df_balanceado["nombre_categoria_es"] = df_balanceado["categoria_riesgo"].map(DESCRIPCIONES_RIESGO_ES)

    # Filtrar textos que resulten demasiado cortos (< 15 caracteres)
    df_balanceado = df_balanceado[df_balanceado["narrativa_texto"].str.len() >= 15].reset_index(drop=True)

    print(f"Total de registros balanceados limpios: {len(df_balanceado):,}")

    # División estratificada 80% entrenamiento / 20% prueba
    df_entrenamiento, df_prueba = train_test_split(
        df_balanceado,
        test_size=0.20,
        random_state=42,
        stratify=df_balanceado["categoria_riesgo"],
    )

    columnas_guardar = [
        "id",
        "establishment_id",
        "job_description",
        "categoria_riesgo",
        "nombre_categoria_es",
        "narrativa_texto",
    ]

    ruta_train = carpeta_salida / "dataset_narrativas_entrenamiento.csv"
    ruta_test = carpeta_salida / "dataset_narrativas_prueba.csv"

    df_entrenamiento[columnas_guardar].to_csv(ruta_train, index=False)
    df_prueba[columnas_guardar].to_csv(ruta_test, index=False)

    # Guardar metadatos de las clases
    metadatos = {
        "categorias": list(DESCRIPCIONES_RIESGO_ES.keys()),
        "descripciones_es": DESCRIPCIONES_RIESGO_ES,
        "total_entrenamiento": len(df_entrenamiento),
        "total_prueba": len(df_prueba),
    }
    with open(carpeta_salida / "etiquetas_categorias.json", "w", encoding="utf-8") as f:
        json.dump(metadatos, f, indent=4, ensure_ascii=False)

    print(f"\n-> Dataset de Entrenamiento guardado: {ruta_train} ({len(df_entrenamiento):,} filas)")
    print(f"-> Dataset de Prueba guardado:        {ruta_test} ({len(df_prueba):,} filas)")
    print("=" * 80)


if __name__ == "__main__":
    construir_conjuntos_pln()
