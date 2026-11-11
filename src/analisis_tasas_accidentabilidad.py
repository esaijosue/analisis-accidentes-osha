"""
Módulo de Cálculo de Tasas Oficiales de Accidentabilidad (OSHA)
================================================================
Este script implementa las fórmulas oficiales de la Administración de
Seguridad y Salud Ocupacional (OSHA) para evaluar el desempeño en seguridad
y salud en el trabajo (SST) a nivel de sector económico (código NAICS)
y ocupación laboral (código SOC).

Fórmulas estándar utilizadas (base 200,000 horas hombre trabajadas):
--------------------------------------------------------------------
1. TIR (Total Incident Rate / Tasa de Incidencia Total de Lesiones Registrables):
   TIR = (Total Casos Registrables * 200,000) / Total Horas Trabajadas

2. DART (Days Away, Restricted, or Transferred):
   DART = ((Casos DAFW + Casos DJTR) * 200,000) / Total Horas Trabajadas

3. LTIR (Lost Time Incident Rate / Tasa con Pérdida de Jornada):
   LTIR = (Casos DAFW * 200,000) / Total Horas Trabajadas

4. Tasa de Severidad (Severity Rate - Días Perdidos y Restringidos):
   Severidad = ((Días DAFW + Días DJTR) * 200,000) / Total Horas Trabajadas
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Mapeo oficial de los 2 dígitos iniciales del código NAICS a sectores industriales en español
DICCIONARIO_SECTORES_NAICS = {
    "11": "Agricultura, Silvicultura, Caza y Pesca",
    "21": "Minería, Canteras y Extracción de Petróleo/Gas",
    "22": "Servicios Públicos (Electricidad, Gas y Agua)",
    "23": "Construcción",
    "31": "Manufactura (Alimentos, Textiles, Madera)",
    "32": "Manufactura (Química, Plásticos, Minerales)",
    "33": "Manufactura (Metales, Maquinaria, Equipos)",
    "42": "Comercio Mayorista",
    "44": "Comercio Minorista (Vehículos, Materiales, Alimentos)",
    "45": "Comercio Minorista (Tiendas Generales, Farmacias)",
    "48": "Transporte y Logística (Aéreo, Terrestre, Marítimo)",
    "49": "Almacenamiento y Depósitos",
    "51": "Información y Telecomunicaciones",
    "52": "Finanzas y Seguros",
    "53": "Bienes Raíces, Alquiler y Arrendamiento",
    "54": "Servicios Profesionales, Científicos y Técnicos",
    "55": "Gestión de Empresas y Sociedades",
    "56": "Servicios Administrativos y Gestión de Residuos",
    "61": "Servicios Educativos",
    "62": "Atención Médica y Asistencia Social",
    "71": "Arte, Entretenimiento y Recreación",
    "72": "Servicios de Alojamiento y Alimentación",
    "81": "Otros Servicios (Reparación, Mantenimiento)",
    "92": "Administración Pública",
}


def obtener_descripcion_sector(codigo_naics: str) -> str:
    """Retorna el nombre comprensible en español del sector industrial según NAICS."""
    if pd.isna(codigo_naics):
        return "Sector No Especificado"
    codigo_str = str(codigo_naics).strip()[:2]
    return DICCIONARIO_SECTORES_NAICS.get(codigo_str, f"Sector Industrial {codigo_str}")


def procesar_tasas_establecimientos(ruta_summary: str) -> pd.DataFrame:
    """
    Carga el Formulario 300A (Summary), limpia inconsistencias de horas y calcula
    tasas OSHA por establecimiento.
    """
    print(f"Cargando archivo de Resumen Anual (300A): {ruta_summary} ...")
    columnas_resumen = [
        "establishment_id",
        "establishment_name",
        "naics_code",
        "industry_description",
        "annual_average_employees",
        "total_hours_worked",
        "total_deaths",
        "total_dafw_cases",
        "total_djtr_cases",
        "total_other_cases",
        "total_dafw_days",
        "total_djtr_days",
        "total_injuries",
        "total_skin_disorders",
        "total_respiratory_conditions",
        "total_poisonings",
        "total_hearing_loss",
        "total_other_illnesses",
    ]

    df_resumen = pd.read_csv(ruta_summary, usecols=columnas_resumen, low_memory=False)

    # Conversión numérica de variables clave
    columnas_numericas = [
        "annual_average_employees",
        "total_hours_worked",
        "total_deaths",
        "total_dafw_cases",
        "total_djtr_cases",
        "total_other_cases",
        "total_dafw_days",
        "total_djtr_days",
        "total_injuries",
    ]
    for col in columnas_numericas:
        df_resumen[col] = pd.to_numeric(df_resumen[col], errors="coerce").fillna(0)

    # Filtrar registros válidos para el cálculo de tasas:
    # Deben tener al menos 20,000 horas trabajadas (equivalente a 10 trabajadores a tiempo completo)
    # y horas positivas no nulas.
    registros_iniciales = len(df_resumen)
    df_valido = df_resumen[df_resumen["total_hours_worked"] >= 10000].copy()
    print(f"Registros totales: {registros_iniciales:,} | Registros válidos con horas trabajadas: {len(df_valido):,}")

    # Cálculo del total de casos registrables (Recordable Cases = Fatalidades + DAFW + DJTR + Otros)
    df_valido["total_casos_registrables"] = (
        df_valido["total_deaths"]
        + df_valido["total_dafw_cases"]
        + df_valido["total_djtr_cases"]
        + df_valido["total_other_cases"]
    )

    # Factor de normalización OSHA: 200,000 horas hombre
    FACTOR_OSHA = 200000.0

    # Tasas por establecimiento
    df_valido["tasa_tir"] = (
        df_valido["total_casos_registrables"] * FACTOR_OSHA
    ) / df_valido["total_hours_worked"]

    df_valido["tasa_dart"] = (
        (df_valido["total_dafw_cases"] + df_valido["total_djtr_cases"]) * FACTOR_OSHA
    ) / df_valido["total_hours_worked"]

    df_valido["tasa_ltir"] = (
        df_valido["total_dafw_cases"] * FACTOR_OSHA
    ) / df_valido["total_hours_worked"]

    df_valido["tasa_severidad"] = (
        (df_valido["total_dafw_days"] + df_valido["total_djtr_days"]) * FACTOR_OSHA
    ) / df_valido["total_hours_worked"]

    # Código NAICS a 2 dígitos y descripción en español
    df_valido["sector_naics_2dig"] = df_valido["naics_code"].astype(str).str.strip().str[:2]
    df_valido["nombre_sector_es"] = df_valido["sector_naics_2dig"].map(
        lambda x: DICCIONARIO_SECTORES_NAICS.get(x, f"Sector {x}")
    )

    return df_valido


def generar_tasas_agregadas_sector(df_establecimientos: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega los datos por sector industrial sumando casos y horas trabajadas reales,
    calculando la tasa ponderada oficial del sector.
    """
    agrupado = df_establecimientos.groupby(["sector_naics_2dig", "nombre_sector_es"]).agg(
        num_establecimientos=("establishment_id", "count"),
        total_empleados=("annual_average_employees", "sum"),
        total_horas_trabajadas=("total_hours_worked", "sum"),
        total_muertes=("total_deaths", "sum"),
        total_casos_dafw=("total_dafw_cases", "sum"),
        total_casos_djtr=("total_djtr_cases", "sum"),
        total_otros_casos=("total_other_cases", "sum"),
        total_dias_dafw=("total_dafw_days", "sum"),
        total_dias_djtr=("total_djtr_days", "sum"),
        total_casos_registrables=("total_casos_registrables", "sum"),
    ).reset_index()

    FACTOR_OSHA = 200000.0

    agrupado["tir_ponderado"] = (
        agrupado["total_casos_registrables"] * FACTOR_OSHA
    ) / agrupado["total_horas_trabajadas"]

    agrupado["dart_ponderado"] = (
        (agrupado["total_casos_dafw"] + agrupado["total_casos_djtr"]) * FACTOR_OSHA
    ) / agrupado["total_horas_trabajadas"]

    agrupado["ltir_ponderado"] = (
        agrupado["total_casos_dafw"] * FACTOR_OSHA
    ) / agrupado["total_horas_trabajadas"]

    agrupado["severidad_ponderada"] = (
        (agrupado["total_dias_dafw"] + agrupado["total_dias_djtr"]) * FACTOR_OSHA
    ) / agrupado["total_horas_trabajadas"]

    agrupado = agrupado.sort_values(by="tir_ponderado", ascending=False).reset_index(drop=True)
    return agrupado


def procesar_tasas_por_ocupacion(ruta_casos: str) -> pd.DataFrame:
    """
    Analiza la base de casos individuales (Case Detail Data) por ocupación (código SOC
    y puesto de trabajo), evaluando la distribución de severidad y tipos de incidentes.
    """
    print(f"Cargando archivo de Detalle de Casos (300/301): {ruta_casos} ...")
    columnas_casos = [
        "establishment_id",
        "job_description",
        "soc_code",
        "soc_description",
        "incident_outcome",
        "dafw_num_away",
        "djtr_num_tr",
        "event_title_pred",
    ]

    df_casos = pd.read_csv(ruta_casos, usecols=columnas_casos, low_memory=False)

    df_casos["dafw_num_away"] = pd.to_numeric(df_casos["dafw_num_away"], errors="coerce").fillna(0)
    df_casos["djtr_num_tr"] = pd.to_numeric(df_casos["djtr_num_tr"], errors="coerce").fillna(0)
    df_casos["dias_totales_perdidos"] = df_casos["dafw_num_away"] + df_casos["djtr_num_tr"]

    # Agrupar por ocupación SOC
    agrupado_soc = df_casos.groupby(["soc_code", "soc_description"]).agg(
        total_casos=("establishment_id", "count"),
        promedio_dias_fuera=("dafw_num_away", "mean"),
        promedio_dias_restringido=("djtr_num_tr", "mean"),
        promedio_dias_totales=("dias_totales_perdidos", "mean"),
        maximo_dias_perdidos=("dias_totales_perdidos", "max"),
    ).reset_index()

    agrupado_soc = agrupado_soc[agrupado_soc["total_casos"] >= 100]  # Ocupaciones con significancia estadística
    agrupado_soc = agrupado_soc.sort_values(by="total_casos", ascending=False).reset_index(drop=True)

    return agrupado_soc


def ejecutar_analisis_completo():
    """Ejecuta el procesamiento completo y genera los archivos analíticos procesados."""
    directorio_raiz = Path(__file__).resolve().parent.parent
    ruta_summary = directorio_raiz / "datos" / "ITA_300A_Summary_Data_2024_through_12-31-2025_v2.csv"
    ruta_casos = directorio_raiz / "datos" / "ITA_Case_Detail_Data_2024_through_12-31-2025.csv"
    carpeta_salida = directorio_raiz / "datos" / "procesados"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ANÁLISIS DE TASAS DE ACCIDENTABILIDAD SEGÚN METODOLOGÍA OFICIAL OSHA")
    print("=" * 80)

    # 1. Procesar datos de resumen por establecimiento
    df_establecimientos = procesar_tasas_establecimientos(str(ruta_summary))

    # 2. Calcular tasas por sector NAICS
    print("\nCalculando tasas agregadas ponderadas por sector industrial (NAICS)...")
    df_sector = generar_tasas_agregadas_sector(df_establecimientos)

    ruta_salida_sector = carpeta_salida / "tasas_por_sector_naics.csv"
    df_sector.to_csv(ruta_salida_sector, index=False)
    print(f"-> Archivo guardado: {ruta_salida_sector}")

    # 3. Procesar datos por ocupación SOC
    print("\nCalculando métricas de incidentes por ocupación (código SOC)...")
    df_ocupaciones = procesar_tasas_por_ocupacion(str(ruta_casos))

    ruta_salida_ocupaciones = carpeta_salida / "tasas_por_ocupacion_soc.csv"
    df_ocupaciones.to_csv(ruta_salida_ocupaciones, index=False)
    print(f"-> Archivo guardado: {ruta_salida_ocupaciones}")

    # 4. Generar métricas globales para el dashboard
    metricas_globales = {
        "total_establecimientos_analizados": int(len(df_establecimientos)),
        "total_horas_trabajadas_muestra": float(df_establecimientos["total_hours_worked"].sum()),
        "total_casos_registrables": int(df_establecimientos["total_casos_registrables"].sum()),
        "total_muertes_reportadas": int(df_establecimientos["total_deaths"].sum()),
        "tir_global_ponderado": round(
            float((df_establecimientos["total_casos_registrables"].sum() * 200000.0) / df_establecimientos["total_hours_worked"].sum()), 3
        ),
        "dart_global_ponderado": round(
            float(((df_establecimientos["total_dafw_cases"].sum() + df_establecimientos["total_djtr_cases"].sum()) * 200000.0) / df_establecimientos["total_hours_worked"].sum()), 3
        ),
        "ltir_global_ponderado": round(
            float((df_establecimientos["total_dafw_cases"].sum() * 200000.0) / df_establecimientos["total_hours_worked"].sum()), 3
        ),
        "top_5_sectores_mayor_riesgo": df_sector[["nombre_sector_es", "tir_ponderado", "dart_ponderado"]].head(5).to_dict(orient="records"),
    }

    ruta_metricas = carpeta_salida / "resumen_metricas_globales.json"
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas_globales, f, indent=4, ensure_ascii=False)
    print(f"-> Resumen global guardado: {ruta_metricas}")

    print("\n" + "=" * 80)
    print("RESUMEN EJECUTIVO DE TASAS GLOBALES OSHA:")
    print("=" * 80)
    print(f"• Establecimientos analizados:     {metricas_globales['total_establecimientos_analizados']:,}")
    print(f"• Total horas trabajadas:          {metricas_globales['total_horas_trabajadas_muestra']:,.0f} horas")
    print(f"• Total casos registrables:        {metricas_globales['total_casos_registrables']:,}")
    print(f"• Tasa TIR Global (TRIR):          {metricas_globales['tir_global_ponderado']} por cada 200,000 hrs")
    print(f"• Tasa DART Global:                {metricas_globales['dart_global_ponderado']} por cada 200,000 hrs")
    print(f"• Tasa LTIR Global:                {metricas_globales['ltir_global_ponderado']} por cada 200,000 hrs")
    print("-" * 80)
    print("Top 5 Sectores con Mayor Tasa de Accidentabilidad (TIR):")
    for i, sec in enumerate(metricas_globales["top_5_sectores_mayor_riesgo"], start=1):
        print(f"  {i}. {sec['nombre_sector_es']}: TIR = {sec['tir_ponderado']:.2f} | DART = {sec['dart_ponderado']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    ejecutar_analisis_completo()

