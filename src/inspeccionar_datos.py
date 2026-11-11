"""
Módulo de Inspección de Datos de Accidentes OSHA ITA
=====================================================
Este script se encarga de analizar los archivos CSV oficiales descargados
de la plataforma OSHA ITA (Injury Tracking Application).

Calcula y presenta de forma clara y detallada:
- Cantidad total de registros (filas).
- Cantidad y nombres de columnas.
- Conteo y porcentaje de valores nulos (vacíos) por cada variable.
"""

import csv
import os
import sys
from pathlib import Path


def formatear_numero(numero: int) -> str:
    """
    Formatea un número entero con separadores de miles para facilitar su lectura.
    Ejemplo: 398620 -> '398,620'
    """
    return f"{numero:,}"


def inspeccionar_archivo_csv(ruta_archivo: str) -> dict:
    """
    Lee un archivo CSV de forma eficiente (secuencial) sin saturar la memoria RAM.
    Calcula el total de filas, columnas y la cantidad de valores nulos por columna.

    Parámetros:
        ruta_archivo (str): Ruta completa o relativa al archivo CSV.

    Retorna:
        dict: Diccionario con los resultados del análisis.
    """
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")

    nombre_archivo = os.path.basename(ruta_archivo)
    tamano_megabytes = os.path.getsize(ruta_archivo) / (1024 * 1024)

    total_filas = 0
    nombres_columnas = []
    conteo_nulos = []

    # Abrimos con codificación utf-8 (o latin-1 como respaldo si hay caracteres especiales)
    with open(ruta_archivo, mode="r", encoding="utf-8", errors="replace") as archivo:
        lector_csv = csv.reader(archivo)

        # Leer la primera fila como cabecera (nombres de las columnas)
        try:
            cabecera = next(lector_csv)
            # Limpiar espacios en blanco de los nombres de columna
            nombres_columnas = [columna.strip() for columna in cabecera]
            conteo_nulos = [0] * len(nombres_columnas)
        except StopIteration:
            return {
                "nombre_archivo": nombre_archivo,
                "tamano_mb": tamano_megabytes,
                "total_filas": 0,
                "total_columnas": 0,
                "columnas": [],
                "nulos_por_columna": {},
            }

        numero_columnas_esperadas = len(nombres_columnas)

        # Procesar fila por fila acumulando estadísticas
        for fila in lector_csv:
            total_filas += 1
            longitud_fila = len(fila)

            for indice in range(numero_columnas_esperadas):
                if indice < longitud_fila:
                    valor = fila[indice].strip()
                    # Se considera nulo si la cadena está vacía o es 'NA', 'NULL', 'None'
                    if valor == "" or valor.upper() in {"NA", "NULL", "NONE"}:
                        conteo_nulos[indice] += 1
                else:
                    # Si la fila tiene menos columnas que la cabecera, se contabiliza como nulo
                    conteo_nulos[indice] += 1

    # Construir el diccionario de resultados por columna
    detalle_columnas = {}
    for indice, nombre_columna in enumerate(nombres_columnas):
        cantidad_nulos = conteo_nulos[indice]
        porcentaje_nulos = (cantidad_nulos / total_filas * 100) if total_filas > 0 else 0.0
        cantidad_validos = total_filas - cantidad_nulos
        detalle_columnas[nombre_columna] = {
            "posicion": indice + 1,
            "valores_validos": cantidad_validos,
            "valores_nulos": cantidad_nulos,
            "porcentaje_nulos": porcentaje_nulos,
        }

    return {
        "nombre_archivo": nombre_archivo,
        "tamano_mb": tamano_megabytes,
        "total_filas": total_filas,
        "total_columnas": len(nombres_columnas),
        "columnas": nombres_columnas,
        "detalle_columnas": detalle_columnas,
    }


def imprimir_reporte_inspeccion(resultados: dict) -> None:
    """
    Imprime en pantalla un reporte ordenado, claro y legible en español con los
    hallazgos de la inspección.
    """
    separador_principal = "=" * 90
    separador_secundario = "-" * 90

    print(separador_principal)
    print(f"REPORTE DE INSPECCIÓN: {resultados['nombre_archivo']}")
    print(separador_principal)
    print(f"• Tamaño en disco:        {resultados['tamano_mb']:.2f} MB")
    print(f"• Total de registros (filas):   {formatear_numero(resultados['total_filas'])}")
    print(f"• Total de variables (columnas): {resultados['total_columnas']}")
    print(separador_secundario)
    print(f"{'#':<4} {'Nombre de Columna':<35} {'Válidos':<15} {'Nulos':<15} {'% Nulos':<10}")
    print(separador_secundario)

    for nombre_columna, estadisticas in resultados["detalle_columnas"].items():
        num = estadisticas["posicion"]
        validos_str = formatear_numero(estadisticas["valores_validos"])
        nulos_str = formatear_numero(estadisticas["valores_nulos"])
        pct_str = f"{estadisticas['porcentaje_nulos']:.2f}%"

        # Truncar nombre si es muy largo para mantener alineación estética
        nombre_mostrado = (
            nombre_columna if len(nombre_columna) <= 34 else nombre_columna[:31] + "..."
        )
        print(f"{num:<4} {nombre_mostrado:<35} {validos_str:<15} {nulos_str:<15} {pct_str:<10}")

    print(separador_secundario)
    print()


def ejecutar_inspeccion():
    """
    Función principal que localiza los archivos en 'datos/' y ejecuta la inspección.
    """
    directorio_base = Path(__file__).resolve().parent.parent
    directorio_datos = directorio_base / "datos"

    if not directorio_datos.exists():
        print(f"Error: La carpeta '{directorio_datos}' no existe.")
        sys.exit(1)

    archivos_csv = sorted(directorio_datos.glob("*.csv"))

    if not archivos_csv:
        print(f"Aviso: No se encontraron archivos .csv en '{directorio_datos}'.")
        print("Asegúrate de colocar los archivos de OSHA ITA en la carpeta 'datos/'.")
        return

    print(f"\nSe encontraron {len(archivos_csv)} archivos CSV en el directorio 'datos/'.")
    print("Iniciando análisis detallado de filas, columnas y valores nulos...\n")

    for archivo in archivos_csv:
        print(f"Analizando: {archivo.name} ...")
        resultados = inspeccionar_archivo_csv(str(archivo))
        imprimir_reporte_inspeccion(resultados)


if __name__ == "__main__":
    ejecutar_inspeccion()

