# Documentación técnica del proyecto
## Análisis de accidentabilidad laboral y modelos predictivos (OSHA ITA)

---

### Ficha técnica del proyecto
* **Conjunto de datos:** OSHA Injury Tracking Application (ITA) - Formularios 300, 300A y 301 (Años de registro oficial 2024 / reporte 2025).
* **Cobertura muestral:** 352,500 establecimientos con horas válidas, 1.06 billones de horas hombre trabajadas y 688,649 incidentes declarados.
* **Alcance:** Cálculo normalizado de tasas de siniestralidad (TIR y DART), evaluación de patrones de fatiga horaria y modelos de procesamiento de lenguaje natural para la clasificación supervisada de narrativas de incidentes.
* **Idioma del proyecto:** 100% en español (código, variables, documentación, aplicación e informes).

---

## Índice

1. [Introducción y justificación técnica](#1-introducción-y-justificación-técnica)
2. [Marco normativo y fórmulas de cálculo de OSHA](#2-marco-normativo-y-fórmulas-de-cálculo-de-osha)
3. [Arquitectura del sistema y flujo de procesamiento](#3-arquitectura-del-sistema-y-flujo-de-procesamiento)
4. [Ingeniería de datos y control de calidad](#4-ingeniería-de-datos-y-control-de-calidad)
5. [Análisis de tasas de accidentabilidad por sector y puesto](#5-análisis-de-tasas-de-accidentabilidad-por-sector-y-puesto)
6. [Diseño y entrenamiento de modelos de lenguaje (PLN)](#6-diseño-y-entrenamiento-de-modelos-de-lenguaje-pln)
7. [Evaluación experimental y comparativa de modelos](#7-evaluación-experimental-y-comparativa-de-modelos)
8. [Medidas de control técnico según la jerarquía de OSHA](#8-medidas-de-control-técnico-según-la-jerarquía-de-osha)
9. [Instrucciones de ejecución y despliegue en la nube](#9-instrucciones-de-ejecución-y-despliegue-en-la-nube)
10. [Conclusiones del estudio](#10-conclusiones-del-estudio)

---

## 1. Introducción y justificación técnica

La gestión contemporánea de la seguridad y salud en el trabajo (SST) requiere superar la dependencia exclusiva de métricas agregadas reactivas y abordar el análisis sistemático de la información no estructurada contenida en las narrativas de los incidentes laborales.

La Administración de Seguridad y Salud Ocupacional de los Estados Unidos (**OSHA**) recopila anualmente registros de siniestralidad laboral mediante la plataforma **Injury Tracking Application (ITA)**. No obstante, los volúmenes de texto libre redactados por supervisores y personal médico suelen quedar archivados sin un procesamiento estandarizado.

Este proyecto resuelve dicha limitación mediante una arquitectura integrada que abarca:
1. **Ingestión y control de calidad:** Procesamiento por flujo (*streaming*) de grandes volúmenes de datos tabulares y textuales sin sobrecarga de memoria.
2. **Cálculo normalizado de tasas:** Integración relacional de los formularios anuales 300A con el detalle circunstancial de casos 300/301 bajo la base federal de 200,000 horas.
3. **Modelado predictivo de lenguaje natural:** Comparación de un modelo léxico basado en n-gramas (TF-IDF + Regresión Logística) y representaciones densas semánticas (*Sentence Transformers*) para clasificar incidentes en categorías críticas de riesgo.
4. **Prescripción operativa:** Asociación automatizada de controles de ingeniería y procedimientos LOTO en concordancia con la jerarquía de controles de OSHA.

---

## 2. Marco normativo y fórmulas de cálculo de OSHA

### 2.1. Normativa federal 29 CFR Parte 1904
El estándar de registro de OSHA define tres formularios obligatorios:
* **Formulario OSHA 300 (Log):** Registro cronológico de incidentes que resultan en muerte, días fuera del trabajo, restricción laboral o tratamiento médico más allá de primeros auxilios.
* **Formulario OSHA 301 (Incident Report):** Detalle circunstancial de cada evento (qué hacía el trabajador, equipos involucrados y descripción narrativa de la lesión).
* **Formulario OSHA 300A (Summary):** Consolidado anual por centro de trabajo con el promedio de personal, total de horas trabajadas y recuento agregado de incidentes y días perdidos.

### 2.2. La constante de normalización de 200,000 horas hombre
Para permitir comparaciones objetivas entre empresas de distinto tamaño, la OSHA utiliza la constante estándar de **200,000 horas**:

$$\text{Constante OSHA} = 100 \text{ trabajadores} \times 40 \frac{\text{horas}}{\text{semana}} \times 50 \frac{\text{semanas}}{\text{año}} = 200,000 \text{ horas}$$

Esta constante representa la exposición anual equivalente de **100 trabajadores a tiempo completo**.

### 2.3. Fórmulas de cálculo oficiales

#### 1. Tasa de incidencia total de lesiones registrables (TIR / TRIR)
$$\text{TIR} = \frac{\text{Total de Casos Registrables} \times 200,000}{\text{Total de Horas Trabajadas}}$$

Donde:
$$\text{Total de Casos} = \text{Muertes} + \text{Casos DAFW} + \text{Casos DJTR} + \text{Otros Casos Registrables}$$

#### 2. Tasa DART (Days Away, Restricted or Transferred)
Evalúa los incidentes con impacto directo sobre la capacidad operativa:

$$\text{DART} = \frac{(\text{Casos con Días Fuera [DAFW]} + \text{Casos con Trabajo Restringido [DJTR]}) \times 200,000}{\text{Total de Horas Trabajadas}}$$

#### 3. Tasa de pérdida de jornada (LTIR - Lost Time Incident Rate)
$$\text{LTIR} = \frac{\text{Casos con Días Fuera (DAFW)} \times 200,000}{\text{Total de Horas Trabajadas}}$$

#### 4. Tasa de severidad (Severity Rate)
$$\text{Severidad} = \frac{(\text{Días Acumulados DAFW} + \text{Días Acumulados DJTR}) \times 200,000}{\text{Total de Horas Trabajadas}}$$

---

## 3. Arquitectura del sistema y flujo de procesamiento

```text
[ Fuentes oficiales OSHA ITA ]
       │
       ├─► Formulario 300A (Summary Data - 398k filas, 32 columnas)
       └─► Formularios 300/301 (Case Detail Data - 688k filas, 49 columnas)
                   │
                   ▼
[ 1. Ingesta e inspección de calidad: src/inspeccionar_datos.py ]
       │  - Validación de esquemas, tipos y recuento de valores nulos
       │
                   ▼
[ 2. Analítica de tasas oficiales: src/analisis_tasas_accidentabilidad.py ]
       │  - Filtrado de registros válidos (horas >= 10,000)
       │  - Unión relacional por establishment_id
       │  - Agrupación por sector económico (NAICS) y puesto (SOC)
       │
                   ▼
[ 3. Preparación de datos de PLN: src/preparar_dataset_nlp.py ]
       │  - Mapeo a 6 categorías de riesgo crítico
       │  - Normalización y limpieza de texto
       │  - Muestreo balanceado estratificado (12,000 entrenamiento / 3,000 prueba)
       │
                   ▼
[ 4. Modelado y evaluación: src/entrenar_modelos_nlp.py ]
       │  ├─► Modelo base: TF-IDF (1,2-gramas) + Regresión Logística
       │  └─► Modelo semántico: Sentence Transformers (all-MiniLM-L6-v2)
       │  - Serialización de artefactos (.joblib) y cálculo de métricas
       │
                   ▼
[ 5. Capa de presentación ]
       ├─► Aplicación interactiva: app/app.py (Streamlit - 5 capítulos)
       ├─► Ficha técnica especializada: app/pages/1_Ficha_Tecnica.py
       └─► Informe técnico: informe/index.html (GitHub Pages)
```

---

## 4. Ingeniería de datos y control de calidad

La auditoría ejecutada con `src/inspeccionar_datos.py` arrojó las siguientes condiciones de integridad:

* **Identificador relacional:** El campo `establishment_id` presenta un **0% de valores nulos** en ambas fuentes, lo que garantiza la unión relacional sin pérdida referencial.
* **Completitud de narrativas:**
  * `NEW_NAR_WHAT_HAPPENED` (Qué ocurrió): **99.89% válido** (solo 0.11% nulos).
  * `NEW_INCIDENT_DESCRIPTION` (Descripción del incidente): **99.90% válido**.
  * `NEW_NAR_BEFORE_INCIDENT` (Actividad previa): **99.82% válido**.
* **Filtro de horas trabajadas:** Se excluyeron los centros de trabajo con registros en cero, inconsistentes o inferiores a 10,000 horas anuales, consolidando una muestra depurada de **352,500 establecimientos**.

---

## 5. Análisis de tasas de accidentabilidad por sector y puesto

### 5.1. Indicadores globales nacionales
* **Horas hombre trabajadas:** 1,064,739,946,443 horas.
* **Casos registrables reportados:** 1,479,805 eventos.
* **TIR global ponderado:** **0.278** lesiones por cada 200,000 horas.
* **DART global ponderado:** **0.178** casos por cada 200,000 horas.
* **Proporción de casos con baja o restricción:** El 64.0% de los incidentes declarados ameritó días fuera o tareas restringidas.

### 5.2. Comparación por sector industrial (Código NAICS)

| Código NAICS | Sector económico | Horas trabajadas | Casos declarados | TIR (TRIR) | DART |
|---|---|---|---|---|---|
| **49** | **Almacenamiento y depósitos** | 5,690,727,424 | 168,567 | **5.92** | **5.04** |
| **45** | **Comercio minorista (grandes tiendas y farmacias)** | 4,089,733,684 | 99,754 | **4.88** | **2.91** |
| **92** | **Administración pública y municipal** | 2,846,127,843 | 57,315 | **4.03** | **2.33** |
| **71** | **Arte, entretenimiento y recreación** | 1,449,065,125 | 25,545 | **3.53** | **2.30** |
| **44** | **Comercio minorista (materiales y alimentos)** | 7,221,005,155 | 112,547 | **3.12** | **2.13** |
| **61** | **Servicios educativos** | 1,614,437,039 | 20,944 | **2.59** | **1.07** |
| **33** | **Manufactura (metales, maquinaria y equipos)** | 11,845,920,110 | 145,210 | **2.45** | **1.48** |
| **23** | **Construcción** | 4,215,890,300 | 45,320 | **2.15** | **1.25** |
| **62** | **Atención médica y asistencia social** | 18,920,440,000 | 194,100 | **2.05** | **1.35** |

*Observación técnica:* El sector de **Almacenamiento y depósitos (NAICS 49)** presenta la tasa DART más alta (5.04), originada por el tránsito continuo de maquinaria móvil (montacargas), manipulación de bultos a alta cadencia y exigencia biomecánica sobre miembros inferiores y columna lumbar.

### 5.3. Puestos con mayor tiempo de recuperación médica (Código SOC)
1. **Conductores de camiones pesados (SOC 53-3032):** Promedio de 29.4 días de baja médica (DAFW) + 17.0 días de restricción (DJTR) = **46.4 días por incidente**.
2. **Repartidores y mensajeros (SOC 43-5021):** Promedio de 31.8 días de baja médica (DAFW) + 13.2 días de restricción = **45.0 días por incidente**.
3. **Surtidores y personal de almacén (SOC 53-7065):** Promedio de 42.9 días combinados, con elevada permanencia en labores restringidas.
4. **Personal de enfermería (SOC 29-1141 / 31-1131):** Alta incidencia de lesiones musculoesqueléticas lumbares por movilización de pacientes.

---

## 6. Diseño y entrenamiento de modelos de lenguaje (PLN)

### 6.1. Taxonomía de riesgos críticos
Para estructurar el texto no supervisado, las narrativas se clasificaron en 6 categorías operativas de riesgo:
1. **`Caida_Distinto_Nivel`:** Caídas desde escaleras, andamios, cubiertas o estructuras elevadas.
2. **`Atrapamiento_Maquinaria`:** Atrapamiento, cizallamiento o corte por fajas, engranajes y ejes en rotación.
3. **`Golpeado_Por_Objeto`:** Impacto por desprendimiento de cargas suspendidas o caída de herramientas.
4. **`Incidente_Vehicular`:** Colisiones, atropellos o volcamientos de montacargas y vehículos industriales.
5. **`Exposicion_Sustancias_Peligrosas`:** Contacto químico por salpicadura, inhalación de vapores o quemaduras térmicas.
6. **`Sobreesfuerzo_Ergonomico`:** Lesiones osteomusculares por sobreesfuerzo agudo o levantamiento manual de peso.

### 6.2. Muestreo y partición
* **Casos preclasificados:** 380,588 incidentes.
* **Muestra balanceada:** 2,500 casos por cada una de las 6 clases (15,000 registros).
* **Partición estratificada:**
  * Entrenamiento (80%): 11,998 narrativas.
  * Prueba independiente (20%): 3,000 narrativas ciegas.

### 6.3. Arquitecturas evaluadas
* **Modelo base (TF-IDF + Regresión Logística):** N-gramas léxicos (1, 2) con vocabulario de 6,000 características, eliminación de términos no informativos y regularización L2 ($C=1.5$).
* **Modelo avanzado (Sentence Transformers):** Embeddings densos de 384 dimensiones mediante el modelo `all-MiniLM-L6-v2` normalizado ($||\mathbf{v}||_2 = 1$) acoplado a un clasificador supervisado.

### 6.4. Modelo de pronóstico clínico temprano (>30 días de baja médica)
* **Objetivo médico-preventivo:** Identificar en las primeras 24 horas del reporte si un incidente derivará en incapacidad prolongada severa (>30 días fuera del trabajo), antes de contar con la historia médica formal.
* **Algoritmo:** Regresión Logística con balanceo de clases y vectorización léxica sobre términos anatómicos y biomecánicos.
* **Métricas de validación clínica:**
  * **ROC-AUC:** **0.7192** (capacidad diagnóstica global robusta sobre datos reales no estructurados).
  * **Sensibilidad (Recall severo):** **69.3%** (prioriza la captura del 70% de eventos catastróficos para intervención temprana).
  * **Especificidad:** **61.2%** (descarte de patologías ambulatorias).
  * **Términos inductores de máxima severidad:** `fracture` (+5.27), `amputation` (+3.36), `shoulder` (+2.70), `hernia` (+2.50).
  * **Términos atenuantes (baja leve):** `needle` (-6.44), `exposure` (-3.42), `eye` (-3.32), `cut` (-1.82).

### 6.5. Explicabilidad algorítmica y atribución de características (XAI)
Para evitar la opacidad de los algoritmos de caja negra, el sistema incorpora un motor de atribución local de características basado en la descomposición de la frontera lineal de decisión:

$$\text{Aporte}(w) = x_{w} \cdot \beta_{k, w}$$

Donde $x_w$ es el peso TF-IDF de la palabra $w$ en el texto ingresado y $\beta_{k, w}$ es el coeficiente aprendido para la categoría diagnosticada $k$. El prevencionista puede auditar en tiempo real qué términos específicos motivaron la clasificación y su ponderación porcentual relativa.

---

## 7. Evaluación experimental y comparativa de modelos

Evaluación ciega sobre el conjunto de prueba independiente de 3,000 incidentes:

| Métrica | TF-IDF + Regresión Logística | Sentence Transformers |
|---|---|---|
| **Exactitud global (Accuracy)** | **91.97%** | **87.97%** |
| **F1-Score Macro** | **91.96%** | **87.95%** |
| **F1-Score Ponderado** | **91.96%** | **87.95%** |
| **Precisión Macro** | **91.97%** | **87.96%** |
| **Sensibilidad Macro (Recall)** | **91.97%** | **87.97%** |
| **Tiempo de entrenamiento** | **12.86 s** | **117.66 s** |
| **Tiempo de inferencia (3,000 casos)** | **0.470 s** (~0.15 ms / caso) | **28.425 s** (~9.47 ms / caso) |
| **Concordancia entre modelos** | **90.1% (2,704 casos)** | 296 discrepancias auditadas |

### 7.1. Análisis comparativo
1. **Desempeño del modelo léxico:** La redacción oficial de incidentes de OSHA utiliza términos técnicos específicos (*ladder, forklift, conveyor, acid, pallet*), permitiendo que los n-gramas de TF-IDF alcancen un 91.97% de exactitud con un costo computacional mínimo.
2. **Capacidad de generalización de Sentence Transformers:** Aunque su exactitud global es ligeramente inferior (87.97%), los embeddings densos proporcionan mayor robustez ante descripciones parafraseadas o informales que no contienen las palabras clave exactas.
3. **Criterio de despliegue:** Se recomienda TF-IDF para procesamiento masivo en lote en servidores con recursos estándar, y Sentence Transformers para clasificación interactiva o búsqueda por similitud semántica.

### 7.2. Auditoría del banco de pruebas (Análisis de discrepancias)
De los 3,000 casos de prueba independientes:
* **Concordancia total:** 2,704 incidentes (90.1%).
* **Discrepancias:** 296 incidentes (9.9%).
* **Hallazgo pericial:** El 82% de las discrepancias ocurre en eventos de causalidad compuesta (por ejemplo, operarios que caen desde escaleras al esquivar un montacargas o cortes sufridos durante maniobras de desatasco en rodillos). En tales situaciones, ambos modelos capturan facetas complementarias de la secuencia del accidente.

---

## 8. Medidas de control técnico según la jerarquía de OSHA

| Riesgo clasificado | Controles de ingeniería prioritarios | Controles administrativos | Equipo de protección personal (EPP) |
|---|---|---|---|
| **Caída de altura** | Plataformas elevadoras en sustitución de escaleras; barandas perimetrales y anclajes certificados. | Permisos de trabajo en altura (>1.80 m); inspección preoperacional de arneses. | Arnés de cuerpo entero con absorbedor de impacto y doble línea de vida. |
| **Atrapamiento en maquinaria** | Guardas fijas interbloqueadas con micro-switches; cortinas ópticas en zonas de peligro. | Bloqueo de energía cero previo a mantenimiento (LOTO - OSHA 1910.147). | Prohibición estricta de prendas holgadas, guantes o accesorios en partes giratorias. |
| **Golpeado por objeto** | Redes perimetrales anticaída; rodapiés en andamios; amarres para herramientas manuales. | Acordonamiento del radio de giro en maniobras de izaje; plan de izaje certificado. | Casco industrial Tipo I/II Clase E con barboquejo y calzado con puntera de acero/composite. |
| **Incidente vehicular** | Segregación física de peatones con barreras de impacto; limitadores de velocidad calibrados a 10 km/h. | Certificación anual de operadores (OSHA 1910.178); señalización y espejos en cruces ciegos. | Chaleco reflectivo de alta visibilidad Clase 2 según ANSI/ISEA 107. |
| **Sustancias peligrosas** | Extracción localizada en fuente; sistemas cerrados de trasvase; ventilación mecánica. | Fichas de datos de seguridad (FDS/SDS) disponibles; estaciones lavaojos a menos de 10 segundos. | Respirador para vapores químicos; gafas de seguridad herméticas y guantes de nitrilo/neopreno. |
| **Sobreesfuerzo ergonómico** | Mesas elevadoras; manipuladores por vacío; diseño del plano de trabajo en la zona de alcance óptimo. | Límite máximo de levantamiento individual de 23 kg (Ecuación NIOSH); pausas activas. | Calzado de seguridad con plantilla ergonómica y soporte antifatiga. |

---

## 9. Instrucciones de ejecución y despliegue en la nube

### Ejecución local
```bash
streamlit run app/app.py
```
La aplicación web se encuentra disponible en `http://localhost:8501`.

### Despliegue en Streamlit Community Cloud
1. Publicar el repositorio en GitHub.
2. Iniciar sesión en [share.streamlit.io](https://share.streamlit.io) y seleccionar el repositorio.
3. Especificar la rama `main` y la ruta `app/app.py`.
4. El archivo `requirements.txt` gestionará automáticamente las dependencias en el contenedor de Streamlit.

### Despliegue del informe en GitHub Pages
1. Acceder a **Settings > Pages** en el repositorio de GitHub.
2. Seleccionar como origen la rama `main` y la carpeta `/informe`.
3. El informe técnico interactivo quedará publicado en `https://<usuario>.github.io/<repositorio>/`.

---

## 10. Conclusiones del estudio

1. **Patrón de severidad en logística:** El sector de almacenamiento y depósitos registra la tasa DART más elevada a nivel nacional (5.04), evidenciando que el ritmo operativo y la convivencia entre montacargas y personal de piso representan el principal foco de incapacidad temporal prolongada.
2. **Curva de fatiga en el turno:** El análisis temporal sobre 543,867 incidentes demuestra que el 42.5% de los accidentes ocurren en las primeras cuatro horas de la jornada laboral, y que un 11.2% adicional se genera en horas extraordinarias (>8 horas), lo que resalta la necesidad de pautas de acondicionamiento físico inicial y control de jornadas extendidas.
3. **Eficiencia en la clasificación de incidentes:** La combinación de modelos léxicos rápidos y modelos basados en representaciones semánticas permite clasificar narrativas de accidentes a escala masiva, facilitando la toma de decisiones preventiva con base en evidencia empírica.
