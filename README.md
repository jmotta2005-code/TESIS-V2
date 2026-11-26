# Sistema de análisis ambiental — Entrenador Random Forest

Aplicación para entrenar y evaluar modelos de **Random Forest** (clasificación o regresión)
sobre datos ambientales, por ejemplo para predecir la **salinidad del suelo** a partir de
bandas e índices espectrales.

Está compuesta por dos partes:

- **Backend** (`backend/`): API en FastAPI que lee los archivos, prepara los datos, entrena,
  evalúa y guarda los modelos en memoria.
- **Frontend** (`frontend/`): interfaz en Streamlit desde la que se suben los archivos y se
  revisan los resultados.

---

## 1. Requisitos

- Windows (también funciona en Linux/macOS).
- Python 3.10 o superior.
- `pip`.

Verificar la instalación:

```powershell
py --version
```

---

## 2. Instalar dependencias

Abrir **dos terminales**.

**Terminal 1 – Backend:**

```powershell
cd backend
py -m pip install -r requirements.txt
```

**Terminal 2 – Frontend:**

```powershell
cd frontend
py -m pip install -r requirements.txt
```

---

## 3. Ejecutar el sistema

> Importante: iniciar **primero el backend** y después el frontend. Mantener ambas terminales
> abiertas mientras se usa la aplicación.

**Paso 1 — Backend** (desde la carpeta `backend`):

```powershell
py -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Esperar a ver `Application startup complete` en la terminal.

**Paso 2 — Frontend** (desde la carpeta `frontend`, en la otra terminal):

```powershell
py -m streamlit run app.py
```

Se abrirá el navegador. Si no se abre solo, entrar a:

- Aplicación: <http://localhost:8501>
- Documentación de la API: <http://127.0.0.1:8000/docs>

Si se cierra una terminal, el servicio correspondiente deja de funcionar.

---

## 4. Cómo se usa (paso a paso)

### Paso 1 — Subir las variables de entrenamiento (X)

En la columna **"1️⃣ Variables de entrenamiento (X)"**:

- Subir un archivo **CSV o Excel** (`.csv`, `.xlsx`, `.xls`) donde **cada fila es una muestra**
  y **cada columna una variable predictora**.
- Si el Excel tiene varias hojas, aparecerá un selector para elegir la hoja.
- Si el archivo incluye columnas de bandas espectrales (**B1, B2, ..., B12, B8A**), el sistema
  **calcula automáticamente índices espectrales** de salinidad (EQSI, SI, NDSI, CRSI, etc.) y los
  agrega como nuevas variables. Se muestra un aviso con los índices añadidos.

Ejemplo mínimo:

| B2   | B3   | B4   | B8A  | B11  |
|------|------|------|------|------|
| 0.12 | 0.18 | 0.22 | 0.35 | 0.41 |
| 0.13 | 0.17 | 0.21 | 0.33 | 0.39 |

### Paso 2 — Subir la variable objetivo (y)

En la columna **"2️⃣ Variable objetivo (y)"**:

- Subir otro **CSV o Excel** con la **variable que se quiere predecir** (por ejemplo,
  `salinidad`).
- El número de filas debe coincidir con el archivo X, **o** ambos archivos deben compartir una
  **columna ID común** para poder emparejarlos.

Ejemplo:

| salinidad |
|-----------|
| 4.2       |
| 5.1       |

### Paso 3 — Preparación

En la sección **"3️⃣ Preparación"**:

- **Columna objetivo**: si el archivo y tiene una sola columna se detecta sola; si tiene varias,
  se elige en el selector.
- **Emparejamiento de filas**:
  - *Por posición*: se unen en el mismo orden de filas.
  - *Por columna ID común*: se unen por el valor de una columna que exista en ambos archivos.
- **Orden temporal** (opcional): marcar si los datos tienen orden cronológico; el test serán las
  filas más recientes y la validación respetará el orden.
- **Columnas a excluir**: descartar variables que no deben usarse.
- **Exclusión automática**: elimina columnas constantes, fechas o que parecen IDs.
- **Eliminar filas duplicadas**: recomendado, porque los duplicados inflan las métricas.
- **Tipo de problema**: *Clasificación* (categorías) o *Regresión* (valor continuo). Para
  salinidad normalmente es **Regresión**.

### Paso 4 — Diagnóstico de datos

En **"4️⃣ Diagnóstico de datos"** hay tres pestañas:

- **Calidad de datos**: faltantes, outliers, valores únicos y alertas por columna.
- **Variable objetivo**: distribución (histograma para regresión o clases para clasificación).
- **Posible fuga de información**: entrena un árbol con una sola variable a la vez; si una
  variable predice casi perfecto, probablemente contiene la respuesta y conviene excluirla.

### Paso 5 — Entrenamiento

En **"5️⃣ Entrenamiento"**:

- Elegir la **métrica principal** (para regresión: R², RMSE o MAE; para clasificación: F1,
  accuracy, etc.).
- Elegir la **búsqueda de hiperparámetros** (Ninguna, Aleatoria o Rejilla).
- Ajustar en la **barra lateral** los parámetros del Random Forest y la partición
  (tamaño de test, folds, `random_state`).
- Pulsar **"🚀 Entrenar y evaluar"**.

### Resultados

Tras entrenar se muestra:

- **Comparación de modelos** (línea base, modelo simple y Random Forest) en CV y test.
- **Revisión de sobreajuste** (train vs validación vs test).
- **Métricas en test**: para regresión, **R², RMSE y MAE** con un gráfico *real vs predicho*.
- **Importancia de variables** por impureza y por permutación.
- Botón para **descargar el modelo** entrenado (`.joblib`).

Usa el **R²** y la cercanía entre validación y test para decidir si el modelo es confiable.

### Paso 6 — Predecir con datos nuevos (opcional)

En **"6️⃣ Predecir con datos nuevos"** se puede subir un **CSV o Excel** con las mismas
variables del entrenamiento. El sistema devuelve las predicciones en pantalla y un CSV
descargable.

> Los datos nuevos deben tener una distribución parecida a la del entrenamiento. Si cambian con
> el tiempo, conviene reentrenar.

### Paso 7 — Mapa de salinidad desde un CSV/Excel (coordenadas + bandas)

Después de revisar las métricas (por ejemplo el **R²**) y si el modelo te convence, puedes
generar el **mapa de salinidad** de un conjunto de puntos:

1. Subir uno o **varios CSV o Excel** con columnas de **coordenadas** (`x`, `y` o `lon`, `lat`)
   y las **bandas espectrales** (`B1`, `B2`, ..., `B12`). Útil cuando la exportación de Google
   Earth Engine se parte en varios archivos: se **concatenan y ordenan por coordenadas**
   automáticamente. **No** hace falta incluir los índices derivados: el sistema los calcula
   automáticamente a partir de las bandas.
2. Pulsar **"🗺️ Predecir distribución de salinidad"**. El procesamiento solo se ejecuta al
   pulsar el botón, no antes.
3. El sistema muestra la tabla con las predicciones, un **mapa de temperatura** de la
   salinidad (**azul = valor bajo, rojo = valor alto**) y estadísticas (mínimo, media, mediana y
   máximo).
4. Descargar las predicciones con **"⬇️ Descargar predicciones (CSV)"**.

> Usa datos con la **misma escala de reflectancia** que los de entrenamiento (por ejemplo,
> ambas en 0–1 o ambas en 0–10000). Si no coinciden, las predicciones no tendrán sentido.

---

## 5. Solución de problemas

| Problema | Causa / solución |
|----------|------------------|
| "No se pudo conectar con el backend" | El backend no está corriendo. Iniciarlo con `uvicorn main:app --port 8000`. |
| "Formato no soportado" | Solo se aceptan `.csv`, `.xlsx` y `.xls` en la carga de datos. |
| "Los archivos tienen distinto número de filas" | Usar una **columna ID común** o corregir los datos. |
| "Faltan columnas en el archivo nuevo" | El archivo de predicción debe tener las mismas variables usadas al entrenar. |
| "No quedan variables para entrenar" | Todas las columnas fueron excluidas o detectadas como ID/constantes. |
| Resultado sospechosamente alto | Posible fuga de información o duplicados; revisar el diagnóstico y excluir variables. |

---

## 6. Estructura del proyecto

```
TESIS-V2/
├── backend/
│   ├── main.py            # API FastAPI (endpoints)
│   ├── ml_core.py         # Lógica de ML (lectura, preparación, entrenamiento)
│   ├── indices.py         # Cálculo de índices espectrales
│   └── requirements.txt
├── frontend/
│   ├── app.py             # Interfaz Streamlit
│   ├── api.py             # Cliente HTTP del backend
│   └── requirements.txt
└── README.md
```
