"""Lectura y predicción sobre imágenes GeoTIFF (rásters multiespectrales).

`rasterio` se importa de forma perezosa para que el backend pueda arrancar aunque
la librería no esté instalada; las rutas de ráster devolverán entonces un error
claro pidiendo instalarla.
"""
import io

import numpy as np
import pandas as pd

import indices
import ml_core as ml

# Cuántas filas del ráster se procesan a la vez (controla el uso de memoria).
BLOQUE_FILAS = 64
# Lado máximo de la vista previa que se envía al frontend.
MAX_PREVIEW = 400


def _rasterio():
    try:
        import rasterio  # noqa: F401
        from rasterio.io import MemoryFile
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "Falta la librería 'rasterio'. Instálala con: pip install rasterio"
        ) from e
    return rasterio, MemoryFile


def _nombre_banda(desc, idx: int) -> str:
    """Nombre normalizado de una banda (descripción o, si falta, B{índice})."""
    if desc is not None and str(desc).strip():
        return str(desc).strip().upper().replace(" ", "")
    return f"B{idx}"


def _columnas_bandas(descs, n: int, mapeo=None) -> list:
    """Nombres de columna de cada banda (aplica el mapeo manual si existe)."""
    cols = []
    for i in range(1, n + 1):
        if mapeo and mapeo.get(str(i)):
            nombre = str(mapeo[str(i)])
        else:
            desc = descs[i - 1] if descs and i - 1 < len(descs) else None
            nombre = _nombre_banda(desc, i)
        cols.append(str(nombre).strip().upper().replace(" ", ""))

    # Evita nombres duplicados (rasterio exige columnas únicas en el DataFrame).
    vistos, salida = set(), []
    for i, c in enumerate(cols):
        base, k = c, 1
        while c in vistos:
            c = f"{base}_{i + 1}"
            k += 1
        vistos.add(c)
        salida.append(c)
    return salida


def _disponibles(cols) -> set:
    """Columnas que se pueden calcular a partir de las bandas (incluye índices)."""
    df = pd.DataFrame({c: np.zeros(1, dtype="float64") for c in cols})
    df, _ = indices.agregar_indices_espectrales(df)
    return {str(c) for c in df.columns}


def info_raster(contenido: bytes, features=None, mapeo=None) -> dict:
    """Metadatos del ráster y variables del modelo que faltan por cubrir."""
    rasterio, MemoryFile = _rasterio()
    with MemoryFile(contenido) as mem, mem.open() as src:
        descs = list(src.descriptions) if src.descriptions else [None] * src.count
        cols = _columnas_bandas(descs, src.count, mapeo)
        bandas = [
            {"indice": i + 1, "nombre": cols[i], "descripcion": descs[i]}
            for i in range(src.count)
        ]
        out = {
            "ancho": src.width,
            "alto": src.height,
            "n_bandas": src.count,
            "crs": src.crs.to_string() if src.crs else None,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
            "bandas": bandas,
            "columnas_detectadas": cols,
        }
    if features:
        disponibles = _disponibles(cols)
        out["faltan"] = [str(f) for f in features if str(f) not in disponibles]
    else:
        out["faltan"] = []
    return out


def _stats(salida: np.ndarray) -> dict:
    vals = salida[np.isfinite(salida)]
    if vals.size == 0:
        return {
            "min": None, "max": None, "media": None, "mediana": None, "std": None,
            "p5": None, "p25": None, "p75": None, "p95": None, "hist": None,
        }
    counts, edges = np.histogram(vals, bins=30)
    q = np.percentile(vals, [5, 25, 75, 95])
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "media": float(np.mean(vals)),
        "mediana": float(np.median(vals)),
        "std": float(np.std(vals)),
        "p5": float(q[0]),
        "p25": float(q[1]),
        "p75": float(q[2]),
        "p95": float(q[3]),
        "hist": {"counts": counts.tolist(), "edges": edges.tolist()},
    }


def predecir_raster(modelo: dict, contenido: bytes, mapeo=None) -> dict:
    """Aplica el modelo a cada píxel del GeoTIFF y devuelve el mapa de predicción.

    `modelo` = {"pipeline", "features", "target", "es_clf"}.
    `mapeo`  = {"1": "B2", "2": "B3", ...} (índice de banda -> nombre) o None.
    """
    rasterio, MemoryFile = _rasterio()
    from rasterio.windows import Window

    pipeline = modelo["pipeline"]
    features = [str(c) for c in modelo["features"]]
    es_clf = bool(modelo.get("es_clf", False))

    # Para clasificación con etiquetas de texto se codifican a enteros.
    clases, mapa_clases = None, None
    if es_clf and hasattr(pipeline, "classes_"):
        clases = list(pipeline.classes_)
        if not pd.api.types.is_numeric_dtype(pd.Series(clases)):
            mapa_clases = {c: i for i, c in enumerate(clases)}

    with MemoryFile(contenido) as mem, mem.open() as src:
        alto, ancho = src.height, src.width
        descs = list(src.descriptions) if src.descriptions else [None] * src.count
        cols = _columnas_bandas(descs, src.count, mapeo)

        salida = np.full((alto, ancho), np.nan, dtype="float32")

        for inicio in range(0, alto, BLOQUE_FILAS):
            fin = min(inicio + BLOQUE_FILAS, alto)
            ventana = Window(0, inicio, ancho, fin - inicio)
            arr = src.read(window=ventana, masked=True).astype("float64")
            arr = np.moveaxis(arr, 0, -1)  # (filas, columnas, bandas)

            datos = np.ma.filled(arr, np.nan).reshape(-1, arr.shape[-1])
            mascara = np.ma.getmaskarray(arr).reshape(-1, arr.shape[-1])
            validos = ~mascara.all(axis=1)  # píxel válido si alguna banda tiene dato

            df = pd.DataFrame(datos, columns=cols)
            df, _ = indices.agregar_indices_espectrales(df)

            faltan = [c for c in features if c not in df.columns]
            if faltan:
                raise ValueError(
                    "Faltan variables que el modelo necesita y no se pueden calcular "
                    f"desde las bandas del ráster: {faltan}. Revisa el mapeo de bandas "
                    "o usa una imagen con las bandas correctas."
                )

            pred = np.full(len(df), np.nan, dtype="float64")
            if validos.any():
                yhat = pipeline.predict(ml.preparar_X(df.loc[validos, features]))
                if mapa_clases is not None:
                    yhat = np.array([mapa_clases.get(v, np.nan) for v in yhat], dtype="float64")
                else:
                    yhat = np.asarray(yhat, dtype="float64")
                pred[validos] = yhat

            salida[inicio:fin, :] = pred.reshape(fin - inicio, ancho).astype("float32")

        perfil = {
            "driver": "GTiff",
            "height": alto,
            "width": ancho,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "nodata": np.nan,
            "compress": "deflate",
        }
        crs = src.crs.to_string() if src.crs else None

    buf = io.BytesIO()
    with MemoryFile() as mem_out:
        with mem_out.open(**perfil) as dst:
            dst.write(salida, 1)
        geotiff = mem_out.read()

    paso = max(1, int(np.ceil(max(salida.shape) / MAX_PREVIEW)))
    sub = salida[::paso, ::paso]
    preview = [
        [None if not np.isfinite(v) else float(v) for v in fila]
        for fila in sub
    ]

    return {
        "geotiff": geotiff,
        "preview": preview,
        "stats": _stats(salida),
        "ancho": ancho,
        "alto": alto,
        "crs": crs,
        "n_validos": int(np.isfinite(salida).sum()),
        "es_clf": es_clf,
        "clases": [str(c) for c in clases] if clases is not None else None,
    }
