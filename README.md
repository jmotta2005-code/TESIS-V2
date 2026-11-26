# README – Sistema de análisis ambiental

## 1. Requisitos
- Windows.
- Python instalado.
- pip.

Verificar instalación:

```powershell
py --version
```

## 2. Instalar dependencias

Abrir dos terminales.

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

## 3. Ejecutar el sistema

**Primero: iniciar el backend**

Desde la carpeta `backend`:

```powershell
py -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Verificar que aparezca `Application startup complete`.

**Segundo: iniciar el frontend**

Desde la carpeta `frontend`, en la segunda terminal:

```powershell
py -m streamlit run app.py
```

## 4. Acceder a la aplicación

Abrir en el navegador:

- Aplicación: http://localhost:8501
- API: http://127.0.0.1:8000/docs

## 5. Importante
- Mantener abiertas ambas terminales.
- Ejecutar primero el backend y después el frontend.
- No ejecutar individualmente los archivos auxiliares.
- Si se cierra alguna terminal, el servicio correspondiente dejará de funcionar.</WritingBlock}


