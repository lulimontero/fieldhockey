# Hockey Analyzer MVP

MVP para analizar videos de hockey por colores de camisetas. Permite:
- Cargar un enlace de YouTube o subir un archivo de video
- Mapear colores de camisetas a cada equipo
- Ver métricas por equipo (presencia, dominancia, shares, tercios, mitades, círculo)
- (Opcional) Detectar dorsal de una jugadora con OCR y sumar métricas básicas

Nota: Dorsales vía OCR son sensibles a calidad, contraste y distancia de cámara.

## Requisitos
- Python 3.10+
- macOS (funciona en otros sistemas con dependencias de OpenCV)
- ffmpeg (para YouTube y códecs)
- (Opcional OCR) Tesseract OCR instalado en el sistema

## Instalación
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Si `opencv-python` falla, instalá dependencias del sistema (en macOS con Homebrew):
```bash
brew install ffmpeg
# Para OCR de dorsales:
brew install tesseract
```

## Ejecutar
```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```
Abrí `http://127.0.0.1:8000/` en el navegador.

## Uso
1. Elegí método: YouTube o Archivo
2. Ingresá URL o seleccioná archivo
3. Escribí el nombre y color de cada equipo
4. Calibrá: orientación de cancha, offset de mitad, lado y banda del círculo
5. (Opcional) Elegí equipo y dorsal para OCR
6. Presioná "Analizar"

## Limitaciones del MVP
- Segmentación simple por color (HSV); sensible a iluminación y césped
- Mitades y círculo se estiman por cortes rectos (no homografía)
- OCR de dorsales es básico (pytesseract) y puede fallar con números muy pequeños o borrosos

## Estructura
```
app/
  main.py            # API FastAPI y serving de estáticos
  video_analyzer.py  # Analizador de video por color + OCR dorsal
static/
  index.html         # Frontend simple
requirements.txt
README.md
```
