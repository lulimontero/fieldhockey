import json
import os
import tempfile
import shutil
from typing import Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .video_analyzer import analyze_video_by_colors

app = FastAPI(title="Hockey Analyzer MVP")

# CORS: permitir desde cualquier origen en MVP
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir frontend estático
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")

@app.get("/")
async def index():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)

@app.get("/api/health")
async def health():
    return JSONResponse(content={"status": "ok"})


def _download_youtube_temp(youtube_url: str) -> str:
    import yt_dlp  # lazy import

    temp_dir = tempfile.mkdtemp(prefix="yt_")
    output_template = os.path.join(temp_dir, "video.%(ext)s")
    ydl_opts = {
        "outtmpl": output_template,
        "format": "mp4[height<=360]/best[height<=360]/mp4/bestvideo+bestaudio/best",
        "quiet": True,
        "noprogress": True,
        "merge_output_format": "mp4",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)
    # Normalizar extensión mp4 si hay merge
    mp4_candidate = filename
    if not mp4_candidate.lower().endswith(".mp4"):
        base, _ = os.path.splitext(filename)
        alt = base + ".mp4"
        if os.path.exists(alt):
            mp4_candidate = alt
    return mp4_candidate


@app.post("/api/analyze")
async def analyze_endpoint(
    youtube_url: Optional[str] = Form(default=None),
    team_color_map: Optional[str] = Form(default=None),  # JSON string {team: "#rrggbb"}
    selected_team: Optional[str] = Form(default=None),
    selected_number: Optional[str] = Form(default=None),
    # Calibración de cancha y círculo
    field_orientation: Optional[str] = Form(default="vertical"),  # "vertical" o "horizontal"
    half_offset_pct: Optional[float] = Form(default=0.0),          # desplazamiento del corte de mitad (-0.3..0.3)
    circle_side: Optional[str] = Form(default=None),               # "top"/"bottom" (vertical) o "left"/"right" (horizontal)
    circle_band_pct: Optional[float] = Form(default=0.18),         # tamaño de banda del círculo (0.1..0.3)
    circle_threshold_fraction: Optional[float] = Form(default=0.002),
    # Calibración OCR
    ocr_zoom: Optional[float] = Form(default=1.6),
    ocr_sensitivity: Optional[float] = Form(default=0.5),
    # Autocalibración
    auto_calibrate: Optional[bool] = Form(default=True),
    video_file: Optional[UploadFile] = File(default=None),
):
    if not youtube_url and not video_file:
        raise HTTPException(status_code=400, detail="Proveer youtube_url o video_file")

    # Parsear colores de equipos
    team_hex_by_name: Dict[str, str] = {}
    if team_color_map:
        try:
            team_hex_by_name = json.loads(team_color_map)
            if not isinstance(team_hex_by_name, dict):
                raise ValueError
        except Exception:
            raise HTTPException(status_code=400, detail="team_color_map debe ser JSON {equipo: '#rrggbb'}")

    temp_path = None
    temp_dir = None
    try:
        if youtube_url:
            try:
                temp_path = _download_youtube_temp(youtube_url)
                temp_dir = os.path.dirname(temp_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error descargando YouTube: {e}")
        else:
            # Guardar archivo subido a un temporal
            temp_dir = tempfile.mkdtemp(prefix="upload_")
            extension = os.path.splitext(video_file.filename or "video.mp4")[1] or ".mp4"
            temp_path = os.path.join(temp_dir, f"video{extension}")
            with open(temp_path, "wb") as f:
                f.write(await video_file.read())

        # Ejecutar análisis (valores más rápidos por defecto)
        analysis = analyze_video_by_colors(
            video_path=temp_path,
            team_hex_by_name=team_hex_by_name,
            frame_stride=15,           # muestrear 1 cada 15 frames
            max_frames=450,            # ~30s a 15 fps efectivos
            area_threshold_fraction=0.001,
            field_orientation=field_orientation,
            half_offset_pct=half_offset_pct,
            circle_side=circle_side,
            circle_band_pct=circle_band_pct,
            circle_threshold_fraction=circle_threshold_fraction,
            selected_team=selected_team,
            selected_number=selected_number,
            ocr_zoom=float(ocr_zoom) if ocr_zoom is not None else 1.6,
            ocr_sensitivity=float(ocr_sensitivity) if ocr_sensitivity is not None else 0.5,
            auto_calibrate=bool(auto_calibrate) if auto_calibrate is not None else True,
        )

        return JSONResponse(
            content={
                "teams": analysis["teams"],
                "metadata": analysis["metadata"],
                "player_metrics": analysis.get("player"),
            }
        )
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
