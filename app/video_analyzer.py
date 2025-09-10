from typing import Dict, Tuple, Any, Optional
import cv2
import numpy as np


def _hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Color debe ser #RRGGBB")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def _bgr_to_hsv(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    color = np.uint8([[list(bgr)]])
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0][0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def _hue_range(center_h: int, delta: int = 15) -> Tuple[int, int]:
    low = max(0, center_h - delta)
    high = min(179, center_h + delta)
    return low, high


def _hex_to_hsv_range(hex_color: str) -> Tuple[np.ndarray, np.ndarray]:
    bgr = _hex_to_bgr(hex_color)
    h, s, v = _bgr_to_hsv(bgr)
    h_low, h_high = _hue_range(h, 12)
    # Rango amplio en S y V para tolerar iluminación
    lower = np.array([h_low, max(40, s // 4), max(40, v // 4)], dtype=np.uint8)
    upper = np.array([h_high, 255, 255], dtype=np.uint8)
    return lower, upper


def _ocr_detect_digits_multi(img_bgr: np.ndarray, zoom: float = 1.6, extra_passes: int = 0) -> list:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return []

    results = []

    def run_ocr(gray_img: np.ndarray, invert: bool, use_adaptive: bool, psm: int) -> str:
        img = gray_img.copy()
        if invert:
            img = cv2.bitwise_not(img)
        if use_adaptive:
            th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 31, 10)
        else:
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Morfología para unir trazos
        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        pil_img = Image.fromarray(th)
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
        try:
            text = pytesseract.image_to_string(pil_img, config=config)
        except Exception:
            return ""
        return (text or "").strip()

    h, w = img_bgr.shape[:2]
    z = max(1.0, min(3.0, float(zoom or 1.6)))
    if z != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * z), int(h * z)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)

    psm_modes = [7, 6]
    # Si extra_passes > 0, agregar psm 8 y 10
    if extra_passes > 0:
        psm_modes.extend([8, 10])

    for invert in (False, True):
        for use_adaptive in (True, False):
            for psm in psm_modes:
                txt = run_ocr(gray, invert=invert, use_adaptive=use_adaptive, psm=psm)
                if txt:
                    digits = "".join(ch for ch in txt if ch.isdigit())
                    if digits:
                        results.append(digits)
    # Devolver únicos preservando orden
    unique = []
    seen = set()
    for r in results:
        if r not in seen:
            unique.append(r)
            seen.add(r)
    return unique


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0, (bx1 - bx0) * (by1 - by0))
    union = max(1, area_a + area_b - inter)
    return inter / float(union)


def _expand_bbox(x: int, y: int, w: int, h: int, pad: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width, x + w + pad)
    y1 = min(height, y + h + pad)
    return x0, y0, x1, y1


def analyze_video_by_colors(
    video_path: str,
    team_hex_by_name: Dict[str, str],
    frame_stride: int = 10,
    max_frames: int = 1500,
    area_threshold_fraction: float = 0.001,
    # Calibración de mitades y círculo
    field_orientation: str = "vertical",     # "vertical" (mitades superior/inferior) o "horizontal" (izquierda/derecha)
    half_offset_pct: float = 0.0,             # desplaza la línea media: -0.3..0.3
    circle_side: Optional[str] = None,        # "top"/"bottom" o "left"/"right" según orientación
    circle_band_pct: float = 0.18,            # ancho de banda del círculo (0.1..0.3)
    circle_threshold_fraction: float = 0.002, # umbral de presencia en la banda
    # Jugadora seleccionada (OCR dorsal)
    selected_team: Optional[str] = None,
    selected_number: Optional[str] = None,
    # Calibración OCR
    ocr_zoom: float = 1.6,
    ocr_sensitivity: float = 0.5,
    # Autocalibración
    auto_calibrate: bool = True,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir el video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_area = max(1, width * height)

    # Rango de colores por equipo
    hsv_ranges_by_team: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for team, hex_color in team_hex_by_name.items():
        try:
            hsv_ranges_by_team[team] = _hex_to_hsv_range(hex_color)
        except Exception:
            continue

    # Autocalibración (usar máscaras combinadas de todos los equipos)
    auto_info = None
    if auto_calibrate and hsv_ranges_by_team:
        kernel_auto = np.ones((5, 5), np.uint8)
        sampled = 0
        row_sum_acc = np.zeros((height,), dtype=np.float64)
        col_sum_acc = np.zeros((width,), dtype=np.float64)
        top_band_sum = 0.0
        bottom_band_sum = 0.0
        left_band_sum = 0.0
        right_band_sum = 0.0
        band_pct_probe = 0.18

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame0 = cap.read()
            if not ret:
                break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % max(1, frame_stride * 2) != 0:
                continue
            hsv0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2HSV)
            # Máscara combinada
            combined = np.zeros((height, width), dtype=np.uint8)
            for (lower, upper) in hsv_ranges_by_team.values():
                m = cv2.inRange(hsv0, lower, upper)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel_auto)
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_auto)
                combined = cv2.bitwise_or(combined, m)

            row_sum = combined.sum(axis=1).astype(np.float64)
            col_sum = combined.sum(axis=0).astype(np.float64)
            row_sum_acc += row_sum
            col_sum_acc += col_sum

            band_h = int(round(height * band_pct_probe))
            band_w = int(round(width * band_pct_probe))
            top_band_sum += float(combined[:band_h, :].sum())
            bottom_band_sum += float(combined[height - band_h: height, :].sum())
            left_band_sum += float(combined[:, :band_w].sum())
            right_band_sum += float(combined[:, width - band_w: width].sum())

            sampled += 1
            if sampled >= 20:
                break

        # Decidir orientación por varianza acumulada
        var_rows = float(np.var(row_sum_acc)) if sampled > 0 else 0.0
        var_cols = float(np.var(col_sum_acc)) if sampled > 0 else 0.0
        guessed_orientation = "vertical" if var_rows >= var_cols else "horizontal"

        # Línea de mitad por mediana de distribución acumulada
        if guessed_orientation == "vertical":
            cumsum = np.cumsum(row_sum_acc)
            total = cumsum[-1] if cumsum.size > 0 else 1.0
            idx = int(np.searchsorted(cumsum, total * 0.5)) if total > 0 else height // 2
            guessed_half_offset = (idx / float(height)) - 0.5
            # Círculo lado (top/bottom)
            guessed_circle_side = "top" if top_band_sum >= bottom_band_sum else "bottom"
        else:
            cumsum = np.cumsum(col_sum_acc)
            total = cumsum[-1] if cumsum.size > 0 else 1.0
            idx = int(np.searchsorted(cumsum, total * 0.5)) if total > 0 else width // 2
            guessed_half_offset = (idx / float(width)) - 0.5
            # Círculo lado (left/right)
            guessed_circle_side = "left" if left_band_sum >= right_band_sum else "right"

        auto_info = {
            "orientation": guessed_orientation,
            "half_offset_pct": float(max(-0.49, min(0.49, guessed_half_offset))),
            "circle_side": guessed_circle_side,
            "circle_band_pct": band_pct_probe,
        }

        # Resetear posición para análisis principal
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Sobrescribir parámetros
        field_orientation = auto_info["orientation"]
        half_offset_pct = auto_info["half_offset_pct"]
        circle_side = auto_info["circle_side"]
        circle_band_pct = auto_info["circle_band_pct"]

    # Tercios (verticales)
    third_starts = [0, height // 3, (2 * height) // 3]
    third_stops = [height // 3, (2 * height) // 3, height]
    thirds_labels = ["tercio_superior", "tercio_medio", "tercio_inferior"]
    thirds_areas = [max(1, (third_stops[i] - third_starts[i]) * width) for i in range(3)]

    # Mitades (según orientación final)
    orientation = (field_orientation or "vertical").lower()
    if orientation not in ("vertical", "horizontal"):
        orientation = "vertical"

    if orientation == "vertical":
        line = int(round(height * (0.5 + max(-0.49, min(0.49, half_offset_pct)))))
        line = max(1, min(height - 1, line))
        halves_labels = ["mitad_superior", "mitad_inferior"]
        halves_bounds = [(0, line), (line, height)]  # (y0, y1)
        halves_areas = [max(1, (y1 - y0) * width) for (y0, y1) in halves_bounds]
    else:
        line = int(round(width * (0.5 + max(-0.49, min(0.49, half_offset_pct)))))
        line = max(1, min(width - 1, line))
        halves_labels = ["mitad_izquierda", "mitad_derecha"]
        halves_bounds = [(0, line), (line, width)]  # (x0, x1)
        halves_areas = [max(1, (x1 - x0) * height) for (x0, x1) in halves_bounds]

    # Círculo (banda aproximada)
    circle_band = None  # (axis, start, stop)
    circle_area = None
    circle_side_norm = (circle_side or "").lower() if circle_side else None
    band_pct = max(0.05, min(0.5, float(circle_band_pct or 0.18)))
    if circle_side_norm:
        if orientation == "vertical" and circle_side_norm in ("top", "bottom"):
            band = int(round(height * band_pct))
            if circle_side_norm == "top":
                circle_band = ("y", 0, min(height, band))
            else:
                circle_band = ("y", max(0, height - band), height)
            circle_area = max(1, (circle_band[2] - circle_band[1]) * width)
        elif orientation == "horizontal" and circle_side_norm in ("left", "right"):
            band = int(round(width * band_pct))
            if circle_side_norm == "left":
                circle_band = ("x", 0, min(width, band))
            else:
                circle_band = ("x", max(0, width - band), width)
            circle_area = max(1, (circle_band[2] - circle_band[1]) * height)

    metrics: Dict[str, Any] = {}
    for team in hsv_ranges_by_team.keys():
        metrics[team] = {
            "frames_with_presence": 0,
            "sum_area_fraction": 0.0,
            "frames_dominated": 0,
            "sum_relative_share": 0.0,
            "thirds": {
                thirds_labels[i]: {
                    "sum_area_fraction": 0.0,
                    "frames_with_presence": 0,
                }
                for i in range(3)
            },
            "halves": {
                halves_labels[i]: {
                    "sum_area_fraction": 0.0,
                    "frames_with_presence": 0,
                    "frames_dominated": 0,
                    "sum_relative_share": 0.0,
                }
                for i in range(2)
            },
            "circle": {
                "enabled": circle_band is not None,
                "frames_in_circle": 0,
                "entries": 0,
                "sum_area_fraction": 0.0,
            },
        }

    # Estado jugador y tracking simple
    want_player = bool(selected_team and selected_number and selected_team in hsv_ranges_by_team)
    player = {
        "team": selected_team,
        "number": selected_number,
        "frames_detected": 0,
        "frames_ocr_attempted": 0,
        "dwell_seconds": 0.0,
        "avg_bbox_area_fraction": 0.0,
        "entries_circle": 0,
        "frames_in_circle": 0,
        "recognized_number_observed": None,
        "confidence": 0.0,
        "notes": None,
    } if want_player else None
    last_player_in_circle = False
    last_bbox: Optional[Tuple[int, int, int, int]] = None  # (x0,y0,x1,y1)
    recognized_counts: Dict[str, int] = {}

    # Umbrales dependientes de sensibilidad OCR
    sens = max(0.0, min(1.0, float(ocr_sensitivity or 0.5)))
    min_box_area = int(150 * max(0.4, 1.0 - (sens - 0.5)))
    max_box_area_fraction = float(0.12 * min(1.8, 1.0 + (sens)))
    aspect_slack = 0.5 * (sens)

    frame_index = 0
    sampled_frames = 0
    frames_considered_for_share = 0
    frames_tie = 0
    frames_considered_for_half_share = [0, 0]
    frames_half_tie = [0, 0]

    last_in_circle = {team: False for team in hsv_ranges_by_team.keys()}

    kernel = np.ones((5, 5), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_stride != 0:
            frame_index += 1
            continue
        sampled_frames += 1
        frame_index += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        per_team_area = {}
        per_team_area_thirds = {team: [0, 0, 0] for team in hsv_ranges_by_team.keys()}
        per_team_area_halves = {team: [0, 0] for team in hsv_ranges_by_team.keys()}
        per_team_area_circle = {team: 0 for team in hsv_ranges_by_team.keys()}

        masks_by_team: Dict[str, np.ndarray] = {}
        for team, (lower, upper) in hsv_ranges_by_team.items():
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masks_by_team[team] = mask

            area = int(cv2.countNonZero(mask))
            per_team_area[team] = area

            for i in range(3):
                r0, r1 = third_starts[i], third_stops[i]
                sub = mask[r0:r1, :]
                per_team_area_thirds[team][i] = int(cv2.countNonZero(sub))

            if orientation == "vertical":
                for i in range(2):
                    y0, y1 = halves_bounds[i]
                    sub = mask[y0:y1, :]
                    per_team_area_halves[team][i] = int(cv2.countNonZero(sub))
            else:
                for i in range(2):
                    x0, x1 = halves_bounds[i]
                    sub = mask[:, x0:x1]
                    per_team_area_halves[team][i] = int(cv2.countNonZero(sub))

            if circle_band is not None:
                axis, a0, a1 = circle_band
                if axis == "y":
                    csub = mask[a0:a1, :]
                else:
                    csub = mask[:, a0:a1]
                per_team_area_circle[team] = int(cv2.countNonZero(csub))

        total_area_all_teams = float(sum(per_team_area.values()))
        if total_area_all_teams > 0.0:
            frames_considered_for_share += 1
            sorted_areas = sorted(per_team_area.items(), key=lambda kv: kv[1], reverse=True)
            if len(sorted_areas) >= 2 and (sorted_areas[0][1] - sorted_areas[1][1]) <= 1:
                frames_tie += 1
            elif sorted_areas and sorted_areas[0][1] > 0:
                metrics[sorted_areas[0][0]]["frames_dominated"] += 1
            for team, area_v in per_team_area.items():
                rel = (float(area_v) / total_area_all_teams) if total_area_all_teams > 0 else 0.0
                metrics[team]["sum_relative_share"] += rel

        for hidx in range(2):
            total_half_area = float(sum(per_team_area_halves[t][hidx] for t in per_team_area_halves.keys()))
            if total_half_area > 0.0:
                frames_considered_for_half_share[hidx] += 1
                sorted_half = sorted(((t, per_team_area_halves[t][hidx]) for t in per_team_area_halves.keys()), key=lambda kv: kv[1], reverse=True)
                if len(sorted_half) >= 2 and (sorted_half[0][1] - sorted_half[1][1]) <= 1:
                    frames_half_tie[hidx] += 1
                elif sorted_half and sorted_half[0][1] > 0:
                    metrics[sorted_half[0][0]]["halves"][halves_labels[hidx]]["frames_dominated"] += 1
                for team in per_team_area_halves.keys():
                    rel_h = float(per_team_area_halves[team][hidx]) / total_half_area
                    metrics[team]["halves"][halves_labels[hidx]]["sum_relative_share"] += rel_h

        for team, area in per_team_area.items():
            area_fraction = float(area) / float(frame_area)
            metrics[team]["sum_area_fraction"] += area_fraction
            if area_fraction >= area_threshold_fraction:
                metrics[team]["frames_with_presence"] += 1

            for i in range(3):
                third_area = per_team_area_thirds[team][i]
                third_fraction = float(third_area) / float(thirds_areas[i])
                label = thirds_labels[i]
                metrics[team]["thirds"][label]["sum_area_fraction"] += third_fraction
                if third_fraction >= area_threshold_fraction:
                    metrics[team]["thirds"][label]["frames_with_presence"] += 1

            for i in range(2):
                h_area = per_team_area_halves[team][i]
                h_fraction = float(h_area) / float(halves_areas[i])
                h_label = halves_labels[i]
                metrics[team]["halves"][h_label]["sum_area_fraction"] += h_fraction
                if h_fraction >= area_threshold_fraction:
                    metrics[team]["halves"][h_label]["frames_with_presence"] += 1

            if circle_band is not None and circle_area is not None:
                c_area = per_team_area_circle[team]
                c_fraction = float(c_area) / float(circle_area)
                metrics[team]["circle"]["sum_area_fraction"] += c_fraction
                in_circle_now = c_fraction >= circle_threshold_fraction
                if in_circle_now:
                    metrics[team]["circle"]["frames_in_circle"] += 1
                if in_circle_now and not last_in_circle[team]:
                    metrics[team]["circle"]["entries"] += 1
                last_in_circle[team] = in_circle_now

        # OCR de dorsal robusta (multi-pass + tracking simple)
        if want_player:
            mask_sel = masks_by_team[selected_team]
            contours, _ = cv2.findContours(mask_sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 12 or h < 12:
                    continue
                aspect = w / float(max(1, h))
                if aspect < (0.3 - aspect_slack) or aspect > (3.5 + aspect_slack):
                    continue
                area_bbox = w * h
                if area_bbox < min_box_area or area_bbox > (frame_area * max_box_area_fraction):
                    continue
                x0, y0, x1, y1 = _expand_bbox(x, y, w, h, pad=int(0.1 * max(w, h)), width=width, height=height)
                cand = (x0, y0, x1, y1)
                iou = _bbox_iou(cand, last_bbox) if last_bbox is not None else 0.0
                score = iou * 2.0 + ((x1 - x0) * (y1 - y0)) / float(frame_area)
                candidates.append((score, cand))
            candidates.sort(key=lambda t: t[0], reverse=True)

            detected_this_frame = False
            best_bbox = None
            observed_digits = None

            tried_ocr = False
            extra_passes = 1 if sens >= 0.6 else 0
            for _, (x0, y0, x1, y1) in candidates[:8]:
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                tried_ocr = True
                results = _ocr_detect_digits_multi(crop, zoom=ocr_zoom, extra_passes=extra_passes)
                if not results:
                    continue
                for digits in results:
                    if str(selected_number) == digits or str(selected_number) in digits:
                        detected_this_frame = True
                        observed_digits = digits
                        best_bbox = (x0, y0, x1, y1)
                        break
                if detected_this_frame:
                    break

            if not detected_this_frame and last_bbox is not None:
                x0, y0, x1, y1 = last_bbox
                pad = int(0.15 * max(x1 - x0, y1 - y0))
                x0, y0, x1, y1 = _expand_bbox(x0, y0, x1 - x0, y1 - y0, pad=pad, width=width, height=height)
                crop = frame[y0:y1, x0:x1]
                if crop.size > 0:
                    tried_ocr = True
                    results = _ocr_detect_digits_multi(crop, zoom=ocr_zoom, extra_passes=extra_passes)
                    if results:
                        for digits in results:
                            if str(selected_number) == digits or str(selected_number) in digits:
                                detected_this_frame = True
                                observed_digits = digits
                                best_bbox = (x0, y0, x1, y1)
                                break

            if tried_ocr:
                player["frames_ocr_attempted"] += 1

            if detected_this_frame and best_bbox is not None:
                player["frames_detected"] += 1
                bbox_area = (best_bbox[2] - best_bbox[0]) * (best_bbox[3] - best_bbox[1])
                player["avg_bbox_area_fraction"] += (bbox_area / float(frame_area)) if bbox_area > 0 else 0.0
                last_bbox = best_bbox
                if observed_digits:
                    recognized_counts[observed_digits] = recognized_counts.get(observed_digits, 0) + 1
                if circle_band is not None:
                    axis, a0, a1 = circle_band
                    cx = (best_bbox[0] + best_bbox[2]) // 2
                    cy = (best_bbox[1] + best_bbox[3]) // 2
                    in_circ = (cy >= a0 and cy < a1) if axis == "y" else (cx >= a0 and cx < a1)
                    if in_circ:
                        player["frames_in_circle"] += 1
                    if in_circ and not last_player_in_circle:
                        player["entries_circle"] += 1
                    last_player_in_circle = in_circ
            else:
                last_player_in_circle = False
                if last_bbox is not None and sampled_frames % 10 == 0:
                    last_bbox = None

        if sampled_frames >= max_frames:
            break

    cap.release()

    sum_area_fraction_all_teams = sum(m["sum_area_fraction"] for m in metrics.values()) or 1e-12

    for team, data in metrics.items():
        presence_seconds = (data["frames_with_presence"] * frame_stride) / float(fps)
        avg_area_fraction = (data["sum_area_fraction"] / max(1, sampled_frames))
        overall_share = data["sum_area_fraction"] / sum_area_fraction_all_teams
        avg_relative_share_per_frame = data["sum_relative_share"] / max(1, frames_considered_for_share)

        thirds_out = {}
        for i, label in enumerate(thirds_labels):
            t_presence_seconds = (data["thirds"][label]["frames_with_presence"] * frame_stride) / float(fps)
            t_avg_area_fraction = data["thirds"][label]["sum_area_fraction"] / max(1, sampled_frames)
            thirds_out[label] = {
                "frames_with_presence": data["thirds"][label]["frames_with_presence"],
                "presence_seconds": round(t_presence_seconds, 2),
                "avg_area_fraction": round(t_avg_area_fraction, 6),
            }

        halves_out = {}
        for i, label in enumerate(halves_labels):
            h_presence_seconds = (data["halves"][label]["frames_with_presence"] * frame_stride) / float(fps)
            h_avg_area_fraction = data["halves"][label]["sum_area_fraction"] / max(1, sampled_frames)
            h_avg_relative_share = data["halves"][label]["sum_relative_share"] / max(1, frames_considered_for_half_share[i])
            halves_out[label] = {
                "frames_with_presence": data["halves"][label]["frames_with_presence"],
                "presence_seconds": round(h_presence_seconds, 2),
                "avg_area_fraction": round(h_avg_area_fraction, 6),
                "frames_dominated": int(data["halves"][label]["frames_dominated"]),
                "avg_relative_share_per_frame": round(float(h_avg_relative_share), 6),
            }

        circle_out = None
        if metrics[next(iter(metrics))]["circle"]["enabled"] if metrics else False:
            # Tomar enabled de cualquier equipo (comparten configuración)
            total_frames_in_circle = 0
            total_entries = 0
            sum_area_fraction_circle = 0.0
            for tdata in metrics.values():
                total_frames_in_circle += tdata["circle"]["frames_in_circle"]
                total_entries += tdata["circle"]["entries"]
                sum_area_fraction_circle += tdata["circle"]["sum_area_fraction"]
            dwell_seconds = (total_frames_in_circle * frame_stride) / float(fps)
            avg_area_fraction_circle = sum_area_fraction_circle / max(1, len(metrics) * max(1, sampled_frames))
            circle_out = {
                "frames_in_circle": total_frames_in_circle,
                "entries": total_entries,
                "dwell_seconds": round(dwell_seconds, 2),
                "avg_area_fraction": round(avg_area_fraction_circle, 6),
            }

        data.update(
            {
                "presence_seconds": round(presence_seconds, 2),
                "avg_area_fraction": round(avg_area_fraction, 6),
                "frames_dominated": int(data["frames_dominated"]),
                "relative_share_overall": round(float(overall_share), 6),
                "avg_relative_share_per_frame": round(float(avg_relative_share_per_frame), 6),
                "thirds": thirds_out,
                "halves": halves_out,
                "circle": circle_out,
            }
        )

    metadata: Dict[str, Any] = {
        "fps": fps,
        "sampled_frames": sampled_frames,
        "frame_stride": frame_stride,
        "video_size": {"width": width, "height": height},
        "thirds_bounds": {
            "tercio_superior": [third_starts[0], third_stops[0]],
            "tercio_medio": [third_starts[1], third_stops[1]],
            "tercio_inferior": [third_starts[2], third_stops[2]],
        },
        "frames_considered_for_share": frames_considered_for_share,
        "frames_tie": frames_tie,
        "field_orientation": orientation,
        "auto_calibrated": bool(auto_calibrate),
    }

    if orientation == "vertical":
        metadata.update(
            {
                "halves_bounds": {
                    "mitad_superior": [halves_bounds[0][0], halves_bounds[0][1]],
                    "mitad_inferior": [halves_bounds[1][0], halves_bounds[1][1]],
                },
                "half_line": line,
            }
        )
    else:
        metadata.update(
            {
                "halves_bounds": {
                    "mitad_izquierda": [halves_bounds[0][0], halves_bounds[0][1]],
                    "mitad_derecha": [halves_bounds[1][0], halves_bounds[1][1]],
                },
                "half_line": line,
            }
        )

    if circle_band is not None:
        axis, a0, a1 = circle_band
        metadata.update(
            {
                "circle_band": {
                    "axis": axis,
                    "start": a0,
                    "stop": a1,
                    "band_pct": band_pct,
                    "side": circle_side_norm,
                    "threshold_fraction": circle_threshold_fraction,
                }
            }
        )

    if auto_info is not None:
        metadata["auto_guess"] = auto_info

    player_out = None
    if want_player and player is not None:
        attempts = max(1, player["frames_ocr_attempted"])
        conf = float(player["frames_detected"]) / float(attempts)
        player_out = dict(player)
        player_out["dwell_seconds"] = round((player["frames_detected"] * frame_stride) / float(fps), 2)
        if player["frames_detected"] > 0:
            player_out["avg_bbox_area_fraction"] = round(player["avg_bbox_area_fraction"] / float(player["frames_detected"]), 6)
        else:
            player_out["avg_bbox_area_fraction"] = 0.0
        if recognized_counts:
            observed = max(recognized_counts.items(), key=lambda kv: kv[1])[0]
            player_out["recognized_number_observed"] = observed
        player_out["confidence"] = round(conf, 4)
        if conf < 0.2:
            player_out["notes"] = "Baja confianza: video o dorsal poco legible. Ajustá zoom o color."

    return {
        "teams": metrics,
        "metadata": metadata,
        "player": player_out,
    }
