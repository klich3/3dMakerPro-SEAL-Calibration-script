#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_calib_writer.py
--------------------
Script para:
1) Clonar la **estructura** del archivo de calibración SEAL/SEAL Lite suministrado (plantilla)
   y generar un nuevo archivo con la misma estructura y metadatos actualizados.
   Se reemplazan los parámetros intrínsecos (fx, fy, cx, cy) y distorsiones (k1..k5) en una línea configurable.

⚠️ Nota importante
- El archivo de calibración del fabricante incluye parámetros adicionales (extrínsecos, LUTs, etc.).
  Este script preserva la estructura y **solo actualiza** los campos que podemos estimar con OpenCV, dejando
  el resto tal cual aparece en la plantilla. Si necesitas recalcular esos campos avanzados (p. ej. proyector),
  tendrás que implementar un pipeline de luz estructurada (Gray code / phase shifting).
- Debes tener instalado: opencv-python, numpy.

Uso (ejemplos)
--------------
# Solo clonar estructura y actualizar metadatos (sin recalibrar):
python seal_calib_writer.py --template calibJMS1006207.txt --out calib_CLON.txt --dev-id JMSNEW0002 --no-calibrate

Parámetros clave
----------------
--intrinsics-line-idx: Índice (1-based) de la línea donde insertar fx, fy, cx, cy, k1..k5.
   Por defecto 5 (coincide con el archivo de ejemplo).
--no-calibrate: Si se especifica, no se ejecuta OpenCV y se copian los valores tal cual la plantilla
   (solo se actualizan metadatos/fecha y, opcionalmente, resolución si la proporcionas con --override-res).
--override-res: Forzar resolución "W H", útil si no calibras (p.ej., "--override-res 1280 720").
"""

import argparse
import re
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None


FLOAT_FMT = "{:.6f}"  # Formato para escribir floats, como en la plantilla


def replace_leading_floats_preserving_layout(line: str, new_values: List[float], max_replace: Optional[int] = None) -> str:
    """
    Reemplaza los primeros N números (floats/ints con signo) en 'line' por los de new_values,
    preservando los espacios y el resto del contenido (por ejemplo, literal '...' en la plantilla).
    """
    pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")
    matches = list(pattern.finditer(line))

    n_to_replace = len(new_values) if max_replace is None else min(len(new_values), max_replace)
    if n_to_replace == 0 or not matches:
        return line

    out = []
    last_idx = 0
    for i, m in enumerate(matches):
        if i >= n_to_replace:
            break
        out.append(line[last_idx : m.start()])
        out.append(FLOAT_FMT.format(float(new_values[i])))
        last_idx = m.end()
    out.append(line[last_idx:])  # resto de la línea sin modificar
    return "".join(out)


def update_metadata_line(old_line: str, dev_id: Optional[str], soft_version: Optional[str]) -> str:
    """
    Mantiene el 'Type:...' de la plantilla. Actualiza DevID y CalibrateDate.
    Si soft_version es None, intenta conservar la de la plantilla.
    """
    # Extraer Type:xxxx
    type_match = re.search(r"Type:([^\*]+)", old_line)
    type_val = type_match.group(1) if type_match else "Factory-12"

    # Extraer SoftVersion si no se pasa
    if soft_version is None:
        sv_match = re.search(r"SoftVersion:([^\s\*]+)", old_line)
        soft_version = sv_match.group(1) if sv_match else "3.0.0.1116"

    # DevID
    if dev_id is None:
        dev_match = re.search(r"DevID:([^\*]+)", old_line)
        dev_id = dev_match.group(1) if dev_match else "UNKNOWN"

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"***DevID:{dev_id}***CalibrateDate:{now}***Type:{type_val}***SoftVersion:{soft_version}"


def write_new_calibration(
    template_path: str,
    out_path: str,
    intrinsics_line_idx: int,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
    k3: float = 0.0,
    override_res: Optional[Tuple[int, int]] = None,
    dev_id: Optional[str] = None,
    soft_version: Optional[str] = None,
) -> None:
    """
    Clona la estructura del archivo y reemplaza:
    - Línea 1: resolución, si override_res se especifica.
    - Línea intrinsics_line_idx: los primeros 9 números por [fx, fy, cx, cy, k1, k2, p1, p2, k3].
    - Última línea (metadatos): DevID, CalibrateDate, Type (conservado), SoftVersion (conservado o actualizado).
    """
    with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    if not lines:
        raise ValueError("La plantilla está vacía.")

    # 1) Resolución (línea 1): "W H"
    if override_res is not None:
        w, h = override_res
        lines[0] = f"{int(w)} {int(h)}"

    # 2) Intrínsecos + distorsión
    if fx is not None and fy is not None and cx is not None and cy is not None:
        intrinsics_vals = [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        li = intrinsics_line_idx - 1  # a índice 0-based
        if li < 0 or li >= len(lines):
            raise IndexError(f"intrinsics_line_idx {intrinsics_line_idx} fuera de rango")
        lines[li] = replace_leading_floats_preserving_layout(lines[li], intrinsics_vals, max_replace=len(intrinsics_vals))

    # 3) Metadatos (última línea no vacía)
    # Buscar desde el final la primera línea no vacía que empiece por "***DevID:"
    last_non_empty_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip():
            last_non_empty_idx = i
            break
    if last_non_empty_idx is None:
        raise ValueError("No se encontró una línea de metadatos en la plantilla.")

    if lines[last_non_empty_idx].startswith("***DevID:"):
        lines[last_non_empty_idx] = update_metadata_line(lines[last_non_empty_idx], dev_id=dev_id, soft_version=soft_version)
    else:
        # Si no existe, la añadimos al final
        lines.append(update_metadata_line("", dev_id=dev_id, soft_version=soft_version))

    # Escribir
    out_text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_text)


def main():
    ap = argparse.ArgumentParser(description="Reproducir/actualizar calibración y generar archivo con estructura idéntica.")
    ap.add_argument("--template", type=str, required=True, help="Ruta de la plantilla (archivo de calibración original).")
    ap.add_argument("--out", type=str, required=True, help="Ruta de salida para el nuevo archivo.")

    ap.add_argument("--intrinsics-line-idx", type=int, default=5, help="Línea 1-based donde insertar fx,fy,cx,cy,k1..k3.")
    ap.add_argument("--no-calibrate", action="store_true", help="No ejecutar calibración; solo clonar estructura.")

    ap.add_argument("--dev-id", type=str, default=None, help="Nuevo DevID para metadatos.")
    ap.add_argument("--soft-version", type=str, default=None, help="SoftVersion para metadatos (por defecto se preserva).")

    ap.add_argument("--override-res", type=int, nargs=2, default=None, metavar=("W", "H"), help="Forzar resolución W H.")

    args = ap.parse_args()

    fx = fy = cx = cy = None
    k1 = k2 = p1 = p2 = k3 = 0.0
    override_res = None

    if not args.no_calibrate:
        ap.error("Este script solo funciona con --no-calibrate. Para calibración, usa stereo_calibration.py")
    else:
        if args.override_res is not None:
            override_res = (args.override_res[0], args.override_res[1])

    write_new_calibration(
        template_path=args.template,
        out_path=args.out,
        intrinsics_line_idx=args.intrinsics_line_idx,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        k1=k1,
        k2=k2,
        p1=p1,
        p2=p2,
        k3=k3,
        override_res=override_res,
        dev_id=args.dev_id,
        soft_version=args.soft_version,
    )

    print(f"[OK] Archivo generado en: {args.out}")


if __name__ == "__main__":
    main()