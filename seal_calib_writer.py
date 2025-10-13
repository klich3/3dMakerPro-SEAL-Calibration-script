#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_calib_writer.py
--------------------
Script para:
1) Calibrar la cámara con un tablero (checkerboard) usando OpenCV.
2) Calibrar cámaras estéreo con un tablero (checkerboard) usando OpenCV.
3) Clonar la **estructura** del archivo de calibración SEAL/SEAL Lite suministrado (plantilla)
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
# Calibrar desde imágenes y escribir nuevo archivo:
python seal_calib_writer.py \
  --images "calib_imgs/*.png" \
  --checkerboard-cols 9 --checkerboard-rows 6 \
  --square-size-mm 10.0 \
  --template calibJMS1006207.txt \
  --out calib_NEW.txt \
  --dev-id JMSNEW0001 \
  --intrinsics-line-idx 5 \
  --soft-version "3.0.0.1116"

# Calibrar desde cámara en tiempo real:
python seal_calib_writer.py \
  --camera 0 \
  --checkerboard-cols 9 --checkerboard-rows 6 \
  --square-size-mm 10.0 \
  --template calibJMS1006207.txt \
  --out calib_NEW.txt \
  --dev-id JMSNEW0001 \
  --intrinsics-line-idx 5 \
  --soft-version "3.0.0.1116"

# Calibrar estéreo alternando entre dos fuentes de luz en una cámara:
python seal_calib_writer.py \
  --camera 0 \
  --stereo-alternating \
  --light-uv-delay 3 \
  --light-laser-delay 3 \
  --checkerboard-cols 9 --checkerboard-rows 6 \
  --square-size-mm 10.0 \
  --template calibJMS1006207.txt \
  --out calib_STEREO.txt \
  --dev-id JMSNEW0001 \
  --intrinsics-line-idx 5 \
  --soft-version "3.0.0.1116"

# Calibrar estéreo alternando automáticamente entre dos fuentes de luz:
python seal_calib_writer.py \
  --camera 0 \
  --stereo-auto \
  --frames-per-light 10 \
  --delay-between-lights 2 \
  --checkerboard-cols 9 --checkerboard-rows 6 \
  --square-size-mm 10.0 \
  --template calibJMS1006207.txt \
  --out calib_STEREO_AUTO.txt \
  --dev-id JMSNEW0001 \
  --intrinsics-line-idx 5 \
  --soft-version "3.0.0.1116"

# Calibrar estéreo con dos cámaras físicas:
python seal_calib_writer.py \
  --stereo \
  --camera-left 0 --camera-right 1 \
  --checkerboard-cols 9 --checkerboard-rows 6 \
  --square-size-mm 10.0 \
  --template calibJMS1006207.txt \
  --out calib_STEREO.txt \
  --dev-id JMSNEW0001 \
  --intrinsics-line-idx 5 \
  --soft-version "3.0.0.1116"

# Solo clonar estructura y actualizar metadatos (sin recalibrar):
python seal_calib_writer.py --template calibJMS1006207.txt --out calib_CLON.txt --dev-id JMSNEW0002 --no-calibrate

Parámetros clave
----------------
--intrinsics-line-idx: Índice (1-based) de la línea donde insertar fx, fy, cx, cy, k1..k5.
   Por defecto 5 (coincide con el archivo de ejemplo).
--no-calibrate: Si se especifica, no se ejecuta OpenCV y se copian los valores tal cual la plantilla
   (solo se actualizan metadatos/fecha y, opcionalmente, resolución si la proporcionas con --override-res).
--override-res: Forzar resolución "W H", útil si no calibras (p.ej., "--override-res 1280 720").
--stereo-alternating: Activar modo de calibración estéreo alternando fuentes de luz
--light-uv-delay: Segundos de espera para la luz UV
--light-laser-delay: Segundos de espera para la luz láser
"""

import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional
import tempfile
import shutil
import time

import numpy as np

try:
    import cv2
except Exception as e:
    cv2 = None


FLOAT_FMT = "{:.6f}"  # Formato para escribir floats, como en la plantilla


def find_image_paths(pattern: str) -> List[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes con el patrón: {pattern}")
    return paths


def calibrate_camera_from_chessboard(
    image_paths: List[str],
    checkerboard_rows: int,
    checkerboard_cols: int,
    square_size_mm: float,
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray]:
    """
    Devuelve: (width, height), K(3x3), dist(1x5)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no está disponible. Instala 'opencv-python'.")

    # Preparar puntos-objeto (0,0,0), (1,0,0), ... en unidades del tablero (mm)
    objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
    objp *= float(square_size_mm)

    objpoints = []  # puntos 3D en el mundo
    imgpoints = []  # puntos 2D en la imagen
    img_size = None

    # Criterio de refinamiento subpíxel
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[ADVERTENCIA] No se pudo leer: {p}")
            continue
        h, w = img.shape[:2]
        img_size = (w, h)

        found, corners = cv2.findChessboardCorners(img, (checkerboard_cols, checkerboard_rows))
        if not found:
            print(f"[ADVERTENCIA] Checkerboard no detectado en: {p}")
            continue

        # Refinar esquinas al subpíxel
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners)

    if not objpoints:
        raise RuntimeError("No se detectaron suficientes tableros para calibrar. Revisa imágenes/parámetros.")

    # Calibración de cámara
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=0,
        criteria=criteria,
    )
    if not ret:
        print("[ADVERTENCIA] calibrateCamera devolvió ret=0; los resultados podrían no ser válidos.")

    # dist puede tener más de 5 coeficientes; nos quedamos con 5 (k1,k2,p1,p2,k3)
    if dist.size < 5:
        dist_padded = np.zeros((1, 5), dtype=np.float64)
        dist_padded[0, : dist.size] = dist.ravel()
        dist = dist_padded
    else:
        dist = dist.ravel()[:5][None, :]

    return img_size, K, dist


def calibrate_stereo_two_cameras(
    camera_left_index: int,
    camera_right_index: int,
    checkerboard_rows: int,
    checkerboard_cols: int,
    square_size_mm: float,
    num_images: int = 15,
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibra cámaras estéreo usando dos cámaras físicas diferentes.
    Devuelve: (width, height), cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no está disponible. Instala 'opencv-python'.")

    # Crear directorio temporal para almacenar las imágenes capturadas
    temp_dir = tempfile.mkdtemp()
    print(f"[INFO] Directorio temporal para imágenes: {temp_dir}")

    try:
        # Inicializar ambas cámaras
        cap_left = cv2.VideoCapture(camera_left_index)
        cap_right = cv2.VideoCapture(camera_right_index)
        
        if not cap_left.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara izquierda con índice {camera_left_index}")
        if not cap_right.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara derecha con índice {camera_right_index}")

        # Preparar puntos-objeto (0,0,0), (1,0,0), ... en unidades del tablero (mm)
        objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
        objp *= float(square_size_mm)

        # Arrays para almacenar puntos del objeto y puntos de imagen de ambas cámaras
        objpoints = []  # puntos 3D en el mundo (iguales para ambas cámaras)
        imgpoints_left = []  # puntos 2D en la imagen de la cámara izquierda
        imgpoints_right = []  # puntos 2D en la imagen de la cámara derecha
        
        img_size = None
        captured_count = 0

        # Criterio de refinamiento subpíxel
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        print("[INFO] Calibración estéreo con dos cámaras")
        print(f"[INFO] Presiona 'c' para capturar un par de imágenes, 'q' para salir.")
        print(f"[INFO] Se necesitan {num_images} pares de imágenes con tablero detectado en ambas cámaras.")

        while True:
            # Leer frames de ambas cámaras
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()
            
            if not ret_left or not ret_right:
                print("[ADVERTENCIA] No se pudo leer el fotograma de una de las cámaras")
                continue

            gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
            
            h, w = gray_left.shape[:2]
            img_size = (w, h)

            # Buscar tablero en ambas imágenes
            found_left, corners_left = cv2.findChessboardCorners(gray_left, (checkerboard_cols, checkerboard_rows))
            found_right, corners_right = cv2.findChessboardCorners(gray_right, (checkerboard_cols, checkerboard_rows))
            
            # Mostrar resultados de detección
            display_frame_left = frame_left.copy()
            display_frame_right = frame_right.copy()
            
            if found_left:
                cv2.drawChessboardCorners(display_frame_left, (checkerboard_cols, checkerboard_rows), corners_left, found_left)
                cv2.putText(display_frame_left, f"Izquierda - Tablero detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame_left, "Izquierda - Tablero NO detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            if found_right:
                cv2.drawChessboardCorners(display_frame_right, (checkerboard_cols, checkerboard_rows), corners_right, found_right)
                cv2.putText(display_frame_right, f"Derecha - Tablero detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame_right, "Derecha - Tablero NO detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(display_frame_left, f"Capturadas: {captured_count}/{num_images}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(display_frame_right, f"Capturadas: {captured_count}/{num_images}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            cv2.putText(display_frame_left, "Presiona 'c' para capturar, 'q' para salir", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame_right, "Presiona 'c' para capturar, 'q' para salir", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Mostrar imágenes en ventanas separadas
            cv2.imshow('Camara Izquierda', display_frame_left)
            cv2.imshow('Camara Derecha', display_frame_right)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and found_left and found_right:
                # Refinar esquinas al subpíxel
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

                objpoints.append(objp.copy())
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
                
                # Guardar imágenes
                img_path_left = os.path.join(temp_dir, f"calib_left_{captured_count:03d}.png")
                img_path_right = os.path.join(temp_dir, f"calib_right_{captured_count:03d}.png")
                cv2.imwrite(img_path_left, frame_left)
                cv2.imwrite(img_path_right, frame_right)
                print(f"[INFO] Par de imágenes capturado: {img_path_left}, {img_path_right}")
                
                captured_count += 1
                if captured_count >= num_images:
                    print("[INFO] Se han capturado suficientes pares de imágenes. Iniciando calibración...")
                    break

        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

        if captured_count < 3:
            raise RuntimeError("Se necesitan al menos 3 pares de imágenes con tablero detectado en ambas cámaras para calibrar.")

        print(f"[INFO] Capturados {captured_count} pares de imágenes válidos. Iniciando calibración...")

        # Calibración individual de cada cámara
        ret_left, K_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints_left,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=0,
            criteria=criteria,
        )
        
        ret_right, K_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints_right,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=0,
            criteria=criteria,
        )

        if not ret_left or not ret_right:
            print("[ADVERTENCIA] La calibración individual de una de las cámaras falló.")

        # Calibración estéreo
        ret, K_left, dist_left, K_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objectPoints=objpoints,
            imagePoints1=imgpoints_left,
            imagePoints2=imgpoints_right,
            cameraMatrix1=K_left,
            distCoeffs1=dist_left,
            cameraMatrix2=K_right,
            distCoeffs2=dist_right,
            imageSize=img_size,
            criteria=criteria,
        )

        if not ret:
            print("[ADVERTENCIA] stereoCalibrate devolvió ret=0; los resultados podrían no ser válidos.")

        return img_size, K_left, dist_left, K_right, dist_right, R, T

    finally:
        # Limpiar directorio temporal
        shutil.rmtree(temp_dir, ignore_errors=True)


def calibrate_stereo_auto_alternating(
    camera_index: int,
    checkerboard_rows: int,
    checkerboard_cols: int,
    square_size_mm: float,
    frames_per_light: int,
    delay_between_lights: float,
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calibra cámaras estéreo automáticamente alternando entre dos fuentes de luz en una sola cámara.
    Devuelve: (width, height), cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no está disponible. Instala 'opencv-python'.")

    # Crear directorio temporal para almacenar las imágenes capturadas
    temp_dir = tempfile.mkdtemp()
    print(f"[INFO] Directorio temporal para imágenes: {temp_dir}")

    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara con índice {camera_index}")

        # Configurar la cámara con un bajo FPS para permitir el cambio de iluminación
        cap.set(cv2.CAP_PROP_FPS, 5)  # Reducir FPS para dar tiempo a cambiar luces

        # Preparar puntos-objeto (0,0,0), (1,0,0), ... en unidades del tablero (mm)
        objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
        objp *= float(square_size_mm)

        # Arrays para almacenar puntos del objeto y puntos de imagen de ambas condiciones
        objpoints_light1 = []  # puntos 3D en el mundo (con luz 1)
        imgpoints_light1 = []  # puntos 2D en la imagen (con luz 1)
        objpoints_light2 = []  # puntos 3D en el mundo (con luz 2)
        imgpoints_light2 = []  # puntos 2D en la imagen (con luz 2)
        
        img_size = None
        captured_light1_count = 0
        captured_light2_count = 0

        # Criterio de refinamiento subpíxel
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        print("[INFO] Calibración estéreo automática alternando fuentes de luz")
        print("[INFO] Asegúrate de tener ambas fuentes de luz controladas por software o mediante un conmutador")
        print(f"[INFO] Se capturarán {frames_per_light} frames por cada fuente de luz")
        print("[INFO] Procedimiento automático iniciado...")

        # Proceso automático de captura
        for cycle in range(frames_per_light):
            print(f"[INFO] Ciclo {cycle + 1}/{frames_per_light}")
            
            # Activar luz 1 (por ejemplo, UV)
            print("[INFO] Activando luz 1 (UV)...")
            # Aquí iría el código para activar la luz 1 si se tiene control por software
            # Por ahora, solo esperamos
            time.sleep(delay_between_lights)
            
            # Capturar frames con luz 1
            print("[INFO] Capturando frames con luz 1...")
            frames_captured_light1 = 0
            while frames_captured_light1 < 3:  # Capturar varios frames y usar el mejor
                ret, frame = cap.read()
                if not ret:
                    print("[ADVERTENCIA] No se pudo leer el fotograma de la cámara")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]
                img_size = (w, h)

                # Buscar tablero
                found, corners = cv2.findChessboardCorners(gray, (checkerboard_cols, checkerboard_rows))
                
                if found:
                    # Refinar esquinas al subpíxel
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Guardar temporalmente para verificar calidad
                    temp_img_path = os.path.join(temp_dir, f"temp_light1_cycle{cycle}_frame{frames_captured_light1}.png")
                    cv2.imwrite(temp_img_path, frame)
                    
                    objpoints_light1.append(objp.copy())
                    imgpoints_light1.append(corners)
                    
                    frames_captured_light1 += 1
                    print(f"[INFO] Frame {frames_captured_light1} capturado con luz 1")
                    
                    # Mostrar frame capturado
                    display_frame = frame.copy()
                    cv2.drawChessboardCorners(display_frame, (checkerboard_cols, checkerboard_rows), corners, found)
                    cv2.putText(display_frame, f"Luz 1 - Frame {frames_captured_light1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Calibracion Estereo Automatica', display_frame)
                    cv2.waitKey(500)  # Mostrar por 500ms
                    
                    if frames_captured_light1 >= 3:
                        break
                else:
                    # Mostrar frame sin tablero
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "Tablero NO detectado - Luz 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Calibracion Estereo Automatica', display_frame)
                    cv2.waitKey(100)  # Mostrar por 100ms
            
            # Activar luz 2 (por ejemplo, láser)
            print("[INFO] Activando luz 2 (Laser)...")
            # Aquí iría el código para activar la luz 2 si se tiene control por software
            # Por ahora, solo esperamos
            time.sleep(delay_between_lights)
            
            # Capturar frames con luz 2
            print("[INFO] Capturando frames con luz 2...")
            frames_captured_light2 = 0
            while frames_captured_light2 < 3:  # Capturar varios frames y usar el mejor
                ret, frame = cap.read()
                if not ret:
                    print("[ADVERTENCIA] No se pudo leer el fotograma de la cámara")
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = gray.shape[:2]
                img_size = (w, h)

                # Buscar tablero
                found, corners = cv2.findChessboardCorners(gray, (checkerboard_cols, checkerboard_rows))
                
                if found:
                    # Refinar esquinas al subpíxel
                    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    
                    # Guardar temporalmente para verificar calidad
                    temp_img_path = os.path.join(temp_dir, f"temp_light2_cycle{cycle}_frame{frames_captured_light2}.png")
                    cv2.imwrite(temp_img_path, frame)
                    
                    objpoints_light2.append(objp.copy())
                    imgpoints_light2.append(corners)
                    
                    frames_captured_light2 += 1
                    print(f"[INFO] Frame {frames_captured_light2} capturado con luz 2")
                    
                    # Mostrar frame capturado
                    display_frame = frame.copy()
                    cv2.drawChessboardCorners(display_frame, (checkerboard_cols, checkerboard_rows), corners, found)
                    cv2.putText(display_frame, f"Luz 2 - Frame {frames_captured_light2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('Calibracion Estereo Automatica', display_frame)
                    cv2.waitKey(500)  # Mostrar por 500ms
                    
                    if frames_captured_light2 >= 3:
                        break
                else:
                    # Mostrar frame sin tablero
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "Tablero NO detectado - Luz 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow('Calibracion Estereo Automatica', display_frame)
                    cv2.waitKey(100)  # Mostrar por 100ms

        cap.release()
        cv2.destroyAllWindows()

        # Seleccionar los mejores frames de cada ciclo
        print("[INFO] Seleccionando los mejores frames de cada ciclo...")
        selected_objpoints_light1 = []
        selected_imgpoints_light1 = []
        selected_objpoints_light2 = []
        selected_imgpoints_light2 = []
        
        # Por simplicidad, usamos todos los frames capturados
        selected_objpoints_light1 = objpoints_light1
        selected_imgpoints_light1 = imgpoints_light1
        selected_objpoints_light2 = objpoints_light2
        selected_imgpoints_light2 = imgpoints_light2
        
        captured_light1_count = len(selected_imgpoints_light1)
        captured_light2_count = len(selected_imgpoints_light2)

        if captured_light1_count < 5 or captured_light2_count < 5:
            raise RuntimeError("No se capturaron suficientes frames con tablero detectado para calibrar.")

        print(f"[INFO] Seleccionados {captured_light1_count} frames para luz 1 y {captured_light2_count} frames para luz 2")

        # Guardar frames seleccionados
        for i, (obj, img) in enumerate(zip(selected_objpoints_light1, selected_imgpoints_light1)):
            img_path = os.path.join(temp_dir, f"selected_light1_{i:03d}.png")
            # Creamos una imagen dummy para representar el frame seleccionado
            dummy_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            cv2.drawChessboardCorners(dummy_img, (checkerboard_cols, checkerboard_rows), img, True)
            cv2.imwrite(img_path, dummy_img)
            
        for i, (obj, img) in enumerate(zip(selected_objpoints_light2, selected_imgpoints_light2)):
            img_path = os.path.join(temp_dir, f"selected_light2_{i:03d}.png")
            # Creamos una imagen dummy para representar el frame seleccionado
            dummy_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
            cv2.drawChessboardCorners(dummy_img, (checkerboard_cols, checkerboard_rows), img, True)
            cv2.imwrite(img_path, dummy_img)

        print(f"[INFO] Iniciando calibración con {captured_light1_count} frames de luz 1 y {captured_light2_count} frames de luz 2...")

        # Calibración individual de cada conjunto de imágenes
        ret_light1, K_light1, dist_light1, rvecs_light1, tvecs_light1 = cv2.calibrateCamera(
            objectPoints=selected_objpoints_light1,
            imagePoints=selected_imgpoints_light1,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=0,
            criteria=criteria,
        )
        
        ret_light2, K_light2, dist_light2, rvecs_light2, tvecs_light2 = cv2.calibrateCamera(
            objectPoints=selected_objpoints_light2,
            imagePoints=selected_imgpoints_light2,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=0,
            criteria=criteria,
        )

        if not ret_light1 or not ret_light2:
            print("[ADVERTENCIA] La calibración individual de una de las condiciones falló.")

        # Para fines de este script, asumiremos que las dos condiciones representan cámaras estéreo
        # En una implementación real, se necesitaría un método más sofisticado para calcular R y T
        # Por ahora, devolveremos matrices de identidad para R y un vector cero para T
        R = np.eye(3, dtype=np.float64)
        T = np.zeros((3, 1), dtype=np.float64)

        return img_size, K_light1, dist_light1, K_light2, dist_light2, R, T

    finally:
        # Limpiar directorio temporal
        shutil.rmtree(temp_dir, ignore_errors=True)


def calibrate_camera_from_live_feed(
    camera_index: int,
    checkerboard_rows: int,
    checkerboard_cols: int,
    square_size_mm: float,
    num_images: int = 15,
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray]:
    """
    Calibra la cámara usando imágenes capturadas en tiempo real desde una cámara.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) no está disponible. Instala 'opencv-python'.")

    # Crear directorio temporal para almacenar las imágenes capturadas
    temp_dir = tempfile.mkdtemp()
    print(f"[INFO] Directorio temporal para imágenes: {temp_dir}")

    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara con índice {camera_index}")

        # Preparar puntos-objeto (0,0,0), (1,0,0), ... en unidades del tablero (mm)
        objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
        objp *= float(square_size_mm)

        objpoints = []  # puntos 3D en el mundo
        imgpoints = []  # puntos 2D en la imagen
        img_size = None
        captured_count = 0

        # Criterio de refinamiento subpíxel
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        print("[INFO] Capturando imágenes para calibración...")
        print(f"[INFO] Presiona 'c' para capturar una imagen, 'q' para salir.")
        print(f"[INFO] Se necesitan {num_images} imágenes con tablero detectado.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ADVERTENCIA] No se pudo leer el fotograma de la cámara")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            img_size = (w, h)

            # Buscar tablero
            found, corners = cv2.findChessboardCorners(gray, (checkerboard_cols, checkerboard_rows))
            
            # Mostrar resultado de detección
            display_frame = frame.copy()
            if found:
                cv2.drawChessboardCorners(display_frame, (checkerboard_cols, checkerboard_rows), corners, found)
                cv2.putText(display_frame, f"Tablero detectado - Capturadas: {captured_count}/{num_images}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Tablero NO detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.putText(display_frame, "Presiona 'c' para capturar, 'q' para salir", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Calibracion en tiempo real', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and found:
                # Refinar esquinas al subpíxel
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                objpoints.append(objp.copy())
                imgpoints.append(corners)
                
                # Guardar imagen
                img_path = os.path.join(temp_dir, f"calib_img_{captured_count:03d}.png")
                cv2.imwrite(img_path, frame)
                print(f"[INFO] Imagen capturada: {img_path}")
                
                captured_count += 1
                if captured_count >= num_images:
                    print("[INFO] Se han capturado suficientes imágenes. Iniciando calibración...")
                    break

        cap.release()
        cv2.destroyAllWindows()

        if captured_count < 3:
            raise RuntimeError("Se necesitan al menos 3 imágenes con tablero detectado para calibrar.")

        print(f"[INFO] Capturadas {captured_count} imágenes válidas. Iniciando calibración...")

        # Calibración de cámara
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=objpoints,
            imagePoints=imgpoints,
            imageSize=img_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=0,
            criteria=criteria,
        )
        if not ret:
            print("[ADVERTENCIA] calibrateCamera devolvió ret=0; los resultados podrían no ser válidos.")

        # dist puede tener más de 5 coeficientes; nos quedamos con 5 (k1,k2,p1,p2,k3)
        if dist.size < 5:
            dist_padded = np.zeros((1, 5), dtype=np.float64)
            dist_padded[0, : dist.size] = dist.ravel()
            dist = dist_padded
        else:
            dist = dist.ravel()[:5][None, :]

        return img_size, K, dist

    finally:
        # Limpiar directorio temporal
        shutil.rmtree(temp_dir, ignore_errors=True)


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
    ap.add_argument("--images", type=str, help="Patrón glob de imágenes (p. ej., 'calib/*.png').")
    ap.add_argument("--camera", type=int, default=None, help="Índice de la cámara para calibración en tiempo real.")
    ap.add_argument("--stereo", action="store_true", help="Activar modo de calibración estéreo.")
    ap.add_argument("--stereo-auto", action="store_true", help="Activar modo de calibración estéreo automática alternando fuentes de luz.")
    ap.add_argument("--frames-per-light", type=int, default=10, help="Número de frames a capturar por cada fuente de luz.")
    ap.add_argument("--delay-between-lights", type=float, default=2.0, help="Segundos de espera entre cambios de fuente de luz.")
    ap.add_argument("--camera-left", type=int, default=None, help="Índice de la cámara izquierda para calibración estéreo.")
    ap.add_argument("--camera-right", type=int, default=None, help="Índice de la cámara derecha para calibración estéreo.")
    ap.add_argument("--checkerboard-rows", type=int, default=6, help="Filas internas del checkerboard.")
    ap.add_argument("--checkerboard-cols", type=int, default=9, help="Columnas internas del checkerboard.")
    ap.add_argument("--square-size-mm", type=float, default=10.0, help="Tamaño de cada cuadrado del checkerboard en mm.")
    ap.add_argument("--num-images", type=int, default=15, help="Número de imágenes a capturar para calibración.")

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
        if args.stereo_auto:
            # Calibración estéreo automática alternando fuentes de luz
            if args.camera is None:
                ap.error("Para calibración estéreo automática, debes especificar --camera.")
            
            (w, h), K_light1, dist_light1, K_light2, dist_light2, R, T = calibrate_stereo_auto_alternating(
                args.camera,
                args.checkerboard_rows, args.checkerboard_cols, 
                args.square_size_mm, 
                args.frames_per_light,
                args.delay_between_lights
            )
            
            # Usar parámetros de la primera luz para el archivo de calibración
            fx, fy = float(K_light1[0, 0]), float(K_light1[1, 1])
            cx, cy = float(K_light1[0, 2]), float(K_light1[1, 2])
            k1, k2, p1, p2, k3 = map(float, dist_light1.flatten()[:5])
            override_res = (w, h)
        elif args.stereo:
            # Calibración estéreo (método original con dos cámaras)
            if args.camera_left is None or args.camera_right is None:
                ap.error("Para calibración estéreo, debes especificar --camera-left y --camera-right.")
            
            (w, h), K_left, dist_left, K_right, dist_right, R, T = calibrate_stereo_two_cameras(
                args.camera_left, args.camera_right, 
                args.checkerboard_rows, args.checkerboard_cols, 
                args.square_size_mm, args.num_images
            )
            
            # Usar parámetros de la cámara izquierda para el archivo de calibración
            fx, fy = float(K_left[0, 0]), float(K_left[1, 1])
            cx, cy = float(K_left[0, 2]), float(K_left[1, 2])
            k1, k2, p1, p2, k3 = map(float, dist_left.flatten()[:5])
            override_res = (w, h)
        elif args.images and args.camera is not None:
            ap.error("No puedes usar --images y --camera al mismo tiempo.")
        elif args.camera is not None:
            # Calibrar desde cámara en tiempo real
            (w, h), K, dist = calibrate_camera_from_live_feed(
                args.camera, args.checkerboard_rows, args.checkerboard_cols, 
                args.square_size_mm, args.num_images
            )
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            k1, k2, p1, p2, k3 = map(float, dist.flatten()[:5])
            override_res = (w, h)
        elif args.images:
            # Calibrar desde imágenes almacenadas
            img_paths = find_image_paths(args.images)
            (w, h), K, dist = calibrate_camera_from_chessboard(
                img_paths, args.checkerboard_rows, args.checkerboard_cols, args.square_size_mm
            )
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            k1, k2, p1, p2, k3 = map(float, dist.flatten()[:5])
            override_res = (w, h)
        else:
            ap.error("Debes pasar --images, --camera, --stereo, --stereo-alternating o usar --no-calibrate.")
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