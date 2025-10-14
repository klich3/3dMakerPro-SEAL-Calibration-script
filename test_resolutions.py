#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_resolutions.py
-------------------
Script para probar las resoluciones disponibles en las cámaras
"""

import cv2
import numpy as np

def test_camera_resolutions(camera_index=0, camera_name="Camera"):
    """Test para probar diferentes resoluciones en una cámara"""
    print(f"[TEST] Probando resoluciones para {camera_name} (índice {camera_index})")
    
    # Resoluciones comunes a probar
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
        (2560, 1440),
        (3840, 2160)
    ]
    
    # Probar diferentes backends
    backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2]
    
    for backend in backends:
        try:
            print(f"\n[TEST] Probando backend {backend}...")
            cap = cv2.VideoCapture(camera_index, backend)
            
            if not cap.isOpened():
                print(f"[ERROR] No se pudo abrir la cámara {camera_index} con backend {backend}")
                continue
                
            print(f"[SUCCESS] Cámara {camera_index} abierta con backend {backend}")
            
            # Obtener resolución original
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[INFO] Resolución original: {orig_width}x{orig_height}")
            
            # Probar diferentes resoluciones
            supported_resolutions = []
            
            for width, height in resolutions:
                try:
                    print(f"[TEST] Configurando resolución {width}x{height}...")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Verificar si la resolución se estableció correctamente
                    if new_width == width and new_height == height:
                        print(f"[SUCCESS] Resolución {width}x{height} soportada")
                        supported_resolutions.append((width, height))
                        
                        # Intentar leer un frame para verificar
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"[FRAME] Frame válido: {frame.shape}")
                        else:
                            print("[FRAME] No se pudo leer frame válido")
                    else:
                        print(f"[INFO] Resolución {width}x{height} no soportada. Obtenida: {new_width}x{new_height}")
                        
                except Exception as e:
                    print(f"[ERROR] Error con resolución {width}x{height}: {e}")
            
            print(f"\n[RESULT] Resoluciones soportadas para {camera_name} (backend {backend}):")
            for res in supported_resolutions:
                print(f"  - {res[0]}x{res[1]}")
                
            cap.release()
            
        except Exception as e:
            print(f"[ERROR] Error con backend {backend}: {e}")
            try:
                cap.release()
            except:
                pass

def test_camera_properties(camera_index=0, camera_name="Camera"):
    """Test para verificar propiedades de la cámara"""
    print(f"\n[TEST] Verificando propiedades de {camera_name} (índice {camera_index})...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {camera_index}")
        return
    
    # Probar diferentes propiedades
    properties = [
        (cv2.CAP_PROP_FRAME_WIDTH, "FRAME_WIDTH"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "FRAME_HEIGHT"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_BUFFERSIZE, "BUFFERSIZE"),
        (cv2.CAP_PROP_BRIGHTNESS, "BRIGHTNESS"),
        (cv2.CAP_PROP_CONTRAST, "CONTRAST"),
        (cv2.CAP_PROP_SATURATION, "SATURATION"),
        (cv2.CAP_PROP_FOURCC, "FOURCC"),
    ]
    
    for prop_id, prop_name in properties:
        try:
            value = cap.get(prop_id)
            if prop_name == "FOURCC":
                # Convertir FOURCC a string legible
                fourcc_str = "".join([chr((int(value) >> 8 * i) & 0xFF) for i in range(4)])
                print(f"[PROP] {prop_name}: {int(value)} ({fourcc_str})")
            else:
                print(f"[PROP] {prop_name}: {value}")
        except Exception as e:
            print(f"[ERROR] {prop_name}: {e}")
    
    cap.release()

if __name__ == "__main__":
    print("=== Test de Resoluciones de Cámaras ===")
    
    # Test para cámara A (Laser) - índice 0
    test_camera_properties(0, "Cámara A (Laser)")
    test_camera_resolutions(0, "Cámara A (Laser)")
    
    print("\n" + "="*50 + "\n")
    
    # Test para cámara B (UV) - índice 1
    test_camera_properties(1, "Cámara B (UV)")
    test_camera_resolutions(1, "Cámara B (UV)")