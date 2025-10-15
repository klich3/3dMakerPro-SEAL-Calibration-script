#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_camera.py
---------------
Script para debuggear problemas con la cámara B (UV) índice 0
"""

import cv2
import time
import numpy as np

def debug_camera_detailed(camera_index, camera_name, duration=10):
    """Debug detallado de una cámara"""
    print(f"[DEBUG] === Iniciando debug detallado de cámara {camera_index} ({camera_name}) ===")
    
    # Probar diferentes backends
    backends = [
        (cv2.CAP_ANY, "CAP_ANY"),
        (cv2.CAP_AVFOUNDATION, "CAP_AVFOUNDATION"),
        (cv2.CAP_V4L2, "CAP_V4L2")
    ]
    
    for backend, backend_name in backends:
        print(f"\n[DEBUG] Probando backend {backend_name} ({backend})")
        
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            
            if not cap.isOpened():
                print(f"[ERROR] No se pudo abrir cámara {camera_index} con backend {backend_name}")
                continue
                
            print(f"[SUCCESS] Cámara {camera_index} abierta con backend {backend_name}")
            
            # Configurar propiedades
            print("[DEBUG] Configurando propiedades...")
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Obtener propiedades
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = cap.get(cv2.CAP_PROP_CONTRAST)
            
            print(f"[PROPS] Dimensiones: {width}x{height}")
            print(f"[PROPS] FPS: {fps}")
            print(f"[PROPS] Brillo: {brightness}")
            print(f"[PROPS] Contraste: {contrast}")
            
            # Intentar leer frames durante un tiempo
            print(f"[DEBUG] Leyendo frames durante {duration} segundos...")
            start_time = time.time()
            frame_count = 0
            success_count = 0
            
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                frame_count += 1
                
                if ret and frame is not None:
                    success_count += 1
                    frame_size = frame.shape if frame is not None else "None"
                    print(f"[FRAME] {frame_count}: SUCCESS, size={frame_size}")
                    
                    # Mostrar frame si es válido
                    if frame is not None and frame.size > 0:
                        try:
                            cv2.imshow(f'Debug Camara {camera_index} ({camera_name})', frame)
                            cv2.waitKey(1)
                        except Exception as e:
                            print(f"[WARN] No se pudo mostrar frame: {e}")
                else:
                    print(f"[FRAME] {frame_count}: FAILED, ret={ret}, frame={'None' if frame is None else 'invalid'}")
                
                time.sleep(0.1)  # 100ms entre frames
            
            print(f"[RESULT] Frames totales: {frame_count}, Exitosos: {success_count}, Fallidos: {frame_count - success_count}")
            
            cap.release()
            
            # Si tuvimos éxito, salir del loop
            if success_count > 0:
                print(f"[SUCCESS] Cámara {camera_index} funciona con backend {backend_name}")
                break
                
        except Exception as e:
            print(f"[ERROR] Excepción con backend {backend_name}: {e}")
            try:
                cap.release()
            except:
                pass
    
    cv2.destroyAllWindows()

def test_frame_generation():
    """Test para generar frames de prueba"""
    print("[TEST] Generando frame de prueba...")
    
    # Crear frame de prueba
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Test Frame", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    print("[TEST] Mostrando frame de prueba...")
    cv2.imshow('Test Frame', test_frame)
    
    print("[TEST] Presiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("[TEST] Frame de prueba mostrado correctamente")

if __name__ == "__main__":
    print("=== Debug de Cámara B (UV) ===")
    
    # Test de frame de prueba primero
    test_frame_generation()
    
    # Debug detallado de cámara 0 (B - UV)
    debug_camera_detailed(0, "UV (B)")
    
    print("\n=== Debug de Cámara A (Laser) ===")
    # Debug detallado de cámara 1 (A - Laser)
    debug_camera_detailed(1, "Laser (A)")