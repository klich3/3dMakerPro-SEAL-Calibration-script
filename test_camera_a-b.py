#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_camera_b.py
----------------
Script para testear específicamente la cámara B (UV) con índice 0
"""

import cv2
import time

def test_camera_b():
    """Test específico para la cámara B (UV) índice 0"""
    print("[TEST] Iniciando test de cámara B (UV) índice 0")
    
    # Probar diferentes configuraciones
    for backend in [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2]:
        try:
            print(f"[TEST] Intentando abrir cámara 0 con backend {backend}...")
            cap = cv2.VideoCapture(0, backend)
            
            if cap.isOpened():
                print(f"[SUCCESS] Cámara 0 abierta con backend {backend}")
                
                # Probar diferentes configuraciones
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Obtener información de la cámara
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[INFO] Resolución: {width}x{height} @ {fps} FPS")
                
                # Probar diferentes codecs/formats
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                print(f"[INFO] FOURCC: {fourcc} ({fourcc_str})")
                
                frame_count = 0
                start_time = time.time()
                
                print("[INFO] Mostrando stream continuo (presiona 'q' para salir)...")
                
                while True:
                    ret, frame = cap.read()
                    frame_count += 1
                    
                    if ret and frame is not None and frame.size > 0:
                        # Mostrar frame
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"Cam B (UV) - Frame: {frame_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Backend: {backend}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(display_frame, f"Size: {frame.shape}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        cv2.imshow('Preview Camara B (UV)', display_frame)
                        
                        # Calcular FPS real
                        if frame_count % 30 == 0:
                            elapsed = time.time() - start_time
                            real_fps = 30 / elapsed if elapsed > 0 else 0
                            print(f"[FPS] Teórico: {fps:.1f}, Real: {real_fps:.1f}")
                            start_time = time.time()
                    else:
                        print(f"[ERROR] Frame {frame_count} inválido: ret={ret}, frame={'None' if frame is None else 'size='+str(frame.size if frame is not None else 0)}")
                        # Crear frame negro para mantener la ventana abierta
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(black_frame, "SIN SEÑAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Preview Camara B (UV)', black_frame)
                    
                    # Salir con 'q'
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Salida solicitada")
                        break
                    elif key == ord('s'):
                        # Guardar screenshot
                        cv2.imwrite(f"camera_b_frame_{frame_count}.jpg", frame)
                        print(f"[INFO] Frame guardado como camera_b_frame_{frame_count}.jpg")
                
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print(f"[FAIL] No se pudo abrir cámara 0 con backend {backend}")
                
        except Exception as e:
            print(f"[ERROR] Error con backend {backend}: {e}")
            try:
                cap.release()
            except:
                pass
    
    return False

def test_camera_a():
    """Test específico para la cámara A (Laser) índice 1"""
    print("[TEST] Iniciando test de cámara A (Laser) índice 1")
    
    # Probar diferentes configuraciones
    for backend in [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2]:
        try:
            print(f"[TEST] Intentando abrir cámara 1 con backend {backend}...")
            cap = cv2.VideoCapture(1, backend)
            
            if cap.isOpened():
                print(f"[SUCCESS] Cámara 1 abierta con backend {backend}")
                
                # Probar diferentes configuraciones
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Obtener información de la cámara
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"[INFO] Resolución: {width}x{height} @ {fps} FPS")
                
                # Probar diferentes codecs/formats
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                print(f"[INFO] FOURCC: {fourcc} ({fourcc_str})")
                
                frame_count = 0
                start_time = time.time()
                
                print("[INFO] Mostrando stream continuo (presiona 'q' para salir)...")
                
                while True:
                    ret, frame = cap.read()
                    frame_count += 1
                    
                    if ret and frame is not None and frame.size > 0:
                        # Mostrar frame
                        display_frame = frame.copy()
                        cv2.putText(display_frame, f"Cam A (Laser) - Frame: {frame_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Backend: {backend}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(display_frame, f"Size: {frame.shape}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        cv2.imshow('Preview Camara A (Laser)', display_frame)
                        
                        # Calcular FPS real
                        if frame_count % 30 == 0:
                            elapsed = time.time() - start_time
                            real_fps = 30 / elapsed if elapsed > 0 else 0
                            print(f"[FPS] Teórico: {fps:.1f}, Real: {real_fps:.1f}")
                            start_time = time.time()
                    else:
                        print(f"[ERROR] Frame {frame_count} inválido: ret={ret}, frame={'None' if frame is None else 'size='+str(frame.size if frame is not None else 0)}")
                        # Crear frame negro para mantener la ventana abierta
                        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(black_frame, "SIN SEÑAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Preview Camara A (Laser)', black_frame)
                    
                    # Salir con 'q'
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Salida solicitada")
                        break
                    elif key == ord('s'):
                        # Guardar screenshot
                        cv2.imwrite(f"camera_a_frame_{frame_count}.jpg", frame)
                        print(f"[INFO] Frame guardado como camera_a_frame_{frame_count}.jpg")
                
                cap.release()
                cv2.destroyAllWindows()
                return True
            else:
                print(f"[FAIL] No se pudo abrir cámara 1 con backend {backend}")
                
        except Exception as e:
            print(f"[ERROR] Error con backend {backend}: {e}")
            try:
                cap.release()
            except:
                pass
    
    return False

def test_camera_properties(camera_index=0, camera_name="B (UV)"):
    """Test para verificar propiedades de la cámara"""
    print(f"[TEST] Verificando propiedades de cámara {camera_index} ({camera_name})...")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara {camera_index}")
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

def test_different_resolutions(camera_index=0):
    """Test para probar diferentes resoluciones"""
    print(f"[TEST] Probando diferentes resoluciones para cámara {camera_index}...")
    
    resolutions = [
        (640, 480),
        (1280, 720),
        (1920, 1080),
    ]
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara {camera_index}")
        return
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Resolución original: {original_width}x{original_height}")
    
    for width, height in resolutions:
        try:
            print(f"[TEST] Intentando resolución {width}x{height}...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"[RESULT] Resolución configurada: {new_width}x{new_height}")
            
            # Leer un frame para verificar
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[FRAME] Frame válido: {frame.shape}")
            else:
                print("[FRAME] No se pudo leer frame válido")
                
        except Exception as e:
            print(f"[ERROR] Error con resolución {width}x{height}: {e}")
    
    cap.release()

if __name__ == "__main__":
    import numpy as np
    
    print("=== Test de Cámara B (UV) ===")
    
    # Test de propiedades para cámara B
    test_camera_properties(0, "B (UV)")
    
    print("\n=== Test de resoluciones para cámara B ===")
    test_different_resolutions(0)
    
    print("\n=== Test de captura continua para cámara B ===")
    success_b = test_camera_b()
    
    print("\n=== Test de Cámara A (Laser) ===")
    
    # Test de propiedades para cámara A
    test_camera_properties(1, "A (Laser)")
    
    print("\n=== Test de resoluciones para cámara A ===")
    test_different_resolutions(1)
    
    print("\n=== Test de captura continua para cámara A ===")
    success_a = test_camera_a()
    
    if success_b:
        print("\n[SUCCESS] Cámara B funciona correctamente")
    else:
        print("\n[FAIL] Cámara B no funciona correctamente")
        
    if success_a:
        print("\n[SUCCESS] Cámara A funciona correctamente")
    else:
        print("\n[FAIL] Cámara A no funciona correctamente")