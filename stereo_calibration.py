#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stereo_calibration.py
---------------------
Script simplificado para calibración estéreo con dos cámaras.
"""

import cv2
import numpy as np
import argparse
import os
import shutil
import tempfile
from pathlib import Path
import threading
import queue
import time


class CameraStream:
    def __init__(self, camera_index, name):
        self.camera_index = camera_index
        self.name = name
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=3)
        self.running = False
        self.thread = None
        self.last_frame = None
        self.last_frame_time = 0
        self.actual_fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self.initialized = False
        self.last_error = ""
        
    def _test_camera_property(self, prop_id, prop_name):
        """Prueba si una propiedad de cámara es compatible y se puede ajustar"""
        try:
            # Obtener valor actual
            current_value = self.cap.get(prop_id)
            
            # Intentar establecer un valor ligeramente diferente
            test_value = current_value + 0.1 if current_value < 0.9 else current_value - 0.1
            success_set = self.cap.set(prop_id, test_value)
            
            # Verificar si el valor se aplicó
            new_value = self.cap.get(prop_id)
            success_get = abs(new_value - test_value) <= 0.01
            
            # Restaurar valor original
            self.cap.set(prop_id, current_value)
            
            return success_set and success_get, current_value
        except Exception as e:
            return False, None
    
    def _detect_camera_capabilities(self):
        """Detecta las capacidades de la cámara"""
        capabilities = {}
        
        # Propiedades comunes a verificar
        properties = {
            'BRIGHTNESS': cv2.CAP_PROP_BRIGHTNESS,
            'CONTRAST': cv2.CAP_PROP_CONTRAST,
            'SATURATION': cv2.CAP_PROP_SATURATION,
            'HUE': cv2.CAP_PROP_HUE,
            'GAIN': cv2.CAP_PROP_GAIN,
            'EXPOSURE': cv2.CAP_PROP_EXPOSURE,
            'AUTO_EXPOSURE': cv2.CAP_PROP_AUTO_EXPOSURE
        }
        
        print(f"[INFO] Detectando capacidades de cámara {self.camera_index} ({self.name})...")
        
        for prop_name, prop_id in properties.items():
            try:
                supported, current_value = self._test_camera_property(prop_id, prop_name)
                if supported:
                    capabilities[prop_name] = {'supported': True, 'current_value': current_value}
                    print(f"[INFO] {prop_name}: SOPORTADO (Valor actual: {current_value:.2f})")
                else:
                    # Verificar si al menos se puede leer
                    try:
                        current_value = self.cap.get(prop_id)
                        capabilities[prop_name] = {'supported': False, 'current_value': current_value}
                        print(f"[INFO] {prop_name}: NO AJUSTABLE (Valor actual: {current_value:.2f})")
                    except:
                        capabilities[prop_name] = {'supported': False, 'current_value': None}
                        print(f"[INFO] {prop_name}: NO SOPORTADO")
            except Exception as e:
                capabilities[prop_name] = {'supported': False, 'current_value': None}
                print(f"[INFO] {prop_name}: ERROR - {str(e)}")
        
        return capabilities
    
    def start(self, brightness=None, contrast=None):
        """Inicia el stream de la cámara"""
        print(f"[INFO] Iniciando cámara {self.camera_index} ({self.name})...")
        print(f"[INFO] Si ves el error 'not authorized to capture video', otorga permisos de cámara a Terminal/Python")
        print(f"[INFO] En Preferencias del Sistema > Seguridad y privacidad > Privacidad > Cámara")
        
        # Probar diferentes backends como en el test que funciona
        backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2]
        
        for backend in backends:
            try:
                print(f"[TEST] Intentando abrir cámara {self.camera_index} con backend {backend}...")
                self.cap = cv2.VideoCapture(self.camera_index, backend)
                
                if self.cap.isOpened():
                    print(f"[SUCCESS] Cámara {self.camera_index} abierta con backend {backend}")
                    
                    # Configurar propiedades de la cámara
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Detectar capacidades de la cámara
                    capabilities = self._detect_camera_capabilities()
                    
                    # Configurar brillo y contraste si se especifican
                    if brightness is not None:
                        if capabilities.get('BRIGHTNESS', {}).get('supported', False):
                            print(f"[INFO] Intentando establecer brillo a {brightness} para cámara {self.camera_index}")
                            success = self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                            actual = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                            if success:
                                if abs(actual - brightness) <= 0.01:
                                    print(f"[INFO] Brillo establecido correctamente a {actual:.2f}")
                                else:
                                    print(f"[WARNING] Brillo ajustado pero con diferencia. Solicitado: {brightness}, Actual: {actual:.2f}")
                                    print(f"[INFO] Esto puede deberse a que la cámara solo acepta valores discretos o tiene un rango limitado")
                            else:
                                print(f"[ERROR] Fallo al establecer brillo. La cámara puede no soportar esta propiedad.")
                                print(f"[INFO] Valor actual: {actual:.2f}")
                        else:
                            print(f"[WARNING] Brillo no se puede ajustar en esta cámara. Propiedad no soportada.")
                    
                    if contrast is not None:
                        if capabilities.get('CONTRAST', {}).get('supported', False):
                            print(f"[INFO] Intentando establecer contraste a {contrast} para cámara {self.camera_index}")
                            success = self.cap.set(cv2.CAP_PROP_CONTRAST, contrast)
                            actual = self.cap.get(cv2.CAP_PROP_CONTRAST)
                            if success:
                                if abs(actual - contrast) <= 0.01:
                                    print(f"[INFO] Contraste establecido correctamente a {actual:.2f}")
                                else:
                                    print(f"[WARNING] Contraste ajustado pero con diferencia. Solicitado: {contrast}, Actual: {actual:.2f}")
                                    print(f"[INFO] Esto puede deberse a que la cámara solo acepta valores discretos o tiene un rango limitado")
                            else:
                                print(f"[ERROR] Fallo al establecer contraste. La cámara puede no soportar esta propiedad.")
                                print(f"[INFO] Valor actual: {actual:.2f}")
                        else:
                            print(f"[WARNING] Contraste no se puede ajustar en esta cámara. Propiedad no soportada.")
                    
                    # Verificar que podemos leer un frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"[INFO] Cámara {self.camera_index} ({self.name}) primer frame válido")
                        # Guardar el frame
                        self.last_frame = frame.copy()
                        break
                    else:
                        print(f"[WARNING] Cámara {self.camera_index} abierta pero no puede leer frames")
                        self.cap.release()
                        self.cap = None
                else:
                    print(f"[WARNING] No se pudo abrir la cámara {self.camera_index} con backend {backend}")
                    if self.cap:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                print(f"[ERROR] Excepción al abrir cámara {self.camera_index} con backend {backend}: {str(e)}")
                self.last_error = str(e)
                if self.cap:
                    self.cap.release()
                    self.cap = None
            
            if self.cap and self.cap.isOpened():
                break
        
        if not self.cap or not self.cap.isOpened():
            print(f"[ERROR] No se pudo abrir la cámara {self.camera_index} ({self.name}) después de todos los intentos")
            print(f"[ERROR] Último error: {self.last_error}")
            print(f"[ERROR] Posible solución: Otorga permisos de cámara a Terminal/Python en Preferencias del Sistema")
            return False
            
        # Obtener propiedades reales de la cámara
        try:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            theoretical_fps = self.cap.get(cv2.CAP_PROP_FPS)
            print(f"[INFO] Cámara {self.camera_index} ({self.name}): {width}x{height} @ {theoretical_fps} FPS (teórico)")
            
            # Mostrar brillo y contraste actuales
            brightness_val = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast_val = self.cap.get(cv2.CAP_PROP_CONTRAST)
            print(f"[INFO] Brillo actual: {brightness_val:.2f}, Contraste actual: {contrast_val:.2f}")
            
            # Probar diferentes codecs/formats como en el test
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"[INFO] FOURCC: {fourcc} ({fourcc_str})")
            
            # Inicializar variables para medir FPS real
            self.actual_fps = 0
            self.frame_count = 0
            self.start_time = time.time()
        except Exception as e:
            print(f"[WARNING] No se pudieron obtener propiedades de cámara {self.camera_index}: {str(e)}")
            width, height = 640, 480  # Valores por defecto
            
        self.running = True
        self.initialized = True
        self.thread = threading.Thread(target=self._capture_loop, name=f"CameraThread-{self.camera_index}")
        self.thread.daemon = True
        self.thread.start()
        
        return True
        
    def _capture_loop(self):
        """Bucle de captura en hilo separado"""
        self.frame_count = 0
        self.start_time = time.time()
        last_fps_report = time.time()
        
        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Calcular FPS real
                    self.frame_count += 1
                    current_time = time.time()
                    
                    # Reportar FPS cada 2 segundos para mejor seguimiento
                    if current_time - last_fps_report >= 2.0:
                        elapsed = current_time - self.start_time
                        if elapsed > 0:
                            self.actual_fps = self.frame_count / elapsed
                            print(f"[FPS] Cámara {self.camera_index} ({self.name}): Teórico={self.cap.get(cv2.CAP_PROP_FPS):.1f}, Real={self.actual_fps:.1f}")
                        self.frame_count = 0
                        self.start_time = current_time
                        last_fps_report = current_time
                    
                    # Actualizar último frame
                    self.last_frame = frame.copy()
                    self.last_frame_time = current_time
                    
                    # Limpiar y actualizar cola
                    try:
                        while not self.frame_queue.empty():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                else:
                    # Añadir más información de depuración
                    print(f"[DEBUG] Cámara {self.camera_index} no obtuvo frame válido: ret={ret}, frame={'None' if frame is None else 'size='+str(frame.size if frame is not None else 0)}")
                    time.sleep(0.005)
            except Exception as e:
                print(f"[ERROR] Captura cámara {self.camera_index}: {str(e)}")
                time.sleep(0.01)
                
    def get_frame(self):
        """Obtiene el último frame disponible"""
        try:
            return True, self.frame_queue.get_nowait()
        except queue.Empty:
            # Devolver el último frame válido si no hay nuevo
            if self.last_frame is not None:
                return True, self.last_frame
            return False, None
            
    def get_latest_frame(self):
        """Obtiene el frame más reciente"""
        return self.last_frame if self.last_frame is not None else None
        
    def get_fps(self):
        """Obtiene el FPS real de la cámara"""
        return self.actual_fps
        
    def is_initialized(self):
        """Verifica si la cámara fue inicializada correctamente"""
        return self.initialized
        
    def stop(self):
        """Detiene el stream de la cámara"""
        print(f"[INFO] Deteniendo cámara {self.camera_index} ({self.name})...")
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print(f"[INFO] Cámara {self.camera_index} ({self.name}) detenida")


class PatternDetector:
    def __init__(self):
        self.detection_cache = {}
        self.last_detection_time = {}
        self.detection_interval = 0.1
        
    def detect_pattern(self, gray_frame, pattern_size, camera_id):
        """Detecta patrón de tablero con cache para evitar sobrecarga"""
        current_time = time.time()
        
        # Verificar si es tiempo de hacer nueva detección
        last_time = self.last_detection_time.get(camera_id, 0)
        if current_time - last_time < self.detection_interval:
            # Devolver resultado cacheado
            return self.detection_cache.get(camera_id, (False, None))
            
        # Hacer nueva detección
        try:
            found, corners = cv2.findChessboardCorners(
                gray_frame, pattern_size, 
                flags=cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH
            )
            
            # Guardar en cache
            self.detection_cache[camera_id] = (found, corners)
            self.last_detection_time[camera_id] = current_time
            return found, corners
        except Exception as e:
            print(f"[WARNING] Error detección cámara {camera_id}: {str(e)}")
            return False, None


def test_single_camera(camera_index, camera_name, duration=10):
    """Función para testear una sola cámara"""
    print(f"[TEST] Probando cámara {camera_index} ({camera_name}) por {duration} segundos...")
    print(f"[INFO] Si ves el error 'not authorized to capture video', otorga permisos de cámara a Terminal/Python")
    
    # Probar diferentes backends como en el test que funciona
    backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2]
    cap = None
    
    for backend in backends:
        try:
            print(f"[TEST] Intentando abrir cámara {camera_index} con backend {backend}...")
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                print(f"[TEST] Cámara {self.camera_index} abierta con backend {backend}")
                
                # Configurar propiedades como en el test que funciona
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
                break
            else:
                if cap:
                    cap.release()
                    cap = None
        except Exception as e:
            print(f"[ERROR] Error con backend {backend}: {e}")
            if cap:
                cap.release()
                cap = None
    
    if not cap or not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara {camera_index} ({camera_name})")
        print(f"[ERROR] Posible solución: Otorga permisos de cámara a Terminal/Python en Preferencias del Sistema")
        return False
    
    print(f"[TEST] Cámara {camera_index} ({camera_name}) abierta correctamente")
    
    # Configurar propiedades
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[TEST] Propiedades cámara {camera_index}: {width}x{height} @ {fps} FPS")
    
    frame_count = 0
    start_time = time.time()
    
    print("[INFO] Mostrando stream continuo (presiona 'q' para salir)...")
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            frame_count += 1
            # Mostrar frame
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Cam {camera_index} ({camera_name})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Size: {frame.shape}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow(f'Test Camara {camera_index} ({camera_name})', display_frame)
            
            # Calcular FPS real
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                real_fps = 30 / elapsed if elapsed > 0 else 0
                print(f"[FPS] Teórico: {fps:.1f}, Real: {real_fps:.1f}")
                start_time = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print(f"[TEST] Frame inválido de cámara {camera_index}: ret={ret}, frame={'None' if frame is None else 'size='+str(frame.size if frame is not None else 0)}")
            # Crear frame negro para mantener la ventana abierta
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(black_frame, "SIN SEÑAL", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(f'Test Camara {camera_index} ({camera_name})', black_frame)
            time.sleep(0.01)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"[TEST] Cámara {camera_index} capturó {frame_count} frames en {duration} segundos")
    return frame_count > 0


def save_calibration_results(img_size, K_right, dist_right, K_left, dist_left, R, T, output_file):
    """Guarda resultados de calibración"""
    try:
        with open(output_file, 'w') as f:
            f.write(f"{img_size[0]} {img_size[1]}\n")
            f.write("Calibración estéreo completada\n")
            f.write(f"K_right: {K_right.flatten()}\n")
            f.write(f"dist_right: {dist_right.flatten()}\n")
            f.write(f"K_left: {K_left.flatten()}\n")
            f.write(f"dist_left: {dist_left.flatten()}\n")
            f.write(f"R: {R.flatten()}\n")
            f.write(f"T: {T.flatten()}\n")
        print(f"[INFO] Resultados guardados en {output_file}")
    except Exception as e:
        print(f"[ERROR] Guardar resultados: {str(e)}")


def save_seal_calibration_file(img_size, K_left, dist_left, K_right, dist_right, R, T, 
                              template_file, output_file, dev_id=None, 
                              square_size_mm=25.0, rows=6, cols=9):
    """Guarda resultados de calibración en formato SEAL compatible"""
    try:
        # Importar las funciones necesarias de seal_calib_writer
        from seal_calib_writer import write_new_calibration
        
        # Extraer parámetros intrínsecos de la cámara izquierda (cámara A - laser)
        fx = float(K_left[0, 0])
        fy = float(K_left[1, 1])
        cx = float(K_left[0, 2])
        cy = float(K_left[1, 2])
        
        # Extraer coeficientes de distorsión (usando 5 parámetros: k1, k2, p1, p2, k3)
        dist_coeffs = dist_left.flatten()
        k1 = float(dist_coeffs[0]) if len(dist_coeffs) > 0 else 0.0
        k2 = float(dist_coeffs[1]) if len(dist_coeffs) > 1 else 0.0
        p1 = float(dist_coeffs[2]) if len(dist_coeffs) > 2 else 0.0
        p2 = float(dist_coeffs[3]) if len(dist_coeffs) > 3 else 0.0
        k3 = float(dist_coeffs[4]) if len(dist_coeffs) > 4 else 0.0
        
        # Calcular tamaño físico del chessboard en mm
        # (cols-1) * square_size porque hay (cols-1) cuadrados en horizontal
        # (rows-1) * square_size porque hay (rows-1) cuadrados en vertical
        board_width_mm = (cols - 1) * square_size_mm
        board_height_mm = (rows - 1) * square_size_mm
        
        # Calcular corrección angular/tilt entre cámaras desde la matriz de rotación
        # Convertir matriz de rotación a ángulos de Euler (en grados)
        # R es la rotación de la cámara derecha respecto a la izquierda
        import math
        
        # Extraer ángulos de Euler de la matriz de rotación (en grados)
        # Usando convención XYZ (pitch, yaw, roll)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = math.atan2(R[2, 1], R[2, 2])  # Rotación alrededor del eje X
            yaw = math.atan2(-R[2, 0], sy)        # Rotación alrededor del eje Y
            roll = math.atan2(R[1, 0], R[0, 0])   # Rotación alrededor del eje Z
        else:
            pitch = math.atan2(-R[1, 2], R[1, 1])
            yaw = math.atan2(-R[2, 0], sy)
            roll = 0
        
        # Convertir de radianes a grados
        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        roll_deg = math.degrees(roll)
        
        # Para el formato SEAL, usamos los ángulos más significativos
        # Típicamente pitch y yaw para corrección angular y tilt
        angular_correction = int(round(yaw_deg))    # Corrección angular horizontal
        tilt_correction = int(round(pitch_deg))     # Corrección de inclinación vertical
        
        print(f"[INFO] Tamaño físico del patrón: {board_width_mm:.1f} x {board_height_mm:.1f} mm")
        print(f"[INFO] Corrección angular/tilt entre cámaras: {angular_correction}° {tilt_correction}°")
        print(f"[INFO] Ángulos de rotación detallados - Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°, Roll: {roll_deg:.2f}°")
        
        # Forzar resolución 1280x720
        override_res = (1280, 720)
        
        # Llamar a la función para escribir el archivo de calibración en formato SEAL
        write_new_calibration(
            template_path=template_file,
            out_path=output_file,
            intrinsics_line_idx=5,  # Índice de línea para los parámetros intrínsecos (1-based)
            fx=fx, fy=fy, cx=cx, cy=cy,
            k1=k1, k2=k2, p1=p1, p2=p2, k3=k3,
            override_res=override_res,
            dev_id=dev_id,  # Usar el ID proporcionado por argumentos
            soft_version="3.0.0.1116"  # Versión fija especificada
        )
        
        # Actualizar el archivo para usar la fecha actual y el tipo Sychev-calibration
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Reemplazar la fecha y el tipo
        import re
        content = re.sub(r'CalibrateDate:[^*]*', f'CalibrateDate:{current_date}', content)
        content = re.sub(r'Type:[^*]*', 'Type:Sychev-calibration', content)
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"[INFO] Archivo de calibración SEAL guardado en {output_file}")
    except Exception as e:
        print(f"[ERROR] Guardar archivo de calibración SEAL: {str(e)}")


def combine_frames(frame_left, frame_right):
    """Combina dos frames en una sola imagen lado a lado"""
    # Asegurar que ambos frames tengan el tamaño correcto 1280x720
    target_width, target_height = 1280, 720
    
    if frame_left is None and frame_right is None:
        return np.zeros((target_height, target_width*2, 3), dtype=np.uint8)
    elif frame_left is None:
        frame_left = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    elif frame_right is None:
        frame_right = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Redimensionar frames si es necesario
    h_left, w_left = frame_left.shape[:2]
    h_right, w_right = frame_right.shape[:2]
    
    if h_left != target_height or w_left != target_width:
        frame_left = cv2.resize(frame_left, (target_width, target_height))
    if h_right != target_height or w_right != target_width:
        frame_right = cv2.resize(frame_right, (target_width, target_height))
    
    # Combinar horizontalmente
    combined_frame = np.hstack((frame_left, frame_right))
    return combined_frame


def calculate_reprojection_errors(objpoints, imgpoints_left, imgpoints_right, 
                                   K_left, dist_left, K_right, dist_right, R, T):
    """Calcula errores de reproyección para cada imagen"""
    errors_left = []
    errors_right = []
    
    rvec, _ = cv2.Rodrigues(R)
    
    for i, objp in enumerate(objpoints):
        # Proyectar puntos 3D a 2D para cámara izquierda
        imgpoints_left_proj, _ = cv2.projectPoints(
            objp, np.zeros((3,1)), np.zeros((3,1)), K_left, dist_left)
        error_left = cv2.norm(imgpoints_left[i], imgpoints_left_proj, 
                             cv2.NORM_L2) / len(imgpoints_left_proj)
        errors_left.append(error_left)
        
        # Para cámara derecha, usar R y T
        imgpoints_right_proj, _ = cv2.projectPoints(
            objp, rvec, T, K_right, dist_right)
        error_right = cv2.norm(imgpoints_right[i], imgpoints_right_proj, 
                              cv2.NORM_L2) / len(imgpoints_right_proj)
        errors_right.append(error_right)
    
    return errors_left, errors_right


def validate_calibration(ret, F, img_size, errors_left, errors_right):
    """Valida la calidad de la calibración"""
    print(f"\n[VALIDACIÓN DE CALIBRACIÓN]")
    
    # 1. Error RMS
    if ret < 0.5:
        print(f"✓ Error RMS excelente: {ret:.4f} (< 0.5)")
    elif ret < 1.0:
        print(f"⚠ Error RMS aceptable: {ret:.4f} (0.5-1.0)")
    else:
        print(f"✗ Error RMS alto: {ret:.4f} (> 1.0) - Recalibrar recomendado")
    
    # 2. Errores de reproyección
    print(f"\n[ERRORES DE REPROYECCIÓN]")
    print(f"Error medio izquierda: {np.mean(errors_left):.4f} píxeles")
    print(f"Error medio derecha: {np.mean(errors_right):.4f} píxeles")
    print(f"Error máximo izquierda: {np.max(errors_left):.4f} píxeles")
    print(f"Error máximo derecha: {np.max(errors_right):.4f} píxeles")
    
    # Detectar outliers (errores > 1.0 píxel)
    outliers_left = [i for i, e in enumerate(errors_left) if e > 1.0]
    outliers_right = [i for i, e in enumerate(errors_right) if e > 1.0]
    
    if outliers_left or outliers_right:
        print(f"\n[WARNING] Outliers detectados:")
        if outliers_left:
            print(f"  Izquierda: imágenes {outliers_left}")
        if outliers_right:
            print(f"  Derecha: imágenes {outliers_right}")
        print(f"  Considera eliminar estas imágenes y recalibrar")
    
    # 3. Matriz fundamental
    if F is not None:
        U, S, Vt = np.linalg.svd(F)
        rank = np.sum(S > 1e-7)
        if rank == 2:
            print(f"\n✓ Matriz fundamental válida (rango = 2)")
        else:
            print(f"\n✗ Matriz fundamental inválida (rango = {rank})")
    
    # 4. Verificar resolución
    if img_size[0] != 1280 or img_size[1] != 720:
        print(f"\n⚠ Resolución {img_size} difiere de SEAL (1280x720)")
        print(f"  Las imágenes serán redimensionadas automáticamente")


def print_calibration_summary(K_left, K_right, dist_left, dist_right, R, T, ret, img_size):
    """Imprime resumen detallado de calibración"""
    print("\n" + "="*60)
    print("RESUMEN DE CALIBRACIÓN ESTÉREO")
    print("="*60)
    
    print(f"\n📐 RESOLUCIÓN: {img_size[0]}x{img_size[1]}")
    print(f"📊 ERROR RMS: {ret:.4f} píxeles")
    
    print(f"\n🎥 CÁMARA IZQUIERDA (Laser - A):")
    print(f"   Focal: fx={K_left[0,0]:.2f}, fy={K_left[1,1]:.2f}")
    print(f"   Centro: cx={K_left[0,2]:.2f}, cy={K_left[1,2]:.2f}")
    print(f"   Distorsión: k1={dist_left[0,0]:.4f}, k2={dist_left[0,1]:.4f}")
    
    print(f"\n🎥 CÁMARA DERECHA (UV - B):")
    print(f"   Focal: fx={K_right[0,0]:.2f}, fy={K_right[1,1]:.2f}")
    print(f"   Centro: cx={K_right[0,2]:.2f}, cy={K_right[1,2]:.2f}")
    print(f"   Distorsión: k1={dist_right[0,0]:.4f}, k2={dist_right[0,1]:.4f}")
    
    print(f"\n🔄 TRANSFORMACIÓN ESTÉREO:")
    baseline = np.linalg.norm(T)
    print(f"   Baseline: {baseline:.2f} mm")
    print(f"   Traslación: X={T[0,0]:.2f}, Y={T[1,0]:.2f}, Z={T[2,0]:.2f} mm")
    
    # Calcular ángulos de rotación
    rvec, _ = cv2.Rodrigues(R)
    angles = np.degrees(rvec.flatten())
    print(f"   Rotación: X={angles[0]:.2f}°, Y={angles[1]:.2f}°, Z={angles[2]:.2f}°")
    
    print("\n" + "="*60)


def process_existing_images(images_dir, checkerboard_rows, checkerboard_cols, 
                           square_size_mm, pattern_type="chessboard"):
    """Procesa imágenes existentes para calibración"""
    import glob
    
    print(f"[INFO] Procesando imágenes desde: {images_dir}")
    
    # Buscar pares de imágenes (left/right) con diferentes formatos
    left_patterns = [
        os.path.join(images_dir, "left_*.jpg"),
        os.path.join(images_dir, "left_*.png"),
        os.path.join(images_dir, "*_left.jpg"),
        os.path.join(images_dir, "*_left.png")
    ]
    
    right_patterns = [
        os.path.join(images_dir, "right_*.jpg"),
        os.path.join(images_dir, "right_*.png"),
        os.path.join(images_dir, "*_right.jpg"),
        os.path.join(images_dir, "*_right.png")
    ]
    
    left_images = []
    right_images = []
    
    for pattern in left_patterns:
        left_images = sorted(glob.glob(pattern))
        if left_images:
            break
    
    for pattern in right_patterns:
        right_images = sorted(glob.glob(pattern))
        if right_images:
            break
    
    if len(left_images) == 0 or len(right_images) == 0:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")
    
    if len(left_images) != len(right_images):
        raise RuntimeError(f"Número diferente de imágenes izq/der: {len(left_images)} vs {len(right_images)}")
    
    print(f"[INFO] Encontrados {len(left_images)} pares de imágenes")
    
    # Preparar puntos del patrón
    objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
    objp *= square_size_mm
    
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = None
    
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"[INFO] Procesando par {i+1}/{len(left_images)}...")
        
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        
        if img_left is None or img_right is None:
            print(f"[WARNING] No se pudo leer par {i+1}, saltando...")
            continue
        
        if img_size is None:
            img_size = (img_left.shape[1], img_left.shape[0])
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Detectar patrón
        if pattern_type == "chessboard":
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, (checkerboard_cols, checkerboard_rows), None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, (checkerboard_cols, checkerboard_rows), None)
        elif pattern_type == "circles":
            ret_left, corners_left = cv2.findCirclesGrid(gray_left, (checkerboard_cols, checkerboard_rows), None, cv2.CALIB_CB_ASYMMETRIC_GRID)
            ret_right, corners_right = cv2.findCirclesGrid(gray_right, (checkerboard_cols, checkerboard_rows), None, cv2.CALIB_CB_ASYMMETRIC_GRID)
        else:
            print(f"[WARNING] Tipo de patrón {pattern_type} no soportado para procesamiento de imágenes")
            continue
        
        if ret_left and ret_right:
            # Refinar esquinas para chessboard
            if pattern_type == "chessboard":
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp.copy())
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            print(f"[OK] Par {i+1} procesado correctamente")
        else:
            print(f"[WARNING] No se detectó patrón en par {i+1}")
    
    if len(objpoints) < 3:
        raise RuntimeError(f"Insuficientes pares válidos: {len(objpoints)}")
    
    print(f"[INFO] Calibrando con {len(objpoints)} pares válidos...")
    
    # Calibración
    ret_left, K_left, dist_left, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None)
    ret_right, K_right, dist_right, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None)
    
    if not ret_left or not ret_right:
        raise RuntimeError("Error en calibración individual")
    
    print("[INFO] Calibración estéreo...")
    # Usar flags optimizados
    flags = (cv2.CALIB_FIX_INTRINSIC |  # Usar intrínsecos previamente calibrados
             cv2.CALIB_RATIONAL_MODEL)   # Modelo de distorsión más preciso
    
    ret, K_right, dist_right, K_left, dist_left, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_right, imgpoints_left,
        K_right, dist_right, K_left, dist_left,
        img_size, criteria=criteria, flags=flags)
    
    if not ret:
        raise RuntimeError("Error en calibración estéreo")
    
    # Calcular errores de reproyección
    errors_left, errors_right = calculate_reprojection_errors(
        objpoints, imgpoints_left, imgpoints_right,
        K_left, dist_left, K_right, dist_right, R, T)
    
    # Validar calibración
    validate_calibration(ret, F, img_size, errors_left, errors_right)
    
    # Resumen detallado
    print_calibration_summary(K_left, K_right, dist_left, dist_right, R, T, ret, img_size)
    
    print("[INFO] Calibración completada")
    return img_size, K_right, dist_right, K_left, dist_left, R, T


def process_existing_images(images_dir, checkerboard_rows, checkerboard_cols, 
                           square_size_mm, pattern_type="chessboard"):
    """Procesa imágenes existentes para calibración"""
    import glob
    
    print(f"[INFO] Procesando imágenes desde: {images_dir}")
    
    # Buscar pares de imágenes (left/right) con diferentes formatos
    left_patterns = [
        os.path.join(images_dir, "left_*.jpg"),
        os.path.join(images_dir, "left_*.png"),
        os.path.join(images_dir, "*_left.jpg"),
        os.path.join(images_dir, "*_left.png")
    ]
    
    right_patterns = [
        os.path.join(images_dir, "right_*.jpg"),
        os.path.join(images_dir, "right_*.png"),
        os.path.join(images_dir, "*_right.jpg"),
        os.path.join(images_dir, "*_right.png")
    ]
    
    left_images = []
    right_images = []
    
    for pattern in left_patterns:
        left_images = sorted(glob.glob(pattern))
        if left_images:
            break
    
    for pattern in right_patterns:
        right_images = sorted(glob.glob(pattern))
        if right_images:
            break
    
    if len(left_images) == 0 or len(right_images) == 0:
        raise RuntimeError(f"No se encontraron imágenes en {images_dir}")
    
    if len(left_images) != len(right_images):
        raise RuntimeError(f"Número diferente de imágenes izq/der: {len(left_images)} vs {len(right_images)}")
    
    print(f"[INFO] Encontrados {len(left_images)} pares de imágenes")
    
    # Preparar puntos del patrón
    objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
    objp *= square_size_mm
    
    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    img_size = None
    
    for i, (left_path, right_path) in enumerate(zip(left_images, right_images)):
        print(f"[INFO] Procesando par {i+1}/{len(left_images)}...")
        
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        
        if img_left is None or img_right is None:
            print(f"[WARNING] No se pudo leer par {i+1}, saltando...")
            continue
        
        if img_size is None:
            img_size = (img_left.shape[1], img_left.shape[0])
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # Detectar patrón
        if pattern_type == "chessboard":
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, (checkerboard_cols, checkerboard_rows), None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, (checkerboard_cols, checkerboard_rows), None)
        elif pattern_type == "circles":
            ret_left, corners_left = cv2.findCirclesGrid(gray_left, (checkerboard_cols, checkerboard_rows), None, cv2.CALIB_CB_ASYMMETRIC_GRID)
            ret_right, corners_right = cv2.findCirclesGrid(gray_right, (checkerboard_cols, checkerboard_rows), None, cv2.CALIB_CB_ASYMMETRIC_GRID)
        else:
            print(f"[WARNING] Tipo de patrón {pattern_type} no soportado para procesamiento de imágenes")
            continue
        
        if ret_left and ret_right:
            # Refinar esquinas para chessboard
            if pattern_type == "chessboard":
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp.copy())
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            print(f"[OK] Par {i+1} procesado correctamente")
        else:
            print(f"[WARNING] No se detectó patrón en par {i+1}")
    
    if len(objpoints) < 3:
        raise RuntimeError(f"Insuficientes pares válidos: {len(objpoints)}")
    
    print(f"[INFO] Calibrando con {len(objpoints)} pares válidos...")
    
    # Calibración
    ret_left, K_left, dist_left, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_size, None, None)
    ret_right, K_right, dist_right, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_size, None, None)
    
    if not ret_left or not ret_right:
        raise RuntimeError("Error en calibración individual")
    
    print("[INFO] Calibración estéreo...")
    # Usar flags optimizados
    flags = (cv2.CALIB_FIX_INTRINSIC |  # Usar intrínsecos previamente calibrados
             cv2.CALIB_RATIONAL_MODEL)   # Modelo de distorsión más preciso
    
    ret, K_right, dist_right, K_left, dist_left, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_right, imgpoints_left,
        K_right, dist_right, K_left, dist_left,
        img_size, criteria=criteria, flags=flags)
    
    if not ret:
        raise RuntimeError("Error en calibración estéreo")
    
    # Calcular errores de reproyección
    errors_left, errors_right = calculate_reprojection_errors(
        objpoints, imgpoints_left, imgpoints_right,
        K_left, dist_left, K_right, dist_right, R, T)
    
    # Validar calibración
    validate_calibration(ret, F, img_size, errors_left, errors_right)
    
    # Resumen detallado
    print_calibration_summary(K_left, K_right, dist_left, dist_right, R, T, ret, img_size)
    
    print("[INFO] Calibración completada")
    return img_size, K_right, dist_right, K_left, dist_left, R, T


def calibrate_stereo_cameras(camera_left_index, camera_right_index, 
                           checkerboard_rows, checkerboard_cols, 
                           square_size_mm, num_images=15, pattern_type="chessboard",
                           uv_brightness=-1, uv_contrast=-1, resolution=None, target_fps=None,
                           aruco_dict_name="DICT_6X6_250", auto_capture=True):
    """
    Calibra cámaras estéreo usando dos cámaras físicas diferentes.
    """
    
    # Crear directorio para guardar imágenes de calibración
    temp_dir = os.path.join(os.getcwd(), "calib_imgs")
    os.makedirs(temp_dir, exist_ok=True)
    print(f"[INFO] Directorio de imágenes: {temp_dir}")
    
    # Inicializar streams de cámara
    # Cámara A (laser) = izquierda = índice 0
    # Cámara B (UV) = derecha = índice 1
    print("[INFO] Iniciando streams de cámara...")
    print("[INFO] Si ves el error 'not authorized to capture video', otorga permisos de cámara a Terminal/Python")
    
    # Crear objetos de cámara
    left_stream = CameraStream(camera_left_index, "Laser (A)")      # Cámara A (laser)
    right_stream = CameraStream(camera_right_index, "UV (B)")       # Cámara B (UV)
    
    try:
        # Iniciar cámaras con más tiempo entre ellas
        print("[INFO] Iniciando cámara izquierda (A - laser)...")
        left_success = left_stream.start()
        
        if not left_success or not left_stream.is_initialized():
            print(f"[ERROR] No se pudo iniciar cámara izquierda (A - laser) {camera_left_index}")
            print(f"[ERROR] Posible solución: Otorga permisos de cámara a Terminal/Python en Preferencias del Sistema")
            # Intentar reiniciar la cámara con un enfoque más robusto
            print("[INFO] Intentando reiniciar cámara izquierda con enfoque más robusto...")
            left_stream = CameraStream(camera_left_index, "Laser (A)")
            left_success = left_stream.start()
            
            if not left_success or not left_stream.is_initialized():
                raise RuntimeError(f"No se pudo iniciar cámara izquierda (A - laser) {camera_left_index}")
        
        # Preparar brillo y contraste para la cámara UV si se especifican
        target_brightness = None
        target_contrast = None
        if uv_brightness >= 0:
            target_brightness = float(uv_brightness)/100.0
        if uv_contrast >= 0:
            target_contrast = float(uv_contrast)/100.0
        
        # Esperar más tiempo antes de iniciar la segunda cámara
        print("[INFO] Esperando 2 segundos antes de iniciar cámara derecha...")
        time.sleep(2.0)
        
        print("[INFO] Iniciando cámara derecha (B - UV)...")
        right_success = right_stream.start(brightness=target_brightness, contrast=target_contrast)
        
        if not right_success or not right_stream.is_initialized():
            print(f"[ERROR] No se pudo iniciar cámara derecha (B - UV) {camera_right_index}")
            # Mostrar información adicional de error
            if hasattr(right_stream, 'last_error') and right_stream.last_error:
                print(f"[ERROR] Detalles: {right_stream.last_error}")
            print(f"[ERROR] Posible solución: Otorga permisos de cámara a Terminal/Python en Preferencias del Sistema")
            # Intentar reiniciar la cámara con un enfoque más robusto
            print("[INFO] Intentando reiniciar cámara derecha con enfoque más robusto...")
            right_stream = CameraStream(camera_right_index, "UV (B)")
            right_success = right_stream.start(brightness=target_brightness, contrast=target_contrast)
            
            if not right_success or not right_stream.is_initialized():
                raise RuntimeError(f"No se pudo iniciar cámara derecha (B - UV) {camera_right_index}")
        
        # Esperar a que las cámaras estén listas
        print("[INFO] Esperando cámaras...")
        time.sleep(3)
        
        # Verificar que las cámaras están funcionando
        left_ready = left_stream.get_latest_frame() is not None
        right_ready = right_stream.get_latest_frame() is not None
        
        print(f"[INFO] Estado cámaras - Izquierda (A): {'OK' if left_ready else 'SIN FRAME'}, Derecha (B): {'OK' if right_ready else 'SIN FRAME'}")
        
        # Debug adicional
        if not left_ready:
            print(f"[DEBUG] Cámara izquierda (A) no tiene frame. Probando leer directamente...")
            if left_stream.cap and left_stream.cap.isOpened():
                ret, frame = left_stream.cap.read()
                print(f"[DEBUG] Lectura directa cámara A: ret={ret}, frame={'válido' if frame is not None and frame.size > 0 else 'inválido'}")
        if not right_ready:
            print(f"[DEBUG] Cámara derecha (B) no tiene frame. Probando leer directamente...")
            if right_stream.cap and right_stream.cap.isOpened():
                ret, frame = right_stream.cap.read()
                print(f"[DEBUG] Lectura directa cámara B: ret={ret}, frame={'válido' if frame is not None and frame.size > 0 else 'inválido'}")
        
        # Preparar puntos del patrón con ajustes específicos por tipo
        if pattern_type == "chessboard":
            objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
            # Ajustes específicos para tablero de ajedrez
            detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
            print("[INFO] Usando patrón de tablero de ajedrez con detección adaptativa")
        elif pattern_type == "circles":
            # Para patrón de círculos asimétricos
            objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
            # Ajustar espaciado para círculos asimétricos
            objp[:, 0] *= square_size_mm * 2  # Espaciado horizontal
            objp[:, 1] *= square_size_mm      # Espaciado vertical
            # Ajustes específicos para círculos asimétricos
            detection_flags = cv2.CALIB_CB_ASYMMETRIC_GRID
            print("[INFO] Usando patrón de círculos asimétricos")
        elif pattern_type == "charuco":
            # Para patrón ChArUco, preparar puntos de objeto
            objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
            objp[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2)
            # Ajustes específicos para ChArUco
            detection_flags = None  # No se usa para ChArUco
            print("[INFO] Usando patrón ChArUco con detección de marcadores ArUco")
        elif pattern_type == "bricks":
            # Para patrón tipo bricks, preparar puntos de objeto
            # El patrón bricks tiene una estructura específica donde las filas alternan
            objp = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
            for i in range(checkerboard_rows):
                for j in range(checkerboard_cols):
                    # Para patrón bricks, desplazamos las filas pares
                    x_offset = (i % 2) * (square_size_mm / 2)
                    objp[i * checkerboard_cols + j, 0] = j * square_size_mm + x_offset
                    objp[i * checkerboard_cols + j, 1] = i * square_size_mm
            detection_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
            print("[INFO] Usando patrón tipo bricks")
        
        objp *= square_size_mm
        
        # Arrays para calibración
        objpoints = []  # Puntos 3D en el espacio real
        imgpoints_right = []  # Puntos 2D en la imagen derecha
        imgpoints_left = []  # Puntos 2D en la imagen izquierda
        
        # Arrays específicos para ChArUco
        charuco_obj_points = []  # Puntos 3D para ChArUco
        charuco_img_points = []  # Puntos 2D para ChArUco
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Inicializar detector de patrones
        detector = PatternDetector()
        
        # Para círculos asimétricos, configurar el detector de blobs
        blob_detector = None
        if pattern_type == "circles":
            try:
                # Configurar parámetros del detector de blobs
                blob_params = cv2.SimpleBlobDetector_Params()
                
                # Cambiar umbrales
                blob_params.minThreshold = 8
                blob_params.maxThreshold = 255
                
                # Filtrar por área
                blob_params.filterByArea = True
                blob_params.minArea = 64
                blob_params.maxArea = 2500
                
                # Filtrar por circularidad
                blob_params.filterByCircularity = True
                blob_params.minCircularity = 0.1
                
                # Filtrar por convexidad
                blob_params.filterByConvexity = True
                blob_params.minConvexity = 0.87
                
                # Filtrar por inercia
                blob_params.filterByInertia = True
                blob_params.minInertiaRatio = 0.01
                
                # Crear detector con los parámetros
                blob_detector = cv2.SimpleBlobDetector_create(blob_params)
                print("[INFO] Detector de blobs para círculos inicializado correctamente")
            except Exception as e:
                print(f"[WARNING] No se pudo inicializar detector de blobs: {e}")
                blob_detector = None
        
        # Para ChArUco, necesitamos parámetros adicionales
        charuco_board = None
        charuco_detector = None
        if pattern_type == "charuco":
            try:
                # Crear diccionario de marcadores ArUco según el especificado
                ARUCO_DICT = {
                    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
                    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
                    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
                    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
                    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
                    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
                    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
                    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
                    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
                    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
                    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
                    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
                    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
                    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
                    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
                    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
                    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
                    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
                    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
                    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
                    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
                }
                
                # Función para detectar automáticamente el diccionario ArUco
                def detect_aruco_dict(gray_frame):
                    """Detecta automáticamente el diccionario ArUco probando varios"""
                    if aruco_dict_name.lower() != "auto":
                        if aruco_dict_name not in ARUCO_DICT:
                            raise ValueError(f"Diccionario ArUco no válido: {aruco_dict_name}")
                        # Usar la API moderna de ArUco
                        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dict_name])
                        return aruco_dict
                
                    # Lista de diccionarios a probar (ordenados por probabilidad de uso)
                    test_dicts = [
                        "DICT_6X6_250", "DICT_5X5_250", "DICT_4X4_250",
                        "DICT_6X6_100", "DICT_5X5_100", "DICT_4X4_100",
                        "DICT_6X6_50", "DICT_5X5_50", "DICT_4X4_50",
                        "DICT_ARUCO_ORIGINAL"
                    ]
                    
                    print("[INFO] Detectando automáticamente diccionario ArUco...")
                    for dict_name in test_dicts:
                        try:
                            # Usar la API moderna de ArUco
                            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_name])
                            # Crear un detector temporal para probar
                            temp_board = cv2.aruco.CharucoBoard(
                                size=(checkerboard_cols, checkerboard_rows),
                                squareLength=square_size_mm,
                                markerLength=square_size_mm * 0.7
                            )
                            # Asignar el diccionario al tablero
                            temp_board.setDictionary(aruco_dict)
                            
                            temp_params = cv2.aruco.CharucoParameters()
                            temp_params.minMarkers = 2
                            # Crear detector usando la API moderna
                            temp_detector = cv2.aruco.CharucoDetector(temp_board)
                            temp_detector.setDetectorParameters(temp_params)
                            
                            # Probar detección
                            charuco_corners, charuco_ids, marker_corners, marker_ids = temp_detector.detectBoard(gray_frame)
                            if charuco_corners is not None and len(charuco_corners) > 0:
                                print(f"[SUCCESS] Diccionario ArUco detectado automáticamente: {dict_name}")
                                return aruco_dict
                        except Exception as e:
                            continue
                    
                    # Si no se detecta ninguno, usar el por defecto
                    print("[WARNING] No se pudo detectar automáticamente el diccionario ArUco, usando DICT_6X6_250")
                    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
                
                # Obtener un frame de muestra para la detección automática
                sample_frame = None
                sample_timeout = 5.0  # segundos
                sample_start = time.time()
                
                # Esperar a obtener un frame válido para la detección
                while sample_frame is None and (time.time() - sample_start) < sample_timeout:
                    left_ret, left_frame = left_stream.get_frame()
                    if left_ret and left_frame is not None and left_frame.size > 0:
                        sample_frame = left_frame.copy()
                        break
                    time.sleep(0.1)
                
                if sample_frame is None:
                    print("[WARNING] No se pudo obtener frame de muestra para detección automática, usando diccionario por defecto")
                    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(aruco_dict_name, cv2.aruco.DICT_6X6_250))
                else:
                    # Convertir a escala de grises para la detección
                    gray_sample = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
                    # Detectar diccionario ArUco
                    aruco_dict = detect_aruco_dict(gray_sample)
                
                # Crear el tablero ChArUco con parámetros específicos usando la API moderna
                charuco_board = cv2.aruco.CharucoBoard(
                    size=(checkerboard_cols, checkerboard_rows),
                    squareLength=square_size_mm,  # En milímetros directamente
                    markerLength=square_size_mm * 0.7  # 70% del tamaño del cuadrado
                )
                # Asignar el diccionario al tablero
                charuco_board.setDictionary(aruco_dict)
                
                # Crear parámetros de detección ChArUco
                charuco_params = cv2.aruco.CharucoParameters()
                charuco_params.minMarkers = 2  # Mínimo número de marcadores para interpolación
                
                # Crear detector ChArUco usando la API moderna
                charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
                charuco_detector.setDetectorParameters(charuco_params)
                
                print(f"[INFO] Detector ChArUco inicializado correctamente con tablero {checkerboard_cols}x{checkerboard_rows}")
                print(f"[INFO] Tamaño de cuadrado: {square_size_mm}mm, Tamaño de marcador: {square_size_mm * 0.7}mm")
            except Exception as e:
                print(f"[WARNING] No se pudo inicializar ChArUco: {e}")
                print("[WARNING] Usando detección de tablero de ajedrez como fallback")
                charuco_board = None
                charuco_detector = None
                pattern_type = "chessboard"  # Fallback a chessboard
        
        captured_count = 0
        print(f"[INFO] Capturando {num_images} pares de imágenes...")
        if auto_capture:
            print("[INFO] Captura automática cuando se detecten ambas cuadrículas")
        else:
            print("[INFO] Presiona BARRA ESPACIADORA para capturar pares de imágenes")
        print(f"[INFO] Tipo de patrón: {pattern_type}")
        
        # Estado de detección
        right_found = False
        left_found = False
        right_corners = None
        left_corners = None
        
        # Contadores para diagnóstico
        frame_counter = 0
        last_window_update = 0
        last_capture_time = 0
        capture_delay = 1.0  # Segundos entre capturas automáticas
        
        while captured_count < num_images:
            frame_counter += 1
            
            # Obtener frames de ambas cámaras
            right_ret, right_frame = right_stream.get_frame()
            left_ret, left_frame = left_stream.get_frame()
            
            # Debug de frames
            # print(f"[DEBUG] Frame derecho: ret={right_ret}, size={'N/A' if right_frame is None else right_frame.shape}")
            # print(f"[DEBUG] Frame izquierdo: ret={left_ret}, size={'N/A' if left_frame is None else left_frame.shape}")
            
            # Usar frames válidos o crear frames negros si no hay datos
            if not right_ret or right_frame is None or right_frame.size == 0:
                print(f"[DEBUG] Frame derecho es None o inválido (contador: {frame_counter})")
                right_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            if not left_ret or left_frame is None or left_frame.size == 0:
                print(f"[DEBUG] Frame izquierdo es None o inválido (contador: {frame_counter})")
                left_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                
            # Redimensionar frames a 1280x720 si es necesario
            target_width, target_height = 1280, 720
            if right_frame.shape[0] != target_height or right_frame.shape[1] != target_width:
                right_frame = cv2.resize(right_frame, (target_width, target_height))
            if left_frame.shape[0] != target_height or left_frame.shape[1] != target_width:
                left_frame = cv2.resize(left_frame, (target_width, target_height))
                
            # Convertir a escala de grises
            try:
                gray_right = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)
                gray_left = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            except:
                h, w = right_frame.shape[:2] if right_frame is not None and right_frame.size > 0 else (target_height, target_width)
                gray_right = np.zeros((h, w), dtype=np.uint8)
                h, w = left_frame.shape[:2] if left_frame is not None and left_frame.size > 0 else (target_height, target_width)
                gray_left = np.zeros((h, w), dtype=np.uint8)
            
            # Detectar patrones según el tipo especificado
            if gray_right.size > 0:
                if pattern_type == "chessboard":
                    try:
                        right_found, right_corners = cv2.findChessboardCorners(
                            gray_right, (checkerboard_cols, checkerboard_rows), 
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
                    except:
                        right_found, right_corners = detector.detect_pattern(
                            gray_right, (checkerboard_cols, checkerboard_rows), "right")
                elif pattern_type == "circles":
                    # Usar el enfoque mejorado con detector de blobs para círculos asimétricos
                    try:
                        if blob_detector is not None:
                            # Detectar blobs
                            keypoints = blob_detector.detect(gray_right)
                            
                            # Dibujar blobs detectados (solo para visualización, no afecta la detección)
                            # Esto ayuda a cv2.findCirclesGrid() a encontrar la cuadrícula
                            if len(keypoints) > 0:
                                im_with_keypoints = cv2.drawKeypoints(right_frame, keypoints, np.array([]), 
                                                                    (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
                                
                                # Buscar la cuadrícula de círculos
                                right_found, right_corners = cv2.findCirclesGrid(
                                    im_with_keypoints_gray, (checkerboard_cols, checkerboard_rows), 
                                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                            else:
                                # Si no se detectan blobs, intentar la detección directa
                                right_found, right_corners = cv2.findCirclesGrid(
                                    gray_right, (checkerboard_cols, checkerboard_rows), 
                                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                        else:
                            # Fallback si no se inicializó el detector de blobs
                            right_found, right_corners = cv2.findCirclesGrid(
                                gray_right, (checkerboard_cols, checkerboard_rows), 
                                flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                    except Exception as e:
                        print(f"[WARNING] Error en detección de círculos cámara derecha: {e}")
                        # Fallback a detección simple
                        try:
                            right_found, right_corners = cv2.findCirclesGrid(
                                gray_right, (checkerboard_cols, checkerboard_rows), 
                                flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                        except:
                            right_found = False
                            right_corners = None
                elif pattern_type == "charuco" and charuco_detector is not None:
                    # Usar el detector ChArUco correcto según las especificaciones (API moderna)
                    try:
                        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray_right)
                        right_found = charuco_corners is not None and len(charuco_corners) > 0
                        right_corners = charuco_corners if right_found else None
                        # Añadir información de depuración
                        if right_found:
                            print(f"[DEBUG] ChArUco detectado en cámara derecha: {len(charuco_corners)} esquinas")
                        else:
                            print(f"[DEBUG] ChArUco no detectado en cámara derecha")
                    except Exception as e:
                        print(f"[WARNING] Error en detección ChArUco cámara derecha: {e}")
                        right_found = False
                        right_corners = None
                elif pattern_type == "charuco":
                    # Fallback si no se inicializó correctamente el detector ChArUco
                    right_found, right_corners = detector.detect_pattern(
                        gray_right, (checkerboard_cols, checkerboard_rows), "right")
                elif pattern_type == "bricks":
                    # Para patrón bricks, usamos una aproximación similar al tablero de ajedrez
                    # pero con ajustes para la estructura específica
                    try:
                        right_found, right_corners = cv2.findChessboardCorners(
                            gray_right, (checkerboard_cols, checkerboard_rows), 
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
                        # Si no se encuentra el patrón completo, intentamos con variaciones
                        if not right_found:
                            # Intentar con un tamaño ligeramente diferente
                            right_found, right_corners = cv2.findChessboardCorners(
                                gray_right, (checkerboard_cols-1, checkerboard_rows), 
                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
                    except Exception as e:
                        print(f"[WARNING] Error en detección de patrón bricks cámara derecha: {e}")
                        right_found = False
                        right_corners = None
                else:
                    right_found = False
                    right_corners = None
                    
            if gray_left.size > 0:
                if pattern_type == "chessboard":
                    try:
                        left_found, left_corners = cv2.findChessboardCorners(
                            gray_left, (checkerboard_cols, checkerboard_rows), 
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
                    except:
                        left_found, left_corners = detector.detect_pattern(
                            gray_left, (checkerboard_cols, checkerboard_rows), "left")
                elif pattern_type == "circles":
                    # Usar el enfoque mejorado con detector de blobs para círculos asimétricos
                    try:
                        if blob_detector is not None:
                            # Detectar blobs
                            keypoints = blob_detector.detect(gray_left)
                            
                            # Dibujar blobs detectados (solo para visualización, no afecta la detección)
                            # Esto ayuda a cv2.findCirclesGrid() a encontrar la cuadrícula
                            if len(keypoints) > 0:
                                im_with_keypoints = cv2.drawKeypoints(left_frame, keypoints, np.array([]), 
                                                                    (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                                im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
                                
                                # Buscar la cuadrícula de círculos
                                left_found, left_corners = cv2.findCirclesGrid(
                                    im_with_keypoints_gray, (checkerboard_cols, checkerboard_rows), 
                                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                            else:
                                # Si no se detectan blobs, intentar la detección directa
                                left_found, left_corners = cv2.findCirclesGrid(
                                    gray_left, (checkerboard_cols, checkerboard_rows), 
                                    flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                        else:
                            # Fallback si no se inicializó el detector de blobs
                            left_found, left_corners = cv2.findCirclesGrid(
                                gray_left, (checkerboard_cols, checkerboard_rows), 
                                flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                    except Exception as e:
                        print(f"[WARNING] Error en detección de círculos cámara izquierda: {e}")
                        # Fallback a detección simple
                        try:
                            left_found, left_corners = cv2.findCirclesGrid(
                                gray_left, (checkerboard_cols, checkerboard_rows), 
                                flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
                        except:
                            left_found = False
                            left_corners = None
                elif pattern_type == "charuco" and charuco_detector is not None:
                    # Usar el detector ChArUco correcto según las especificaciones
                    try:
                        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray_left)
                        left_found = charuco_corners is not None and len(charuco_corners) > 0
                        left_corners = charuco_corners if left_found else None
                        # Añadir información de depuración
                        if left_found:
                            print(f"[DEBUG] ChArUco detectado en cámara izquierda: {len(charuco_corners)} esquinas")
                        else:
                            print(f"[DEBUG] ChArUco no detectado en cámara izquierda")
                    except Exception as e:
                        print(f"[WARNING] Error en detección ChArUco cámara izquierda: {e}")
                        left_found = False
                        left_corners = None
                elif pattern_type == "charuco":
                    # Fallback si no se inicializó correctamente el detector ChArUco
                    left_found, left_corners = detector.detect_pattern(
                        gray_left, (checkerboard_cols, checkerboard_rows), "left")
                elif pattern_type == "bricks":
                    # Para patrón bricks, usamos una aproximación similar al tablero de ajedrez
                    try:
                        left_found, left_corners = cv2.findChessboardCorners(
                            gray_left, (checkerboard_cols, checkerboard_rows), 
                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
                        # Si no se encuentra el patrón completo, intentamos con variaciones
                        if not left_found:
                            # Intentar con un tamaño ligeramente diferente
                            left_found, left_corners = cv2.findChessboardCorners(
                                gray_left, (checkerboard_cols-1, checkerboard_rows), 
                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)
                    except Exception as e:
                        print(f"[WARNING] Error en detección de patrón bricks cámara izquierda: {e}")
                        left_found = False
                        left_corners = None
                else:
                    left_found = False
                    left_corners = None
            
            # Combinar frames para mostrar en una sola ventana
            display_combined = combine_frames(left_frame.copy(), right_frame.copy())
            
            # Dibujar esquinas si se encuentran patrones
            if right_found and right_corners is not None:
                try:
                    if pattern_type == "chessboard":
                        # Crear una copia de las esquinas y ajustar las coordenadas X para la imagen derecha
                        right_corners_adjusted = right_corners.copy()
                        right_corners_adjusted[:, :, 0] += target_width  # Añadir offset de ancho para posición correcta
                        cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                right_corners_adjusted, right_found)
                    elif pattern_type == "charuco":
                        # Para ChArUco, los corners son puntos simples, no tienen dimensión adicional
                        # Dibujar los corners detectados con offset
                        if len(right_corners.shape) == 2 and right_corners.shape[1] == 2:
                            # Ajustar coordenadas X para la imagen derecha
                            right_corners_display = right_corners.copy()
                            right_corners_display[:, 0] += target_width  # Añadir offset de ancho
                            
                            # Dibujar corners ChArUco (puntos simples)
                            for point in right_corners_display:
                                cv2.circle(display_combined, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                        elif len(right_corners.shape) == 3:
                            # Formato alternativo de corners
                            right_corners_adjusted = right_corners.copy()
                            right_corners_adjusted[:, :, 0] += target_width
                            cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                    right_corners_adjusted, right_found)
                    elif pattern_type == "circles":
                        # Crear una copia de las esquinas y ajustar las coordenadas X para la imagen derecha
                        right_corners_adjusted = right_corners.copy()
                        right_corners_adjusted[:, :, 0] += target_width  # Añadir offset de ancho para posición correcta
                        cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                right_corners_adjusted, right_found)
                    elif pattern_type == "bricks":
                        # Para patrón bricks, usar el mismo enfoque que el tablero de ajedrez
                        right_corners_adjusted = right_corners.copy()
                        right_corners_adjusted[:, :, 0] += target_width  # Añadir offset de ancho para posición correcta
                        cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                right_corners_adjusted, right_found)
                except Exception as e:
                    print(f"[WARNING] Error dibujando esquinas derecha: {e}")
            if left_found and left_corners is not None:
                try:
                    if pattern_type == "chessboard":
                        cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                left_corners, left_found)
                    elif pattern_type == "charuco":
                        # Para ChArUco, los corners son puntos simples
                        if len(left_corners.shape) == 2 and left_corners.shape[1] == 2:
                            # Dibujar corners ChArUco (puntos simples)
                            for point in left_corners:
                                cv2.circle(display_combined, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
                        elif len(left_corners.shape) == 3:
                            # Formato alternativo de corners
                            cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                    left_corners, left_found)
                    elif pattern_type == "circles":
                        cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                left_corners, left_found)
                    elif pattern_type == "bricks":
                        # Para patrón bricks, usar el mismo enfoque que el tablero de ajedrez
                        cv2.drawChessboardCorners(display_combined, (checkerboard_cols, checkerboard_rows), 
                                                left_corners, left_found)
                except Exception as e:
                    print(f"[WARNING] Error dibujando esquinas izquierda: {e}")
            
            # Agregar información a la imagen combinada
            cv2.putText(display_combined, f"Pares capturados: {captured_count}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_combined, f"Derecha (B-UV) FPS: {right_stream.get_fps():.1f} | Izquierda (A-Laser) FPS: {left_stream.get_fps():.1f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_combined, f"Patron Derecha: {'SI' if right_found else 'NO'} | Patron Izquierda: {'SI' if left_found else 'NO'}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 255, 0) if (right_found and left_found) else (0, 0, 255), 2)
            cv2.putText(display_combined, f"Tipo: {pattern_type}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Mostrar frame combinado en una sola ventana
            try:
                cv2.imshow('Streams Combinados - Izquierda (A-Laser) | Derecha (B-UV)', display_combined)
            except Exception as e:
                print(f"[ERROR] Error al mostrar ventana: {e}")
            
            # Captura automática o manual cuando se detectan ambas cuadrículas
            current_time = time.time()
            capture_triggered = False
            
            if auto_capture:
                # Captura automática
                if right_found and left_found and (current_time - last_capture_time) > capture_delay:
                    print(f"[INFO] Capturando par {captured_count + 1} automáticamente...")
                    capture_triggered = True
            else:
                # Captura manual con barra espaciadora
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Barra espaciadora
                        if right_found and left_found:
                            print(f"[INFO] Capturando par {captured_count + 1} manualmente...")
                            capture_triggered = True
                        else:
                            print("[INFO] No se detectaron ambas cuadrículas, no se puede capturar")
                    elif key == ord('q'):
                        print("[INFO] Salida solicitada")
                        break
                    elif key == ord('+') and right_stream.cap and right_stream.cap.isOpened():
                        # Aumentar brillo de la cámara UV
                        current_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        new_brightness = min(1.0, current_brightness + 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
                        actual_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        if success:
                            if abs(actual_brightness - new_brightness) <= 0.01:
                                print(f"[INFO] Brillo UV aumentado a: {actual_brightness:.2f}")
                            else:
                                print(f"[WARNING] Brillo UV ajustado pero con diferencia. Solicitado: {new_brightness:.2f}, Actual: {actual_brightness:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar brillo UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_brightness:.2f}")
                    elif key == ord('-') and right_stream.cap and right_stream.cap.isOpened():
                        # Disminuir brillo de la cámara UV
                        current_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        new_brightness = max(-1.0, current_brightness - 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
                        actual_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        if success:
                            if abs(actual_brightness - new_brightness) <= 0.01:
                                print(f"[INFO] Brillo UV disminuido a: {actual_brightness:.2f}")
                            else:
                                print(f"[WARNING] Brillo UV ajustado pero con diferencia. Solicitado: {new_brightness:.2f}, Actual: {actual_brightness:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar brillo UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_brightness:.2f}")
                    elif key == ord('c') and right_stream.cap and right_stream.cap.isOpened():
                        # Aumentar contraste de la cámara UV
                        current_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        new_contrast = min(1.0, current_contrast + 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_CONTRAST, new_contrast)
                        actual_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        if success:
                            if abs(actual_contrast - new_contrast) <= 0.01:
                                print(f"[INFO] Contraste UV aumentado a: {actual_contrast:.2f}")
                            else:
                                print(f"[WARNING] Contraste UV ajustado pero con diferencia. Solicitado: {new_contrast:.2f}, Actual: {actual_contrast:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar contraste UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_contrast:.2f}")
                    elif key == ord('x') and right_stream.cap and right_stream.cap.isOpened():
                        # Disminuir contraste de la cámara UV
                        current_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        new_contrast = max(-1.0, current_contrast - 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_CONTRAST, new_contrast)
                        actual_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        if success:
                            if abs(actual_contrast - new_contrast) <= 0.01:
                                print(f"[INFO] Contraste UV disminuido a: {actual_contrast:.2f}")
                            else:
                                print(f"[WARNING] Contraste UV ajustado pero con diferencia. Solicitado: {new_contrast:.2f}, Actual: {actual_contrast:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar contraste UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_contrast:.2f}")
                except Exception as e:
                    print(f"[ERROR] Error al procesar tecla: {e}")
        
            if capture_triggered:
                try:
                    # Para ChArUco, usar matchImagePoints para obtener puntos correspondientes
                    if pattern_type == "charuco" and charuco_board is not None:
                        try:
                            # Procesar cámara derecha
                            right_obj_points = None
                            right_img_points = None
                            if right_corners is not None and len(right_corners) > 0:
                                # Generar IDs para las esquinas detectadas
                                charuco_ids_right = np.arange(len(right_corners))
                                right_obj_points, right_img_points = charuco_board.matchImagePoints(
                                    right_corners, charuco_ids_right)
                            
                            # Procesar cámara izquierda
                            left_obj_points = None
                            left_img_points = None
                            if left_corners is not None and len(left_corners) > 0:
                                # Generar IDs para las esquinas detectadas
                                charuco_ids_left = np.arange(len(left_corners))
                                left_obj_points, left_img_points = charuco_board.matchImagePoints(
                                    left_corners, charuco_ids_left)
                            
                            # Añadir puntos si ambos son válidos
                            if right_img_points is not None and len(right_img_points) > 0 and \
                               left_img_points is not None and len(left_img_points) > 0:
                                imgpoints_right.append(right_img_points)
                                imgpoints_left.append(left_img_points)
                                # Usar los puntos de objeto de la cámara derecha (son los mismos para ambos)
                                objpoints.append(right_obj_points)
                                print(f"[DEBUG] ChArUco puntos añadidos: {len(right_img_points)} puntos imagen derecha, {len(left_img_points)} puntos imagen izquierda")
                            else:
                                print("[DEBUG] ChArUco matchImagePoints no devolvió puntos válidos")
                        except Exception as e:
                            print(f"[WARNING] Error en matchImagePoints para ChArUco: {e}")
                            # Fallback a la forma estándar
                            if right_corners is not None:
                                imgpoints_right.append(right_corners.copy())
                            if left_corners is not None:
                                imgpoints_left.append(left_corners.copy())
                            objpoints.append(objp.copy())
                    else:
                        # Para otros patrones, usar el enfoque estándar
                        # Aplicar refinamiento específico por tipo de patrón
                        if right_corners is not None:
                            if pattern_type == "chessboard":
                                # Refinamiento para tablero de ajedrez
                                right_corners_refined = cv2.cornerSubPix(gray_right, right_corners, (11, 11), (-1, -1), criteria)
                                imgpoints_right.append(right_corners_refined.copy() if right_corners_refined is not None else right_corners.copy())
                            elif pattern_type == "charuco":
                                # Los corners de ChArUco ya están interpolados, no necesitan subpixel
                                imgpoints_right.append(right_corners.copy())
                            elif pattern_type == "circles":
                                # Para círculos no se necesita subpixel
                                imgpoints_right.append(right_corners.copy())
                            elif pattern_type == "bricks":
                                # Refinamiento para patrón bricks (similar al tablero de ajedrez)
                                right_corners_refined = cv2.cornerSubPix(gray_right, right_corners, (11, 11), (-1, -1), criteria)
                                imgpoints_right.append(right_corners_refined.copy() if right_corners_refined is not None else right_corners.copy())
                        if left_corners is not None:
                            if pattern_type == "chessboard":
                                # Refinamiento para tablero de ajedrez
                                left_corners_refined = cv2.cornerSubPix(gray_left, left_corners, (11, 11), (-1, -1), criteria)
                                imgpoints_left.append(left_corners_refined.copy() if left_corners_refined is not None else left_corners.copy())
                            elif pattern_type == "charuco":
                                # Los corners de ChArUco ya están interpolados, no necesitan subpixel
                                imgpoints_left.append(left_corners.copy())
                            elif pattern_type == "circles":
                                # Para círculos no se necesita subpixel
                                imgpoints_left.append(left_corners.copy())
                            elif pattern_type == "bricks":
                                # Refinamiento para patrón bricks (similar al tablero de ajedrez)
                                left_corners_refined = cv2.cornerSubPix(gray_left, left_corners, (11, 11), (-1, -1), criteria)
                                imgpoints_left.append(left_corners_refined.copy() if left_corners_refined is not None else left_corners.copy())
                        
                        # Añadir puntos de objeto
                        objpoints.append(objp.copy())
                    
                    cv2.imwrite(os.path.join(temp_dir, f"right_{captured_count:02d}.jpg"), right_frame)
                    cv2.imwrite(os.path.join(temp_dir, f"left_{captured_count:02d}.jpg"), left_frame)
                    
                    captured_count += 1
                    last_capture_time = current_time
                    print(f"[INFO] Par {captured_count} capturado")
                    
                except Exception as e:
                    print(f"[ERROR] Error al capturar: {str(e)}")
            
            # Procesar teclas para salir o ajustar configuración (solo en modo manual)
            if not auto_capture:
                try:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[INFO] Salida solicitada")
                        break
                    elif key == ord('+') and right_stream.cap and right_stream.cap.isOpened():
                        # Aumentar brillo de la cámara UV
                        current_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        new_brightness = min(1.0, current_brightness + 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
                        actual_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        if success:
                            if abs(actual_brightness - new_brightness) <= 0.01:
                                print(f"[INFO] Brillo UV aumentado a: {actual_brightness:.2f}")
                            else:
                                print(f"[WARNING] Brillo UV ajustado pero con diferencia. Solicitado: {new_brightness:.2f}, Actual: {actual_brightness:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar brillo UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_brightness:.2f}")
                    elif key == ord('-') and right_stream.cap and right_stream.cap.isOpened():
                        # Disminuir brillo de la cámara UV
                        current_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        new_brightness = max(-1.0, current_brightness - 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_BRIGHTNESS, new_brightness)
                        actual_brightness = right_stream.cap.get(cv2.CAP_PROP_BRIGHTNESS)
                        if success:
                            if abs(actual_brightness - new_brightness) <= 0.01:
                                print(f"[INFO] Brillo UV disminuido a: {actual_brightness:.2f}")
                            else:
                                print(f"[WARNING] Brillo UV ajustado pero con diferencia. Solicitado: {new_brightness:.2f}, Actual: {actual_brightness:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar brillo UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_brightness:.2f}")
                    elif key == ord('c') and right_stream.cap and right_stream.cap.isOpened():
                        # Aumentar contraste de la cámara UV
                        current_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        new_contrast = min(1.0, current_contrast + 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_CONTRAST, new_contrast)
                        actual_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        if success:
                            if abs(actual_contrast - new_contrast) <= 0.01:
                                print(f"[INFO] Contraste UV aumentado a: {actual_contrast:.2f}")
                            else:
                                print(f"[WARNING] Contraste UV ajustado pero con diferencia. Solicitado: {new_contrast:.2f}, Actual: {actual_contrast:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar contraste UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_contrast:.2f}")
                    elif key == ord('x') and right_stream.cap and right_stream.cap.isOpened():
                        # Disminuir contraste de la cámara UV
                        current_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        new_contrast = max(-1.0, current_contrast - 0.1)
                        success = right_stream.cap.set(cv2.CAP_PROP_CONTRAST, new_contrast)
                        actual_contrast = right_stream.cap.get(cv2.CAP_PROP_CONTRAST)
                        if success:
                            if abs(actual_contrast - new_contrast) <= 0.01:
                                print(f"[INFO] Contraste UV disminuido a: {actual_contrast:.2f}")
                            else:
                                print(f"[WARNING] Contraste UV ajustado pero con diferencia. Solicitado: {new_contrast:.2f}, Actual: {actual_contrast:.2f}")
                        else:
                            print(f"[ERROR] No se pudo ajustar contraste UV. La cámara puede no soportar esta propiedad.")
                            print(f"[INFO] Valor actual: {actual_contrast:.2f}")
                except Exception as e:
                    print(f"[ERROR] Error al procesar tecla: {e}")
                
        # Detener streams
        right_stream.stop()
        left_stream.stop()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        # Verificar capturas
        if captured_count < 3:
            raise RuntimeError(f"Insuficientes pares capturados: {captured_count}")
            
        print(f"[INFO] Calibrando con {captured_count} pares...")
        
        # Obtener tamaño de imagen para calibración (forzar 1280x720)
        img_size = (1280, 720)
        
        # Calibración
        try:
            ret_right, K_right, dist_right, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints_right, img_size, None, None)
            ret_left, K_left, dist_left, _, _ = cv2.calibrateCamera(
                objpoints, imgpoints_left, img_size, None, None)
                
            if not ret_right or not ret_left:
                raise RuntimeError("Error en calibración individual")
                
            print("[INFO] Calibración estéreo...")
            # Usar flags optimizados
            flags = (cv2.CALIB_FIX_INTRINSIC |  # Usar intrínsecos previamente calibrados
                     cv2.CALIB_RATIONAL_MODEL)   # Modelo de distorsión más preciso
            
            ret, K_right, dist_right, K_left, dist_left, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_right, imgpoints_left,
                K_right, dist_right, K_left, dist_left,
                img_size, criteria=criteria, flags=flags)
                
            if not ret:
                raise RuntimeError("Error en calibración estéreo")
                
            # Calcular errores de reproyección
            errors_left, errors_right = calculate_reprojection_errors(
                objpoints, imgpoints_left, imgpoints_right,
                K_left, dist_left, K_right, dist_right, R, T)
            
            # Validar calibración
            validate_calibration(ret, F, img_size, errors_left, errors_right)
            
            # Resumen detallado
            print_calibration_summary(K_left, K_right, dist_left, dist_right, R, T, ret, img_size)
                
            print("[INFO] Calibración completada")
            return img_size, K_right, dist_right, K_left, dist_left, R, T
            
        except Exception as e:
            raise RuntimeError(f"Error calibración: {str(e)}")
            
    except Exception as e:
        try:
            right_stream.stop()
        except:
            pass
        try:
            left_stream.stop()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        raise e
        
    finally:
        # No eliminar el directorio de imágenes para que queden guardadas
        print(f"[INFO] Imágenes de calibración guardadas en: {temp_dir}")
        pass


def main():
    parser = argparse.ArgumentParser(description="Calibración estéreo con dos cámaras")
    parser.add_argument("--left", type=int, default=0, help="Índice cámara izquierda (A - laser, por defecto 0)")
    parser.add_argument("--right", type=int, default=1, help="Índice cámara derecha (B - UV, por defecto 1)")
    parser.add_argument("--rows", type=int, default=6, help="Filas tablero")
    parser.add_argument("--cols", type=int, default=9, help="Columnas tablero")
    parser.add_argument("--square-size", type=float, default=25.0, help="Tamaño cuadrado (mm)")
    parser.add_argument("--images", type=int, default=15, help="Número de pares")
    parser.add_argument("--output", type=str, default="stereo_calibration.txt", help="Archivo salida")
    parser.add_argument("--template", type=str, default="calibJMS1006207.txt", help="Archivo plantilla SEAL")
    parser.add_argument("--dev-id", type=str, default="JMS1006207", help="ID del dispositivo")
    parser.add_argument("--test", action="store_true", help="Modo test para debuggear cámaras individuales")
    parser.add_argument("--pattern-type", type=str, default="chessboard", choices=["chessboard", "circles", "charuco", "bricks"], 
                       help="Tipo de patrón de calibración: chessboard, circles, charuco, bricks")
    parser.add_argument("--uv-brightness", type=int, default=-1, help="Brillo para la cámara UV (-1 para no cambiar)")
    parser.add_argument("--uv-contrast", type=int, default=-1, help="Contraste para la cámara UV (-1 para no cambiar)")
    parser.add_argument("--fps", type=float, default=None, help="FPS objetivo para las cámaras (por defecto usa FPS nativo)")
    parser.add_argument("--aruco-dict", type=str, default="DICT_6X6_250", 
                       choices=["auto", "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
                               "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
                               "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
                               "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
                               "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
                               "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"],
                       help="Diccionario ArUco para detección ChArUco (usar 'auto' para detección automática)")
    parser.add_argument("--no-auto-capture", action="store_true", 
                       help="Deshabilitar captura automática y usar barra espaciadora para capturar")
    parser.add_argument("--from-images", type=str, default=None, 
                       help="Procesar imágenes existentes desde directorio (ej: calib_imgs)")
    # Se elimina el argumento de resolución ya que se usará la resolución máxima de la cámara
    
    args = parser.parse_args()
    
    if args.test:
        print("[TEST] Modo de prueba individual de cámaras")
        print("1. Probando cámara 0 (Laser/A)...")
        test_single_camera(0, "Laser (A)")
        print("2. Probando cámara 1 (UV/B)...")
        test_single_camera(1, "UV (B)")
        return 0
    
    try:
        # Modo procesamiento de imágenes existentes
        if args.from_images:
            print(f"[INFO] Modo: Procesamiento de imágenes existentes")
            print(f"  Directorio: {args.from_images}")
            print(f"  Tipo de patrón: {args.pattern_type}")
            
            img_size, K_right, dist_right, K_left, dist_left, R, T = process_existing_images(
                args.from_images, args.rows, args.cols, args.square_size, 
                pattern_type=args.pattern_type)
        else:
            # Modo calibración en vivo
            print(f"[INFO] Calibración estéreo:")
            print(f"  Cámara A (laser) = izquierda = índice {args.left}")
            print(f"  Cámara B (UV) = derecha = índice {args.right}")
            print(f"  Tipo de patrón: {args.pattern_type}")
            print(f"  FPS objetivo: {args.fps if args.fps else 'Usar FPS nativo'}")
            if args.pattern_type == "charuco":
                print(f"  Diccionario ArUco: {args.aruco_dict}")
            print(f"  Captura automática: {'No' if args.no_auto_capture else 'Sí'}")
            print(f"[INFO] Si ves el error 'not authorized to capture video', otorga permisos de cámara a Terminal/Python")
            
            img_size, K_right, dist_right, K_left, dist_left, R, T = calibrate_stereo_cameras(
                args.left, args.right, args.rows, args.cols, args.square_size, args.images, 
                pattern_type=args.pattern_type, uv_brightness=args.uv_brightness, uv_contrast=args.uv_contrast,
                resolution=None, target_fps=args.fps, aruco_dict_name=args.aruco_dict,
                auto_capture=not args.no_auto_capture)
        
        # Guardar resultados en formato técnico
        save_calibration_results(img_size, K_right, dist_right, K_left, dist_left, R, T, args.output)
        
        # Guardar resultados en formato SEAL con resolución fija 1280x720
        seal_output_file = args.output.replace(".txt", "_seal.txt")
        save_seal_calibration_file(img_size, K_left, dist_left, K_right, dist_right, R, T,
                                 args.template, seal_output_file, args.dev_id,
                                 square_size_mm=args.square_size, rows=args.rows, cols=args.cols)
        
        print("\n[RESULTADOS]")
        print(f"Resolución: {img_size}")
        print(f"K derecha (cámara B - UV):\n{K_right}")
        print(f"K izquierda (cámara A - laser):\n{K_left}")
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        print(f"[ERROR] Posible solución: Otorga permisos de cámara a Terminal/Python en Preferencias del Sistema")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())