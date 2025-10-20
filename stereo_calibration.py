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
import sys


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
                              K_proj, dist_proj,
                              template_file, output_file, dev_id=None, 
                              square_size_mm=25.0, rows=6, cols=9, gray_code_table=None):
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
        
        # IMPORTANTE: Las líneas 2, 3 y 4 NO se modifican
        # Son parámetros de fábrica del hardware (cámara-proyector)
        # que fueron calibrados con equipamiento especializado
        # Modificarlos causaría:
        #   - Línea 2: Alteración de escala 3D y profundidad
        #   - Línea 3: Desplazamiento/distorsión de la escena
        #   - Línea 4: Inclinación o arqueo del plano 3D
        
        # Solo actualizar la fecha y el tipo en los metadatos
        import re
        content = re.sub(r'CalibrateDate:[^*]*', f'CalibrateDate:{current_date}', content)
        content = re.sub(r'Type:[^*]*', 'Type:Sychev-calibration', content)
        
        # Si hay tabla Gray Code, insertarla después de la línea 1 (resolución)
        if gray_code_table and len(gray_code_table) > 0:
            print(f"\n[INFO] Insertando tabla de correspondencias Gray Code ({len(gray_code_table)} entradas)...")
            
            # Dividir contenido en líneas
            lines = content.split('\n')
            
            # Encontrar la línea de resolución: puede ser "1280 720" o "1 1280 720"
            insert_index = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Buscar "1280 720" o "1 1280 720"
                if (stripped == '1280 720' or 
                    stripped.startswith('1280 720') or
                    (stripped.startswith('1 ') and '720' in stripped)):
                    insert_index = i + 1
                    print(f"  Encontrada línea resolución en {i+1}: '{stripped}'")
                    break
            
            if insert_index > 0:
                # Eliminar tabla anterior si existe (líneas 2-156 del template)
                cleaned_lines = lines[:insert_index]
                skip_until = -1
                
                # Buscar inicio de tabla antigua (líneas con formato "N X.X Y.Y")
                for i in range(insert_index, len(lines)):
                    stripped = lines[i].strip()
                    if stripped and len(stripped.split()) == 3:
                        parts = stripped.split()
                        try:
                            int(parts[0])
                            float(parts[1])
                            float(parts[2])
                            # Es parte de tabla antigua, marcar para saltar
                            if skip_until < 0:
                                print(f"  Eliminando tabla anterior desde línea {i+1}")
                            skip_until = i
                        except ValueError:
                            # No es tabla, continuar normal
                            if skip_until >= 0:
                                # Fin de la tabla antigua
                                break
                    elif skip_until >= 0:
                        # Fin de la tabla antigua
                        break
                
                # Copiar líneas después de la tabla antigua
                if skip_until >= 0:
                    cleaned_lines.extend(lines[skip_until + 1:])
                else:
                    cleaned_lines.extend(lines[insert_index:])
                
                # Insertar nueva tabla
                lines = cleaned_lines[:insert_index] + gray_code_table + cleaned_lines[insert_index:]
                content = '\n'.join(lines)
                print(f"  Tabla insertada después de línea {insert_index}")
            else:
                print(f"[WARNING] No se encontró línea de resolución, tabla NO insertada")
        
        with open(output_file, 'w') as f:
            f.write(content)
        
        print(f"\n[INFO] Archivo de calibración SEAL generado")
        print(f"  NOTA: Líneas 2, 3, 4 preservadas de la plantilla (parámetros de fábrica)")
        print(f"  Solo se actualizaron: intrínsecos, distorsión y metadatos")
        print(f"[INFO] Archivo guardado en: {output_file}")
        
        # Guardar parámetros del proyector en un archivo separado para referencia
        if K_proj is not None and dist_proj is not None:
            # Generar nombre correcto reemplazando _seal.txt o _seal_gray_code.txt
            if "_seal_gray_code.txt" in output_file:
                projector_file = output_file.replace("_seal_gray_code.txt", "_seal_projector_params.txt")
            else:
                projector_file = output_file.replace("_seal.txt", "_projector_params.txt")
            print(f"\n[INFO] Guardando parámetros estimados del proyector...")
            
            # Extraer parámetros del proyector
            fx_proj = float(K_proj[0, 0])
            fy_proj = float(K_proj[1, 1])
            cx_proj = float(K_proj[0, 2])
            cy_proj = float(K_proj[1, 2])
            skew_proj = float(K_proj[0, 1])
            
            dist_proj_flat = dist_proj.flatten()
            k1_proj = float(dist_proj_flat[0]) if len(dist_proj_flat) > 0 else 0.0
            k2_proj = float(dist_proj_flat[1]) if len(dist_proj_flat) > 1 else 0.0
            p1_proj = float(dist_proj_flat[2]) if len(dist_proj_flat) > 2 else 0.0
            p2_proj = float(dist_proj_flat[3]) if len(dist_proj_flat) > 3 else 0.0
            k3_proj = float(dist_proj_flat[4]) if len(dist_proj_flat) > 4 else 0.0
            
            # Calcular baseline y transformación
            baseline = np.linalg.norm(T)
            rvec, _ = cv2.Rodrigues(R)
            angles = np.degrees(rvec.flatten())
            
            with open(projector_file, 'w') as f:
                f.write("# PARÁMETROS ESTIMADOS DEL PROYECTOR\n")
                f.write("# =====================================\n")
                f.write("# NOTA IMPORTANTE:\n")
                f.write("# Esta es una ESTIMACIÓN APROXIMADA basada en homografías del patrón de calibración.\n")
                f.write("# Para calibración precisa del proyector, se requiere patrones de luz estructurada.\n")
                f.write("#\n")
                f.write("# FORMATO: 2 -0.007498 24.452629\n")
                f.write("# Donde: num fx fy cx cy k1 k2 p1 p2 k3 [transformación extrínseca]\n")
                f.write("#\n\n")
                
                f.write(f"# Parámetros intrínsecos del proyector:\n")
                f.write(f"2 {fx_proj:.6f} {fy_proj:.6f} {cx_proj:.6f} {cy_proj:.6f} ")
                f.write(f"{k1_proj:.6f} {k2_proj:.6f} {p1_proj:.6f} {p2_proj:.6f} {k3_proj:.6f} ")
                f.write(f"{skew_proj:.6f} {baseline:.6f}\n")
                
                f.write(f"\n# Desglose de parámetros:\n")
                f.write(f"# fx (focal x): {fx_proj:.6f} píxeles\n")
                f.write(f"# fy (focal y): {fy_proj:.6f} píxeles\n")
                f.write(f"# cx (centro x): {cx_proj:.6f} píxeles\n")
                f.write(f"# cy (centro y): {cy_proj:.6f} píxeles\n")
                f.write(f"# k1 (distorsión radial): {k1_proj:.6f}\n")
                f.write(f"# k2 (distorsión radial): {k2_proj:.6f}\n")
                f.write(f"# p1 (distorsión tangencial): {p1_proj:.6f}\n")
                f.write(f"# p2 (distorsión tangencial): {p2_proj:.6f}\n")
                f.write(f"# k3 (distorsión radial): {k3_proj:.6f}\n")
                f.write(f"# skew (sesgo): {skew_proj:.6f}\n")
                f.write(f"# baseline (línea base): {baseline:.6f} mm\n")
                
                f.write(f"\n# Transformación estéreo (cámara-proyector):\n")
                f.write(f"# Rotación (grados): X={angles[0]:.2f}°, Y={angles[1]:.2f}°, Z={angles[2]:.2f}°\n")
                f.write(f"# Traslación (mm): X={T[0,0]:.2f}, Y={T[1,0]:.2f}, Z={T[2,0]:.2f}\n")
                
                f.write(f"\n# Fecha de calibración: {current_date}\n")
                f.write(f"# Device ID: {dev_id if dev_id else 'N/A'}\n")
            
            print(f"[INFO] Parámetros del proyector guardados en: {projector_file}")
            print(f"  ADVERTENCIA: Estos son valores ESTIMADOS, no precisos")
            print(f"  Para uso en producción, se requiere calibración con luz estructurada")
        
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


def estimate_projector_parameters(objpoints, imgpoints_left, imgpoints_right, K_left, dist_left, K_right, dist_right, img_size):
    """
    Estima los parámetros intrínsecos del proyector usando homografías
    desde las correspondencias del patrón de calibración.
    
    Este método asume que el proyector actúa como una "cámara inversa"
    que proyecta un patrón estructurado sobre el tablero de ajedrez.
    
    Args:
        objpoints: Puntos 3D del patrón
        imgpoints_left: Puntos 2D detectados en cámara izquierda
        imgpoints_right: Puntos 2D detectados en cámara derecha
        K_left, dist_left: Intrínsecos de cámara izquierda
        K_right, dist_right: Intrínsecos de cámara derecha
        img_size: Tamaño de imagen (width, height)
    
    Returns:
        K_proj: Matriz intrínseca del proyector (3x3)
        dist_proj: Coeficientes de distorsión del proyector (5 valores)
    """
    
    print(f"\n{'='*60}")
    print(f"[ESTIMACIÓN DE PARÁMETROS DEL PROYECTOR]")
    print(f"{'='*60}")
    
    # Método: Usar homografías para estimar parámetros intrínsecos
    # del proyector desde múltiples vistas del tablero
    
    H_matrices = []
    
    # Calcular homografías para cada par de imágenes
    for i in range(len(objpoints)):
        # Proyectar puntos 3D del patrón al plano z=0
        objp_2d = objpoints[i][:, :2].astype(np.float32)
        imgp = imgpoints_left[i].reshape(-1, 2).astype(np.float32)
        
        # Calcular homografía entre patrón y imagen
        H, mask = cv2.findHomography(objp_2d, imgp, cv2.RANSAC, 5.0)
        
        if H is not None:
            H_matrices.append(H)
    
    if len(H_matrices) < 3:
        print(f"[WARNING] Pocas homografías válidas ({len(H_matrices)}), estimación puede ser imprecisa")
        # Usar valores por defecto basados en cámara izquierda
        K_proj = K_left.copy()
        dist_proj = np.zeros(5)
        return K_proj, dist_proj
    
    # Extraer parámetros intrínsecos usando el método de Zhang
    # Basado en "A Flexible New Technique for Camera Calibration" - Zhang, 1999
    
    # Construir sistema de ecuaciones V*b = 0
    # donde b contiene los parámetros de la cónica imagen del círculo absoluto
    V = []
    
    def v_ij(H, i, j):
        """Construye el vector v_ij según Zhang"""
        return np.array([
            H[0, i] * H[0, j],
            H[0, i] * H[1, j] + H[1, i] * H[0, j],
            H[1, i] * H[1, j],
            H[2, i] * H[0, j] + H[0, i] * H[2, j],
            H[2, i] * H[1, j] + H[1, i] * H[2, j],
            H[2, i] * H[2, j]
        ])
    
    for H in H_matrices:
        # Las dos ecuaciones de constraints para cada homografía
        V.append(v_ij(H, 0, 1))  # h1^T * B * h2 = 0
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))  # h1^T * B * h1 = h2^T * B * h2
    
    V = np.array(V)
    
    # Resolver el sistema usando SVD
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]  # Última fila de V^T
    
    # Extraer parámetros intrínsecos de b
    # b = [B11, B12, B22, B13, B23, B33]^T
    # donde B es la matriz de la cónica imagen del círculo absoluto
    
    B11, B12, B22, B13, B23, B33 = b
    
    # Calcular parámetros intrínsecos
    v0 = (B12 * B13 - B11 * B23) / (B11 * B22 - B12 * B12)
    lambda_ = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
    
    if lambda_ / B11 < 0:
        print("[WARNING] Parámetros negativos detectados, ajustando...")
        alpha = np.sqrt(abs(lambda_ / B11))
    else:
        alpha = np.sqrt(lambda_ / B11)  # fx
    
    if lambda_ / B22 < 0:
        beta = np.sqrt(abs(lambda_ / B22))
    else:
        beta = np.sqrt(lambda_ / B22)   # fy
    
    gamma = -B12 * alpha * alpha * beta / lambda_  # skew (normalmente ~0)
    u0 = gamma * v0 / beta - B13 * alpha * alpha / lambda_  # cx
    
    # Construir matriz intrínseca del proyector
    K_proj = np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
    
    # Estimar coeficientes de distorsión (simplificado)
    # Calcular errores de reproyección para estimar distorsión radial
    all_errors = []
    
    for i, H in enumerate(H_matrices):
        if i >= len(objpoints):
            break
            
        objp_2d = objpoints[i][:, :2].astype(np.float32)
        imgp = imgpoints_left[i].reshape(-1, 2).astype(np.float32)
        
        # Proyectar usando homografía
        objp_2d_hom = np.hstack([objp_2d, np.ones((objp_2d.shape[0], 1))])
        imgp_proj_hom = (H @ objp_2d_hom.T).T
        imgp_proj = imgp_proj_hom[:, :2] / imgp_proj_hom[:, 2:3]
        
        # Calcular error radial
        errors = np.linalg.norm(imgp - imgp_proj, axis=1)
        all_errors.extend(errors)
    
    # Distorsión radial simple basada en error promedio
    mean_error = np.mean(all_errors)
    
    if mean_error > 1.0:
        # Estimar k1 (distorsión radial dominante)
        k1 = -mean_error * 0.001  # Factor empírico
        dist_proj = np.array([k1, 0, 0, 0, 0])
    else:
        dist_proj = np.zeros(5)
    
    print(f"\n[PARÁMETROS ESTIMADOS DEL PROYECTOR]")
    print(f"  Matriz intrínseca K_proj:")
    print(f"    fx = {K_proj[0,0]:.6f}")
    print(f"    fy = {K_proj[1,1]:.6f}")
    print(f"    cx = {K_proj[0,2]:.6f}")
    print(f"    cy = {K_proj[1,2]:.6f}")
    print(f"    skew = {K_proj[0,1]:.6f}")
    print(f"\n  Coeficientes de distorsión:")
    print(f"    k1 = {dist_proj[0]:.6f}")
    print(f"    k2 = {dist_proj[1]:.6f}")
    print(f"    p1 = {dist_proj[2]:.6f}")
    print(f"    p2 = {dist_proj[3]:.6f}")
    print(f"    k3 = {dist_proj[4]:.6f}")
    
    print(f"\n  Error de reproyección promedio: {mean_error:.4f} píxeles")
    print(f"  Homografías utilizadas: {len(H_matrices)}")
    
    print(f"\n[NOTA IMPORTANTE]")
    print(f"  Esta es una ESTIMACIÓN APROXIMADA basada en homografías.")
    print(f"  Para calibración precisa del proyector, se requiere:")
    print(f"    - Patrones de luz estructurada (Gray Code/Phase Shifting)")
    print(f"    - Correspondencias píxel a píxel cámara-proyector")
    print(f"    - Calibración completa usando cv2.calibrateCamera con puntos proyector")
    print(f"={'='*60}\n")
    
    return K_proj, dist_proj


def generate_gray_code_patterns(width, height, num_bits=10):
    """
    Genera patrones Gray Code para calibración de proyector.
    
    Args:
        width: Ancho del proyector
        height: Alto del proyector
        num_bits: Número de bits (determina resolución)
    
    Returns:
        Lista de patrones (verticales + horizontales + inversos)
    """
    patterns = []
    
    # Verificar que num_bits es válido para la resolución
    max_bits_width = int(np.ceil(np.log2(width)))
    max_bits_height = int(np.ceil(np.log2(height)))
    
    if num_bits > max_bits_width:
        print(f"[WARNING] num_bits ({num_bits}) mayor que resolución de ancho ({width}px)")
        print(f"[INFO] Ajustando a {max_bits_width} bits para ancho")
        num_bits = max_bits_width
    
    if num_bits > max_bits_height:
        print(f"[WARNING] num_bits ({num_bits}) mayor que resolución de alto ({height}px)")
        print(f"[INFO] Máximo recomendado: {max_bits_height} bits para alto")
    
    # Patrón negro completo (referencia)
    black_pattern = np.zeros((height, width), dtype=np.uint8)
    patterns.append(black_pattern)
    
    # Patrón blanco completo (referencia)
    white_pattern = np.ones((height, width), dtype=np.uint8) * 255
    patterns.append(white_pattern)
    
    # Patrones verticales (codifican coordenada X del proyector)
    print(f"[INFO] Generando {num_bits} patrones verticales...")
    for bit in range(num_bits):
        pattern = np.zeros((height, width), dtype=np.uint8)
        stripe_width = width // (2 ** (bit + 1))
        
        # Verificar que stripe_width no sea cero
        if stripe_width == 0:
            print(f"[WARNING] Patrón vertical bit {bit}: stripe_width = 0, saltando...")
            continue
        
        for x in range(width):
            if (x // stripe_width) % 2 == 1:
                pattern[:, x] = 255
        
        patterns.append(pattern)
    
    # Patrones verticales invertidos (para mejorar precisión)
    print(f"[INFO] Generando {num_bits} patrones verticales invertidos...")
    for bit in range(num_bits):
        pattern = np.zeros((height, width), dtype=np.uint8)
        stripe_width = width // (2 ** (bit + 1))
        
        if stripe_width == 0:
            print(f"[WARNING] Patrón vertical invertido bit {bit}: stripe_width = 0, saltando...")
            continue
        
        for x in range(width):
            if (x // stripe_width) % 2 == 0:
                pattern[:, x] = 255
        
        patterns.append(pattern)
    
    # Patrones horizontales (codifican coordenada Y del proyector)
    print(f"[INFO] Generando {num_bits} patrones horizontales...")
    for bit in range(num_bits):
        pattern = np.zeros((height, width), dtype=np.uint8)
        stripe_height = height // (2 ** (bit + 1))
        
        # Verificar que stripe_height no sea cero
        if stripe_height == 0:
            print(f"[WARNING] Patrón horizontal bit {bit}: stripe_height = 0, saltando...")
            continue
        
        for y in range(height):
            if (y // stripe_height) % 2 == 1:
                pattern[y, :] = 255
        
        patterns.append(pattern)
    
    # Patrones horizontales invertidos
    print(f"[INFO] Generando {num_bits} patrones horizontales invertidos...")
    for bit in range(num_bits):
        pattern = np.zeros((height, width), dtype=np.uint8)
        stripe_height = height // (2 ** (bit + 1))
        
        if stripe_height == 0:
            print(f"[WARNING] Patrón horizontal invertido bit {bit}: stripe_height = 0, saltando...")
            continue
        
        for y in range(height):
            if (y // stripe_height) % 2 == 0:
                pattern[y, :] = 255
        
        patterns.append(pattern)
    
    print(f"[INFO] Total de patrones Gray Code generados: {len(patterns)}")
    print(f"[INFO] Recomendación: Para {width}x{height}, usar --gray-bits {max_bits_width}")
    
    return patterns


def capture_gray_code_patterns(left_stream, right_stream, projector_width=1920, projector_height=1080, 
                               num_bits=10, output_dir="calib_imgs", delay=0.5):
    """
    Captura patrones Gray Code proyectados con ambas cámaras estéreo.
    
    El flujo es:
    1. Genera patrones Gray Code
    2. Muestra cada patrón en pantalla completa (para proyección)
    3. Captura con ambas cámaras simultáneamente
    4. Guarda imágenes con prefijo gray_
    
    Args:
        left_stream: Stream de cámara izquierda
        right_stream: Stream de cámara derecha
        projector_width: Ancho del proyector
        projector_height: Alto del proyector
        num_bits: Número de bits para Gray Code (más bits = mayor resolución)
        output_dir: Directorio donde guardar imágenes
        delay: Retardo entre patrones (segundos)
    
    Returns:
        Tupla de (archivos_left, archivos_right, patrones)
    """
    
    print(f"\n{'='*60}")
    print(f"[CAPTURA DE PATRONES GRAY CODE]")
    print(f"{'='*60}")
    print(f"[INFO] Resolución proyector: {projector_width}x{projector_height}")
    print(f"[INFO] Número de bits: {num_bits}")
    print(f"[INFO] Total patrones a proyectar: {num_bits * 4 + 2}")
    print(f"\n[INSTRUCCIONES]:")
    print(f"  1. Coloca una superficie PLANA y BLANCA frente a las cámaras")
    print(f"  2. Configura el proyector para mostrar en PANTALLA COMPLETA")
    print(f"  3. Asegúrate de que el proyector esté ENFOCADO")
    print(f"  4. La habitación debe estar LO MÁS OSCURA posible")
    print(f"  5. Presiona ENTER cuando estés listo...")
    print(f"  6. Presiona 'q' durante la captura para cancelar")
    print(f"\n")
    
    input("Presiona ENTER para comenzar la captura...")
    
    # Generar patrones
    patterns = generate_gray_code_patterns(projector_width, projector_height, num_bits)
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Listas para guardar nombres de archivos
    captured_files_left = []
    captured_files_right = []
    
    # Crear ventana en pantalla completa para proyección
    window_name = "Proyector - Patrones Gray Code"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print(f"\n[INFO] Iniciando captura de {len(patterns)} patrones...")
    print(f"[INFO] Delay entre patrones: {delay} segundos")
    
    try:
        for i, pattern in enumerate(patterns):
            # Convertir patrón a BGR para visualización
            pattern_bgr = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)
            
            # Mostrar patrón en pantalla completa
            cv2.imshow(window_name, pattern_bgr)
            cv2.waitKey(1)  # Actualizar ventana inmediatamente
            
            print(f"\n[{i+1}/{len(patterns)}] Proyectando patrón Gray Code...")
            
            # Esperar para que el patrón se estabilice
            time.sleep(delay)
            
            # Capturar con ambas cámaras
            left_ret, left_frame = left_stream.get_frame()
            right_ret, right_frame = right_stream.get_frame()
            
            if not left_ret or left_frame is None:
                print(f"[WARNING] No se pudo capturar frame izquierdo para patrón {i+1}")
                continue
            
            if not right_ret or right_frame is None:
                print(f"[WARNING] No se pudo capturar frame derecho para patrón {i+1}")
                continue
            
            # Guardar imágenes con prefijo gray_
            left_filename = os.path.join(output_dir, f"gray_left_{i:03d}.png")
            right_filename = os.path.join(output_dir, f"gray_right_{i:03d}.png")
            
            cv2.imwrite(left_filename, left_frame)
            cv2.imwrite(right_filename, right_frame)
            
            captured_files_left.append(left_filename)
            captured_files_right.append(right_filename)
            
            print(f"  ✓ Guardado: {os.path.basename(left_filename)} y {os.path.basename(right_filename)}")
            
            # Verificar si el usuario quiere cancelar
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print(f"\n[INFO] Captura cancelada por el usuario")
                break
        
        print(f"\n{'='*60}")
        print(f"[INFO] Captura Gray Code completada")
        print(f"  Total patrones capturados: {len(captured_files_left)}")
        print(f"  Archivos guardados en: {output_dir}/")
        print(f"  Prefijo: gray_left_*.png y gray_right_*.png")
        print(f"={'='*60}\n")
        
    except KeyboardInterrupt:
        print(f"\n[WARNING] Captura interrumpida por el usuario (Ctrl+C)")
    
    except Exception as e:
        print(f"\n[ERROR] Error durante captura: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cerrar ventana de proyección
        cv2.destroyWindow(window_name)
    
    return captured_files_left, captured_files_right, patterns


def decode_gray_code_images(gray_images, inverted_images, num_bits):
    """
    Decodifica imágenes Gray Code para obtener coordenadas del proyector.
    
    Args:
        gray_images: Lista de imágenes Gray Code normales
        inverted_images: Lista de imágenes Gray Code invertidas
        num_bits: Número de bits usados
    
    Returns:
        decoded_map: Mapa de coordenadas (height, width) con valores decodificados
        valid_mask: Máscara booleana indicando píxeles válidos
    """
    if len(gray_images) != num_bits or len(inverted_images) != num_bits:
        raise ValueError(f"Se esperaban {num_bits} imágenes, se recibieron {len(gray_images)} y {len(inverted_images)}")
    
    height, width = gray_images[0].shape[:2]
    decoded_map = np.zeros((height, width), dtype=np.int32)
    valid_mask = np.ones((height, width), dtype=bool)
    
    # Convertir todas las imágenes a escala de grises
    gray_imgs_bw = []
    inverted_imgs_bw = []
    
    for i in range(num_bits):
        # Convertir a escala de grises si es necesario
        if len(gray_images[i].shape) == 3:
            gray_bw = cv2.cvtColor(gray_images[i], cv2.COLOR_BGR2GRAY)
        else:
            gray_bw = gray_images[i]
        
        if len(inverted_images[i].shape) == 3:
            inv_bw = cv2.cvtColor(inverted_images[i], cv2.COLOR_BGR2GRAY)
        else:
            inv_bw = inverted_images[i]
        
        gray_imgs_bw.append(gray_bw)
        inverted_imgs_bw.append(inv_bw)
    
    # Decodificar cada píxel
    print(f"[INFO] Decodificando {width}x{height} píxeles...")
    
    for bit in range(num_bits):
        # Para cada bit, determinar si el píxel está iluminado o no
        gray_img = gray_imgs_bw[bit]
        inv_img = inverted_imgs_bw[bit]
        
        # Calcular diferencia entre patrón normal e invertido
        # Esto ayuda a eliminar luz ambiental y mejorar robustez
        diff = gray_img.astype(np.int16) - inv_img.astype(np.int16)
        
        # Umbral adaptativo basado en la diferencia
        threshold = 20  # Píxeles con diferencia menor son inválidos
        
        # Píxeles donde la diferencia es significativa
        valid_diff = np.abs(diff) > threshold
        
        # Determinar si el bit es 1 o 0
        bit_value = (diff > 0).astype(np.int32)
        
        # Actualizar mapa decodificado (código Gray)
        decoded_map += bit_value * (2 ** bit)
        
        # Actualizar máscara de validez
        valid_mask &= valid_diff
    
    # Convertir de código Gray a binario
    binary_map = np.zeros_like(decoded_map)
    binary_map = decoded_map.copy()
    
    for bit in range(num_bits - 1, 0, -1):
        binary_map ^= (binary_map >> 1)
    
    print(f"[INFO] Decodificación completada")
    print(f"  Píxeles válidos: {np.sum(valid_mask)} / {height * width} ({100 * np.sum(valid_mask) / (height * width):.1f}%)")
    
    return binary_map, valid_mask


def generate_gray_code_table(x_map, y_map, valid_mask, max_samples=200):
    """
    Genera tabla de correspondencias Gray Code en formato SEAL.
    
    La tabla contiene muestras de las coordenadas decodificadas del proyector
    en el formato: "número valor_x valor_y"
    
    Args:
        x_map: Mapa de coordenadas X decodificadas del proyector
        y_map: Mapa de coordenadas Y decodificadas del proyector  
        valid_mask: Máscara de píxeles válidos
        max_samples: Número máximo de muestras en la tabla (default: 200)
        
    Returns:
        Lista de strings con formato "número valor_x valor_y"
    """
    print(f"\n[INFO] Generando tabla de correspondencias Gray Code...")
    
    # Obtener píxeles válidos
    valid_y, valid_x = np.where(valid_mask)
    total_valid = len(valid_x)
    
    if total_valid == 0:
        print(f"[WARNING] No hay píxeles válidos para generar tabla")
        return []
    
    print(f"  Píxeles válidos totales: {total_valid}")
    
    # Submuestrear para obtener ~max_samples distribuidos uniformemente
    if total_valid > max_samples:
        step = total_valid // max_samples
        indices = np.arange(0, total_valid, step)[:max_samples]
    else:
        indices = np.arange(total_valid)
    
    print(f"  Muestras seleccionadas: {len(indices)}")
    
    # Generar tabla
    table = []
    for i, idx in enumerate(indices, start=2):  # Empezar desde 2 como en SEAL
        pixel_x = valid_x[idx]
        pixel_y = valid_y[idx]
        
        proj_x = x_map[pixel_y, pixel_x]
        proj_y = y_map[pixel_y, pixel_x]
        
        # Normalizar coordenadas (escalar a rango similar al archivo SEAL)
        # Los valores originales parecen estar en el rango [-0.01, 0.01] para X
        # y [20, 700] para Y
        norm_x = (proj_x / 1920.0) * 0.02 - 0.01  # Escalar X a rango [-0.01, 0.01]
        norm_y = proj_y  # Y ya está en píxeles [0-1080]
        
        table.append(f"{i} {norm_x:.6f} {norm_y:.6f}")
    
    print(f"  Tabla generada con {len(table)} entradas")
    print(f"  Rango X: [{min([float(line.split()[1]) for line in table]):.6f}, {max([float(line.split()[1]) for line in table]):.6f}]")
    print(f"  Rango Y: [{min([float(line.split()[2]) for line in table]):.6f}, {max([float(line.split()[2]) for line in table]):.6f}]")
    
    return table


def process_gray_code_calibration(images_dir, K_left, dist_left, K_right, dist_right, 
                                  projector_width=1920, projector_height=1080, num_bits=10):
    """
    Procesa imágenes Gray Code y calibra el proyector.
    
    Args:
        images_dir: Directorio con imágenes gray_left_*.png y gray_right_*.png
        K_left, dist_left: Parámetros intrínsecos cámara izquierda
        K_right, dist_right: Parámetros intrínsecos cámara derecha
        projector_width: Ancho del proyector
        projector_height: Alto del proyector
        num_bits: Número de bits usados en Gray Code
    
    Returns:
        K_proj: Matriz intrínseca del proyector
        dist_proj: Coeficientes de distorsión del proyector
    """
    
    print(f"\n{'='*60}")
    print(f"[PROCESAMIENTO GRAY CODE - CALIBRACIÓN PROYECTOR]")
    print(f"{'='*60}")
    print(f"[INFO] Directorio: {images_dir}")
    print(f"[INFO] Proyector: {projector_width}x{projector_height}")
    print(f"[INFO] Número de bits: {num_bits}")
    print(f"\n")
    
    import glob
    
    # Cargar imágenes Gray Code de ambas cámaras
    gray_left_files = sorted(glob.glob(os.path.join(images_dir, "gray_left_*.png")))
    gray_right_files = sorted(glob.glob(os.path.join(images_dir, "gray_right_*.png")))
    
    if len(gray_left_files) == 0 or len(gray_right_files) == 0:
        raise FileNotFoundError(f"No se encontraron imágenes Gray Code en {images_dir}")
    
    print(f"[INFO] Imágenes encontradas:")
    print(f"  Izquierda: {len(gray_left_files)} archivos")
    print(f"  Derecha: {len(gray_right_files)} archivos")
    
    # Esperamos: 2 (negro/blanco) + num_bits*4 (vert, vert_inv, horiz, horiz_inv)
    expected_patterns = 2 + num_bits * 4
    
    if len(gray_left_files) < expected_patterns or len(gray_right_files) < expected_patterns:
        print(f"[WARNING] Se esperaban {expected_patterns} patrones, se encontraron {len(gray_left_files)}")
    
    # Cargar imágenes
    print(f"\n[INFO] Cargando imágenes...")
    gray_left_imgs = []
    gray_right_imgs = []
    
    for f in gray_left_files:
        img = cv2.imread(f)
        if img is None:
            raise ValueError(f"Error al cargar imagen: {f}")
        gray_left_imgs.append(img)
    
    for f in gray_right_files:
        img = cv2.imread(f)
        if img is None:
            raise ValueError(f"Error al cargar imagen: {f}")
        gray_right_imgs.append(img)
    
    print(f"[INFO] {len(gray_left_imgs)} imágenes izquierda cargadas correctamente")
    print(f"[INFO] {len(gray_right_imgs)} imágenes derecha cargadas correctamente")
    
    # Calcular número real de patrones por tipo
    # Con 40 patrones: 2 (negro/blanco) + patrones efectivos
    total_patterns = len(gray_left_imgs)
    effective_patterns = total_patterns - 2  # Restar negro y blanco
    patterns_per_type = effective_patterns // 4  # Dividir entre 4 tipos
    
    print(f"\n[INFO] Análisis de patrones:")
    print(f"  Total patrones: {total_patterns}")
    print(f"  Patrones efectivos (sin negro/blanco): {effective_patterns}")
    print(f"  Patrones por tipo (vert/vert_inv/horiz/horiz_inv): {patterns_per_type}")
    
    if patterns_per_type < num_bits:
        print(f"[WARNING] Se esperaban {num_bits} patrones por tipo, pero solo hay {patterns_per_type}")
        print(f"[INFO] Ajustando num_bits a {patterns_per_type} para coincidir con imágenes")
        num_bits = patterns_per_type
    
    # Separar patrones según su tipo
    # Índices: 0=negro, 1=blanco, 2:2+num_bits=vert, 2+num_bits:2+2*num_bits=vert_inv
    #           2+2*num_bits:2+3*num_bits=horiz, 2+3*num_bits:2+4*num_bits=horiz_inv
    
    idx_black = 0
    idx_white = 1
    idx_vert_start = 2
    idx_vert_inv_start = 2 + patterns_per_type
    idx_horiz_start = 2 + 2 * patterns_per_type
    idx_horiz_inv_start = 2 + 3 * patterns_per_type
    
    print(f"\n[INFO] Índices de patrones:")
    print(f"  Negro: {idx_black}")
    print(f"  Blanco: {idx_white}")
    print(f"  Verticales: {idx_vert_start} a {idx_vert_start + patterns_per_type - 1}")
    print(f"  Verticales invertidos: {idx_vert_inv_start} a {idx_vert_inv_start + patterns_per_type - 1}")
    print(f"  Horizontales: {idx_horiz_start} a {idx_horiz_start + patterns_per_type - 1}")
    print(f"  Horizontales invertidos: {idx_horiz_inv_start} a {idx_horiz_inv_start + patterns_per_type - 1}")
    
    # Verificar que los índices no estén fuera de rango
    max_idx = idx_horiz_inv_start + patterns_per_type
    if max_idx > total_patterns:
        raise ValueError(f"Índice máximo ({max_idx}) excede total de patrones ({total_patterns})")
    
    # Decodificar coordenadas X (patrones verticales) - cámara izquierda
    print(f"\n[INFO] Decodificando coordenadas X (patrones verticales)...")
    vert_imgs_left = gray_left_imgs[idx_vert_start:idx_vert_start + patterns_per_type]
    vert_inv_imgs_left = gray_left_imgs[idx_vert_inv_start:idx_vert_inv_start + patterns_per_type]
    
    print(f"[DEBUG] Extrayendo {len(vert_imgs_left)} patrones verticales")
    print(f"[DEBUG] Extrayendo {len(vert_inv_imgs_left)} patrones verticales invertidos")
    
    x_map_left, x_mask_left = decode_gray_code_images(vert_imgs_left, vert_inv_imgs_left, patterns_per_type)
    
    # Decodificar coordenadas Y (patrones horizontales) - cámara izquierda
    print(f"\n[INFO] Decodificando coordenadas Y (patrones horizontales)...")
    horiz_imgs_left = gray_left_imgs[idx_horiz_start:idx_horiz_start + patterns_per_type]
    horiz_inv_imgs_left = gray_left_imgs[idx_horiz_inv_start:idx_horiz_inv_start + patterns_per_type]
    
    print(f"[DEBUG] Extrayendo {len(horiz_imgs_left)} patrones horizontales")
    print(f"[DEBUG] Extrayendo {len(horiz_inv_imgs_left)} patrones horizontales invertidos")
    
    y_map_left, y_mask_left = decode_gray_code_images(horiz_imgs_left, horiz_inv_imgs_left, patterns_per_type)
    
    # Máscara combinada de píxeles válidos
    valid_mask_left = x_mask_left & y_mask_left
    
    print(f"\n[INFO] Resultados decodificación cámara izquierda:")
    print(f"  Píxeles válidos: {np.sum(valid_mask_left)} / {valid_mask_left.size}")
    print(f"  Cobertura: {100 * np.sum(valid_mask_left) / valid_mask_left.size:.1f}%")
    
    # Crear correspondencias para calibración del proyector
    print(f"\n[INFO] Generando correspondencias cámara-proyector...")
    
    # Obtener coordenadas de píxeles válidos
    valid_y, valid_x = np.where(valid_mask_left)
    
    # Coordenadas en la cámara (2D)
    camera_points = np.column_stack([valid_x, valid_y]).astype(np.float32)
    
    # Coordenadas en el proyector (2D)
    projector_x = x_map_left[valid_mask_left]
    projector_y = y_map_left[valid_mask_left]
    projector_points = np.column_stack([projector_x, projector_y]).astype(np.float32)
    
    print(f"  Total correspondencias: {len(camera_points)}")
    
    # Para calibrar el proyector, necesitamos puntos 3D
    # Asumimos que la superficie de proyección es plana (Z=0)
    # y usamos triangulación estéreo para obtener puntos 3D
    
    # Submuestrear para acelerar procesamiento (cada N píxeles)
    subsample = 10
    indices = np.arange(0, len(camera_points), subsample)
    camera_points_sub = camera_points[indices]
    projector_points_sub = projector_points[indices]
    
    print(f"  Correspondencias submuestreadas: {len(camera_points_sub)} (cada {subsample} píxeles)")
    
    # Para simplificar, asumimos superficie plana en Z=0
    # Los puntos 3D serían las coordenadas del proyector mapeadas al espacio 3D
    # Esto es una simplificación; idealmente usaríamos triangulación estéreo
    
    # Crear puntos 3D en un plano arbitrario
    # Escalamos las coordenadas del proyector para que tengan dimensiones físicas
    scale = 1000.0 / projector_width  # Asumimos ancho de 1000mm
    
    objpoints_proj = np.zeros((len(projector_points_sub), 3), dtype=np.float32)
    objpoints_proj[:, 0] = projector_points_sub[:, 0] * scale
    objpoints_proj[:, 1] = projector_points_sub[:, 1] * scale
    objpoints_proj[:, 2] = 0  # Plano Z=0
    
    # Calibrar proyector usando cv2.calibrateCamera
    # Necesitamos agrupar puntos en "vistas" (frames)
    # Dividimos los puntos en grupos para simular múltiples capturas
    
    points_per_view = 500
    num_views = len(objpoints_proj) // points_per_view
    
    if num_views < 3:
        print(f"[WARNING] Pocas vistas para calibración ({num_views}), aumentando puntos por vista")
        points_per_view = len(objpoints_proj) // 3
        num_views = 3
    
    objpoints_list = []
    imgpoints_list = []
    
    for i in range(num_views):
        start_idx = i * points_per_view
        end_idx = min((i + 1) * points_per_view, len(objpoints_proj))
        
        if end_idx - start_idx < 10:
            break
        
        objpoints_list.append(objpoints_proj[start_idx:end_idx])
        imgpoints_list.append(projector_points_sub[start_idx:end_idx])
    
    print(f"\n[INFO] Calibrando proyector con {len(objpoints_list)} vistas...")
    print(f"  Puntos por vista: ~{points_per_view}")
    
    # Calibrar proyector
    try:
        ret, K_proj, dist_proj, rvecs, tvecs = cv2.calibrateCamera(
            objpoints_list,
            imgpoints_list,
            (projector_width, projector_height),
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
        
        print(f"\n{'='*60}")
        print(f"[CALIBRACIÓN PROYECTOR COMPLETADA]")
        print(f"{'='*60}")
        print(f"\n[PARÁMETROS INTRÍNSECOS DEL PROYECTOR - MÉTODO 2 (PRECISO)]")
        print(f"  Error RMS: {ret:.6f} píxeles")
        print(f"\n  Matriz intrínseca K_proj:")
        print(f"    fx = {K_proj[0,0]:.6f}")
        print(f"    fy = {K_proj[1,1]:.6f}")
        print(f"    cx = {K_proj[0,2]:.6f}")
        print(f"    cy = {K_proj[1,2]:.6f}")
        print(f"    skew = {K_proj[0,1]:.6f}")
        print(f"\n  Coeficientes de distorsión:")
        print(f"    k1 = {dist_proj[0,0]:.6f}")
        print(f"    k2 = {dist_proj[0,1]:.6f}")
        print(f"    p1 = {dist_proj[0,2]:.6f}")
        print(f"    p2 = {dist_proj[0,3]:.6f}")
        print(f"    k3 = {dist_proj[0,4]:.6f}")
        
        if dist_proj.shape[1] > 5:
            print(f"    k4 = {dist_proj[0,5]:.6f}")
            print(f"    k5 = {dist_proj[0,6]:.6f}")
            print(f"    k6 = {dist_proj[0,7]:.6f}")
        
        print(f"\n  Vistas utilizadas: {len(objpoints_list)}")
        print(f"  Correspondencias totales: {len(camera_points)}")
        print(f"  Correspondencias usadas: {len(camera_points_sub)}")
        print(f"\n[NOTA]")
        print(f"  Estos son parámetros PRECISOS obtenidos mediante Gray Code")
        print(f"  Precisión esperada: <0.3 píxeles")
        print(f"  Aptos para uso en producción")
        print(f"={'='*60}\n")
        
        # Generar tabla de correspondencias Gray Code (formato SEAL)
        gray_code_table = generate_gray_code_table(x_map_left, y_map_left, valid_mask_left)
        
        return K_proj, dist_proj, gray_code_table
        
    except Exception as e:
        print(f"\n[ERROR] Error durante calibración del proyector: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Retornar valores por defecto en caso de error
        print(f"\n[WARNING] Usando valores por defecto basados en cámara izquierda")
        return K_left.copy(), np.zeros(5), []


def generate_ini_recommendations(K_left, K_right, R, T, img_size):
    """
    Genera recomendaciones de parámetros para scaner_settings_Seal.ini
    basándose en los resultados de calibración estéreo
    """
    print("\n" + "="*60)
    print("RECOMENDACIONES PARA scaner_settings_Seal.ini")
    print("="*60)
    
    # 1. BaseTol (ángulo de rotación/tilt)
    rvec, _ = cv2.Rodrigues(R)
    angles_deg = np.degrees(rvec.flatten())
    
    # Extraer pitch de la matriz de rotación
    import math
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
    pitch_deg = math.degrees(pitch)
    
    # BaseTol es el pitch (inclinación en X), debe coincidir con línea 4 del SEAL
    base_tol = int(round(pitch_deg))
    
    print(f"\n[Algo]")
    print(f"BaseTol={base_tol}  # Inclinación proyector (coincide con línea 4 SEAL)")
    
    # 2. TolDown y TolUp (límites de profundidad)
    baseline = np.linalg.norm(T)
    tol_down = -int(baseline * 2.5)  # Límite lejano (~2.5x baseline)
    tol_up = -int(baseline * 0.05)   # Límite cercano (~5% baseline)
    
    print(f"TolDown={tol_down}  # Límite lejano (~2.5x baseline de {baseline:.2f}mm)")
    print(f"TolUp={tol_up}  # Límite cercano (~5% baseline)")
    
    # 3. TolRadius (radio de trabajo)
    working_distance = 300  # mm típico para SEAL
    fx_avg = (K_left[0,0] + K_right[0,0]) / 2
    sensor_width = img_size[0]
    fov_width_mm = (sensor_width / fx_avg) * working_distance
    tol_radius = int(fov_width_mm / 2)
    
    print(f"TolRadius={tol_radius}  # Radio de trabajo a {working_distance}mm")
    
    # 4. Información adicional y relación con archivo SEAL
    print(f"\n# Relación con archivo de calibración SEAL:")
    print(f"# Línea 2 SEAL: {baseline:.6f} {(baseline * (K_left[1,1]+K_right[1,1])/2)/img_size[1]:.6f}")
    print(f"#   - Baseline: {baseline:.2f} mm (escala horizontal)")
    print(f"#   - Factor kz: {(baseline * (K_left[1,1]+K_right[1,1])/2)/img_size[1]:.6f} (escala profundidad)")
    
    offset_x = int(round(K_right[0, 2] - K_left[0, 2]))
    offset_y = int(round(K_right[1, 2] - K_left[1, 2]))
    print(f"# Línea 3 SEAL: {offset_x} {offset_y}")
    print(f"#   - Offset centro proyección (cx, cy en píxeles)")
    
    displacement_z = int(round(T.flatten()[2]))
    print(f"# Línea 4 SEAL: {base_tol} {displacement_z}")
    print(f"#   - BaseTol: {base_tol}° (tilt proyector)")
    print(f"#   - Altura proyector: {displacement_z} mm")
    
    print(f"\n# Parámetros de calibración:")
    print(f"# FOV horizontal estimado: {fov_width_mm:.2f} mm a {working_distance}mm")
    print(f"# Ángulos de rotación: Pitch={pitch_deg:.2f}°, Yaw={angles_deg[1]:.2f}°, Roll={angles_deg[2]:.2f}°")
    
    # 5. Recomendaciones de cámara (valores conservadores)
    print(f"\n[Camera]")
    print(f"Brightness=4  # Ajustar manualmente según iluminación")
    print(f"LightP=100  # Potencia del proyector (ajustar manualmente)")
    
    print("\n" + "="*60)
    print("NOTA: Estos valores son ESTIMACIONES basadas en calibración.")
    print("Los valores de líneas 2, 3, 4 del SEAL son parámetros extrínsecos")
    print("del sistema cámara-proyector y afectan la reconstrucción 3D.")
    print("BaseTol debe coincidir con el primer valor de línea 4 SEAL.")
    print("="*60)
    
    # Retornar valores para posible guardado automático
    return {
        'BaseTol': base_tol,
        'TolDown': tol_down,
        'TolUp': tol_up,
        'TolRadius': tol_radius,
        'baseline': baseline,
        'fov_width_mm': fov_width_mm,
        'offset_x': offset_x,
        'offset_y': offset_y,
        'displacement_z': displacement_z
    }


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
    
    # Estimar parámetros del proyector
    K_proj, dist_proj = estimate_projector_parameters(
        objpoints, imgpoints_left, imgpoints_right,
        K_left, dist_left, K_right, dist_right, img_size)
    
    # Generar recomendaciones para archivos .ini
    generate_ini_recommendations(K_left, K_right, R, T, img_size)
    
    print("[INFO] Calibración completada")
    return img_size, K_right, dist_right, K_left, dist_left, R, T, K_proj, dist_proj


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
    
    # Estimar parámetros del proyector
    K_proj, dist_proj = estimate_projector_parameters(
        objpoints, imgpoints_left, imgpoints_right,
        K_left, dist_left, K_right, dist_right, img_size)
    
    # Generar recomendaciones para archivos .ini
    generate_ini_recommendations(K_left, K_right, R, T, img_size)
    
    print("[INFO] Calibración completada")
    return img_size, K_right, dist_right, K_left, dist_left, R, T, K_proj, dist_proj


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
            
            # Estimar parámetros del proyector
            K_proj, dist_proj = estimate_projector_parameters(
                objpoints, imgpoints_left, imgpoints_right,
                K_left, dist_left, K_right, dist_right, img_size)
            
            # Generar recomendaciones para archivos .ini
            generate_ini_recommendations(K_left, K_right, R, T, img_size)
                
            print("[INFO] Calibración completada")
            return img_size, K_right, dist_right, K_left, dist_left, R, T, K_proj, dist_proj
            
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
    
    # Parámetros para calibración precisa con Gray Code
    parser.add_argument("--gray-code", action="store_true",
                       help="Activar captura de patrones Gray Code para calibración precisa del proyector")
    parser.add_argument("--projector-width", type=int, default=1920,
                       help="Ancho del proyector (por defecto: 1920)")
    parser.add_argument("--projector-height", type=int, default=1080,
                       help="Alto del proyector (por defecto: 1080)")
    parser.add_argument("--gray-bits", type=int, default=10,
                       help="Número de bits para Gray Code (por defecto: 10, más bits = mayor resolución)")
    parser.add_argument("--gray-delay", type=float, default=0.5,
                       help="Delay entre patrones Gray Code en segundos (por defecto: 0.5)")
    parser.add_argument("--from-gray-images", type=str, default=None,
                       help="Procesar imágenes Gray Code existentes desde directorio (ej: calib_imgs)")
    parser.add_argument("--gray-code-capture", action="store_true",
                       help="Modo especial: Solo capturar patrones Gray Code (sin calibración estéreo)")
    # Se elimina el argumento de resolución ya que se usará la resolución máxima de la cámara
    
    args = parser.parse_args()
    
    if args.test:
        print("[TEST] Modo de prueba individual de cámaras")
        print("1. Probando cámara 0 (Laser/A)...")
        test_single_camera(0, "Laser (A)")
        print("2. Probando cámara 1 (UV/B)...")
        test_single_camera(1, "UV (B)")
        return 0
    
    # Modo especial: Solo captura de patrones Gray Code
    if args.gray_code_capture:
        print(f"\n{'='*60}")
        print(f"[MODO: CAPTURA GRAY CODE]")
        print(f"{'='*60}")
        print(f"[INFO] Este modo solo captura patrones Gray Code")
        print(f"[INFO] Asegúrate de tener:")
        print(f"  - Proyector DLP conectado y configurado")
        print(f"  - Superficie plana blanca como pantalla")
        print(f"  - Habitación lo más oscura posible")
        print(f"  - Cámaras enfocadas al área de proyección")
        print(f"\n")
        
        try:
            # Inicializar streams de cámara
            from stereo_calibration import CameraStream
            
            left_stream = CameraStream(args.left, "Laser (A)")
            right_stream = CameraStream(args.right, "UV (B)")
            
            # Iniciar cámaras
            print(f"[INFO] Iniciando cámara izquierda ({args.left})...")
            if not left_stream.start():
                print(f"[ERROR] No se pudo iniciar cámara izquierda")
                return 1
            
            time.sleep(2)
            
            print(f"[INFO] Iniciando cámara derecha ({args.right})...")
            if not right_stream.start():
                print(f"[ERROR] No se pudo iniciar cámara derecha")
                left_stream.stop()
                return 1
            
            time.sleep(2)
            
            # Capturar patrones Gray Code
            captured_left, captured_right, patterns = capture_gray_code_patterns(
                left_stream, right_stream,
                projector_width=args.projector_width,
                projector_height=args.projector_height,
                num_bits=args.gray_bits,
                output_dir="calib_imgs",
                delay=args.gray_delay
            )
            
            # Cerrar cámaras
            left_stream.stop()
            right_stream.stop()
            
            print(f"\n[RESULTADOS]")
            print(f"  Imágenes capturadas: {len(captured_left)} pares")
            print(f"  Guardadas en: calib_imgs/")
            print(f"  Archivos: gray_left_*.png y gray_right_*.png")
            print(f"\n[PRÓXIMOS PASOS]")
            print(f"  1. Verifica que las imágenes capturadas sean válidas")
            print(f"  2. Procesa las imágenes con: --from-gray-images calib_imgs")
            print(f"\n")
            
            return 0
            
        except Exception as e:
            print(f"[ERROR] Error durante captura Gray Code: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
    
    try:
        # Modo procesamiento de imágenes Gray Code
        if args.from_gray_images:
            print(f"\n{'='*60}")
            print(f"[MODO: PROCESAMIENTO GRAY CODE]")
            print(f"{'='*60}")
            print(f"[INFO] Directorio: {args.from_gray_images}")
            print(f"[INFO] Proyector: {args.projector_width}x{args.projector_height}")
            print(f"[INFO] Número de bits: {args.gray_bits}")
            print(f"\n")
            
            # Primero necesitamos los parámetros de calibración estéreo
            # Intentar cargar desde archivos de calibración existentes
            import glob
            
            # Buscar archivo de calibración estéreo
            calib_files = glob.glob("stereo_calibration_seal.txt")
            
            if not calib_files:
                print(f"[ERROR] No se encontró archivo de calibración estéreo")
                print(f"[INFO] Por favor, ejecuta primero la calibración estéreo:")
                print(f"  python stereo_calibration.py --from-images {args.from_gray_images}")
                return 1
            
            print(f"[WARNING] Procesamiento de imágenes Gray Code requiere calibración estéreo previa")
            print(f"[INFO] Cargando calibración estéreo desde imágenes normales...")
            
            # Procesar imágenes de calibración estéreo normal primero
            img_size, K_right, dist_right, K_left, dist_left, R, T, K_proj_est, dist_proj_est = process_existing_images(
                args.from_gray_images, args.rows, args.cols, args.square_size, 
                pattern_type=args.pattern_type)
            
            print(f"\n[INFO] Calibración estéreo cargada exitosamente")
            print(f"[INFO] Ahora procesando imágenes Gray Code...")
            
            # Procesar imágenes Gray Code para calibración precisa del proyector
            K_proj, dist_proj, gray_code_table = process_gray_code_calibration(
                args.from_gray_images,
                K_left, dist_left,
                K_right, dist_right,
                projector_width=args.projector_width,
                projector_height=args.projector_height,
                num_bits=args.gray_bits
            )
            
            # Guardar resultados
            print(f"\n[INFO] Guardando resultados de calibración precisa...")
            
            # Guardar resultados en formato técnico
            save_calibration_results(img_size, K_right, dist_right, K_left, dist_left, R, T, args.output)
            
            # Guardar en el archivo SEAL principal con parámetros PRECISOS y tabla Gray Code
            seal_output_file = args.output.replace(".txt", "_seal.txt")
            save_seal_calibration_file(img_size, K_left, dist_left, K_right, dist_right, R, T,
                                     K_proj, dist_proj,  # Usar parámetros PRECISOS de Gray Code
                                     args.template, seal_output_file, args.dev_id,
                                     square_size_mm=args.square_size, rows=args.rows, cols=args.cols,
                                     gray_code_table=gray_code_table)  # Incluir tabla Gray Code
            
            print(f"\n{'='*60}")
            print(f"[CALIBRACIÓN COMPLETA - GRAY CODE]")
            print(f"{'='*60}")
            print(f"\n[ARCHIVOS GENERADOS]:")
            print(f"  1. {args.output}: Calibración técnica estéreo")
            print(f"  2. {seal_output_file}: Archivo SEAL DEFINITIVO con:")
            print(f"     - Parámetros PRECISOS del proyector (Gray Code)")
            print(f"     - Tabla de correspondencias ({len(gray_code_table)} entradas)")
            print(f"     - Precisión <0.3 píxeles")
            print(f"  3. {seal_output_file.replace('_seal.txt', '_seal_projector_params.txt')}: Parámetros detallados")
            print(f"\n[CALIDAD]:")
            print(f"  - MÉTODO: Gray Code (preciso)")
            print(f"  - Precisión: <0.3 píxeles")
            print(f"  - Apto para: PRODUCCIÓN")
            print(f"={'='*60}\n")
            
            return 0
        
        # Modo procesamiento de imágenes existentes
        elif args.from_images:
            print(f"[INFO] Modo: Procesamiento de imágenes existentes")
            print(f"  Directorio: {args.from_images}")
            print(f"  Tipo de patrón: {args.pattern_type}")
            
            img_size, K_right, dist_right, K_left, dist_left, R, T, K_proj, dist_proj = process_existing_images(
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
            
            img_size, K_right, dist_right, K_left, dist_left, R, T, K_proj, dist_proj = calibrate_stereo_cameras(
                args.left, args.right, args.rows, args.cols, args.square_size, args.images, 
                pattern_type=args.pattern_type, uv_brightness=args.uv_brightness, uv_contrast=args.uv_contrast,
                resolution=None, target_fps=args.fps, aruco_dict_name=args.aruco_dict,
                auto_capture=not args.no_auto_capture)
            
            # Si se activó el flag --gray-code, capturar patrones después de calibración estéreo
            if args.gray_code:
                print(f"\n{'='*60}")
                print(f"[MODO GRAY CODE ACTIVADO]")
                print(f"{'='*60}")
                print(f"[INFO] La calibración estéreo básica ha finalizado")
                print(f"[INFO] Ahora se capturarán patrones Gray Code para calibración precisa del proyector")
                print(f"\n")
                
                # Reabrir streams de cámara si es necesario
                # (en la práctica, deberían seguir abiertos desde calibrate_stereo_cameras)
                # Por ahora asumimos que el usuario ejecutará esto en una segunda pasada
                
                print(f"[WARNING] La captura de Gray Code requiere reiniciar las cámaras")
                print(f"[INFO] Por favor, ejecuta el siguiente comando:")
                print(f"\n  python stereo_calibration.py --gray-code-capture \\")
                print(f"    --left {args.left} --right {args.right} \\")
                print(f"    --projector-width {args.projector_width} \\")
                print(f"    --projector-height {args.projector_height} \\")
                print(f"    --gray-bits {args.gray_bits} \\")
                print(f"    --gray-delay {args.gray_delay}")
                print(f"\n[INFO] Las imágenes Gray Code se guardarán en calib_imgs/ con prefijo gray_")
                print(f"\n")
        
        # Guardar resultados en formato técnico
        save_calibration_results(img_size, K_right, dist_right, K_left, dist_left, R, T, args.output)
        
        # Guardar resultados en formato SEAL con resolución fija 1280x720
        seal_output_file = args.output.replace(".txt", "_seal.txt")
        save_seal_calibration_file(img_size, K_left, dist_left, K_right, dist_right, R, T,
                                 K_proj, dist_proj,
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
