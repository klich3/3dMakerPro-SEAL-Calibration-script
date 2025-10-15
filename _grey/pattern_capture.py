#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pattern_capture.py
------------------
Captura y procesamiento de patrones Gray Code/Phase Shifting
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Optional
from gray_code_patterns import generate_gray_code_sequence, generate_phase_shift_patterns


class PatternCapture:
    def __init__(self, camera_index: int = 0):
        """
        Inicializa la captura de patrones
        
        Args:
            camera_index: Índice de la cámara a usar
        """
        self.camera_index = camera_index
        self.cap = None
        self.is_open = False
        
    def open_camera(self) -> bool:
        """
        Abre la cámara para captura
        
        Returns:
            True si se abrió correctamente, False en caso contrario
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.cap.isOpened():
                # Configurar propiedades de la cámara
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.is_open = True
                print(f"[INFO] Cámara {self.camera_index} abierta correctamente")
                return True
            else:
                print(f"[ERROR] No se pudo abrir la cámara {self.camera_index}")
                return False
        except Exception as e:
            print(f"[ERROR] Error al abrir cámara: {str(e)}")
            return False
    
    def capture_image(self) -> Optional[np.ndarray]:
        """
        Captura una imagen de la cámara
        
        Returns:
            Imagen capturada o None si falla
        """
        if not self.is_open or not self.cap:
            print("[ERROR] Cámara no abierta")
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return frame
            else:
                print("[ERROR] No se pudo capturar imagen")
                return None
        except Exception as e:
            print(f"[ERROR] Error al capturar imagen: {str(e)}")
            return None
    
    def close_camera(self):
        """Cierra la cámara"""
        if self.cap:
            self.cap.release()
        self.is_open = False
        print("[INFO] Cámara cerrada")
    
    def capture_pattern_sequence(self, patterns: List[np.ndarray], 
                               output_dir: str, delay: float = 1.0) -> List[str]:
        """
        Captura una secuencia de imágenes correspondientes a patrones
        
        Args:
            patterns: Lista de patrones a proyectar
            output_dir: Directorio para guardar las imágenes
            delay: Retraso entre capturas en segundos
            
        Returns:
            Lista de rutas de archivos capturados
        """
        if not self.is_open:
            print("[ERROR] Cámara no abierta")
            return []
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        captured_files = []
        
        # Mostrar cada patrón y capturar imagen
        for i, pattern in enumerate(patterns):
            print(f"[INFO] Mostrando patrón {i+1}/{len(patterns)}")
            
            # Mostrar patrón (en una aplicación real, esto sería proyectado)
            cv2.imshow("Patrón", pattern)
            cv2.waitKey(1)  # Actualizar ventana
            
            # Esperar un poco para que el patrón se estabilice
            time.sleep(delay)
            
            # Capturar imagen
            frame = self.capture_image()
            if frame is not None:
                # Guardar imagen
                filename = os.path.join(output_dir, f"capture_{i:03d}.png")
                cv2.imwrite(filename, frame)
                captured_files.append(filename)
                print(f"[INFO] Imagen capturada: {filename}")
            else:
                print(f"[ERROR] No se pudo capturar imagen para patrón {i+1}")
            
            # Pequeña pausa entre capturas
            time.sleep(0.1)
        
        cv2.destroyAllWindows()
        return captured_files


class PatternDecoder:
    def __init__(self):
        """Inicializa el decodificador de patrones"""
        pass
    
    def decode_gray_code(self, captured_images: List[np.ndarray], 
                        patterns: List[np.ndarray]) -> np.ndarray:
        """
        Decodifica una secuencia de imágenes usando patrones Gray Code
        
        Args:
            captured_images: Lista de imágenes capturadas
            patterns: Lista de patrones Gray Code usados
            
        Returns:
            Mapa de coordenadas decodificado
        """
        if len(captured_images) != len(patterns):
            raise ValueError("Número de imágenes no coincide con número de patrones")
        
        if len(captured_images) == 0:
            return np.array([])
        
        height, width = captured_images[0].shape[:2]
        decoded_map = np.zeros((height, width), dtype=np.int32)
        
        # Convertir imágenes a escala de grises si es necesario
        gray_images = []
        for img in captured_images:
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img
            gray_images.append(gray_img)
        
        # Decodificar cada píxel
        for y in range(height):
            for x in range(width):
                # Construir el código Gray para este píxel
                gray_code = 0
                for i, (img, pattern) in enumerate(zip(gray_images, patterns)):
                    # Determinar si el píxel es blanco o negro
                    img_value = img[y, x]
                    pattern_value = pattern[y, x]
                    
                    # Usar un umbral para determinar blanco/negro
                    threshold = 128
                    is_white_img = img_value > threshold
                    is_white_pattern = pattern_value > threshold
                    
                    # Si coinciden, este bit contribuye al código
                    if is_white_img == is_white_pattern:
                        gray_code |= (1 << i)
                
                decoded_map[y, x] = gray_code
        
        return decoded_map
    
    def decode_phase_shift(self, captured_images: List[np.ndarray]) -> np.ndarray:
        """
        Decodifica una secuencia de imágenes usando Phase Shifting
        
        Args:
            captured_images: Lista de imágenes capturadas con patrones sinusoidales
            
        Returns:
            Mapa de fase decodificado
        """
        if len(captured_images) < 3:
            raise ValueError("Se necesitan al menos 3 imágenes para Phase Shifting")
        
        # Convertir a escala de grises
        gray_images = []
        for img in captured_images[:4]:  # Usar máximo 4 imágenes
            if len(img.shape) == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = img
            gray_images.append(gray_img.astype(np.float32))
        
        height, width = gray_images[0].shape
        phase_map = np.zeros((height, width), dtype=np.float32)
        
        # Algoritmo de 4-step Phase Shifting
        if len(gray_images) >= 4:
            # Calcular la fase usando las 4 imágenes
            # φ = arctan2(I3 - I1, I0 - I2) / 2
            i0, i1, i2, i3 = gray_images[:4]
            numerator = i3 - i1
            denominator = i0 - i2
            
            # Evitar división por cero
            denominator = np.where(denominator == 0, 1e-10, denominator)
            
            phase_map = np.arctan2(numerator, denominator) / 2.0
            
            # Normalizar a [0, 2π]
            phase_map = np.mod(phase_map + 2 * np.pi, 2 * np.pi)
        
        return phase_map


def main():
    """Función principal de demostración"""
    print("[INFO] Iniciando captura de patrones...")
    
    # Inicializar capturador
    capture = PatternCapture(0)  # Usar cámara 0
    
    try:
        # Abrir cámara
        if not capture.open_camera():
            return
        
        # Generar patrones de ejemplo
        print("[INFO] Generando patrones...")
        gray_patterns = generate_gray_code_sequence(1280, 720, 8)
        phase_patterns = generate_phase_shift_patterns(1280, 720, 4)
        all_patterns = gray_patterns + phase_patterns
        
        print(f"[INFO] Generados {len(all_patterns)} patrones")
        
        # Capturar secuencia
        print("[INFO] Capturando secuencia de patrones...")
        print("[INFO] Presiona Ctrl+C para detener")
        print("[INFO] Asegúrate de que los patrones estén siendo proyectados...")
        
        try:
            captured_files = capture.capture_pattern_sequence(
                all_patterns, 
                "captured_patterns", 
                delay=0.5
            )
            print(f"[INFO] Capturadas {len(captured_files)} imágenes")
        except KeyboardInterrupt:
            print("[INFO] Captura interrumpida por usuario")
        
    except Exception as e:
        print(f"[ERROR] Error durante la captura: {str(e)}")
    
    finally:
        # Cerrar cámara
        capture.close_camera()


if __name__ == "__main__":
    main()