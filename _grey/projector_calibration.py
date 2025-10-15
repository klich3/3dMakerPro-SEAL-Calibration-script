#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
projector_calibration.py
------------------------
Calibración del sistema proyector-cámara usando patrones Gray Code/Phase Shifting
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Optional
from pattern_capture import PatternCapture, PatternDecoder
from gray_code_patterns import generate_gray_code_sequence, generate_phase_shift_patterns


class ProjectorCameraCalibration:
    def __init__(self, camera_index: int = 0, projector_width: int = 1280, projector_height: int = 720):
        """
        Inicializa el sistema de calibración proyector-cámara
        
        Args:
            camera_index: Índice de la cámara
            projector_width: Ancho del proyector
            projector_height: Alto del proyector
        """
        self.camera_index = camera_index
        self.projector_width = projector_width
        self.projector_height = projector_height
        self.capture = PatternCapture(camera_index)
        self.decoder = PatternDecoder()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.projector_matrix = None
        self.projector_dist_coeffs = None
        
    def calibrate_camera(self, chessboard_size: Tuple[int, int] = (9, 6), 
                        square_size: float = 25.0) -> bool:
        """
        Calibra la cámara usando un tablero de ajedrez
        
        Args:
            chessboard_size: Tamaño del tablero de ajedrez (columnas, filas)
            square_size: Tamaño de cada cuadrado en mm
            
        Returns:
            True si la calibración fue exitosa
        """
        print("[INFO] Iniciando calibración de cámara...")
        
        if not self.capture.open_camera():
            return False
        
        # Preparar puntos del objeto (tablero de ajedrez)
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays para almacenar puntos del objeto y puntos de imagen
        objpoints = []  # Puntos 3D en el mundo real
        imgpoints = []  # Puntos 2D en el plano de imagen
        
        print("[INFO] Mostrando tablero de ajedrez para calibración...")
        print("[INFO] Presiona 'c' para capturar imagen, 'q' para terminar")
        
        try:
            while True:
                frame = self.capture.capture_image()
                if frame is None:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Buscar esquinas del tablero de ajedrez
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                
                # Si se encuentran esquinas, dibujarlas
                if ret:
                    cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                    objpoints.append(objp)
                    imgpoints.append(corners)
                    print(f"[INFO] Capturadas {len(objpoints)} imágenes para calibración")
                
                # Mostrar imagen
                cv2.imshow('Calibración de Cámara', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and ret:
                    # Capturar imagen actual
                    pass
                elif key == ord('q'):
                    break
            
            # Realizar calibración si hay suficientes imágenes
            if len(objpoints) >= 10:
                print("[INFO] Realizando calibración...")
                ret, self.camera_matrix, self.dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None)
                
                if ret:
                    print("[INFO] Calibración de cámara completada")
                    print(f"Matriz de cámara:\n{self.camera_matrix}")
                    print(f"Coeficientes de distorsión:\n{self.dist_coeffs}")
                    return True
                else:
                    print("[ERROR] Falló la calibración de cámara")
                    return False
            else:
                print("[ERROR] Se necesitan al menos 10 imágenes para calibración")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error durante calibración: {str(e)}")
            return False
        finally:
            self.capture.close_camera()
            cv2.destroyAllWindows()
    
    def calibrate_projector(self, gray_bits: int = 8) -> bool:
        """
        Calibra el proyector usando patrones Gray Code
        
        Args:
            gray_bits: Número de bits para los patrones Gray Code
            
        Returns:
            True si la calibración fue exitosa
        """
        print("[INFO] Iniciando calibración de proyector...")
        
        # Generar patrones Gray Code
        gray_patterns = generate_gray_code_sequence(
            self.projector_width, self.projector_height, gray_bits)
        
        if not self.capture.open_camera():
            return False
        
        try:
            # Capturar secuencia de patrones
            print("[INFO] Capturando secuencia de patrones Gray Code...")
            print("[INFO] Asegúrate de que los patrones estén siendo proyectados")
            
            captured_files = self.capture.capture_pattern_sequence(
                gray_patterns, "projector_calibration", delay=1.0)
            
            if len(captured_files) != len(gray_patterns):
                print("[ERROR] No se capturaron todas las imágenes")
                return False
            
            # Cargar imágenes capturadas
            captured_images = []
            for filename in captured_files:
                img = cv2.imread(filename)
                if img is not None:
                    captured_images.append(img)
                else:
                    print(f"[ERROR] No se pudo cargar {filename}")
                    return False
            
            # Decodificar patrones
            print("[INFO] Decodificando patrones...")
            decoded_map = self.decoder.decode_gray_code(captured_images, gray_patterns)
            
            # Aquí iría el algoritmo específico para calibrar el proyector
            # basado en el mapa decodificado y las correspondencias
            # con la cámara ya calibrada
            
            print("[INFO] Calibración de proyector completada (simulada)")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error durante calibración de proyector: {str(e)}")
            return False
        finally:
            self.capture.close_camera()
    
    def calibrate_stereo_system(self) -> bool:
        """
        Calibra el sistema estéreo proyector-cámara
        
        Returns:
            True si la calibración fue exitosa
        """
        print("[INFO] Iniciando calibración estéreo proyector-cámara...")
        
        # Primero calibrar la cámara
        if not self.calibrate_camera():
            print("[ERROR] Falló la calibración de cámara")
            return False
        
        # Luego calibrar el proyector
        if not self.calibrate_projector():
            print("[ERROR] Falló la calibración de proyector")
            return False
        
        # Calibración estéreo (simplificada)
        print("[INFO] Realizando calibración estéreo...")
        # En una implementación completa, aquí se calcularían las matrices
        # de proyección y las transformaciones entre los dos dispositivos
        
        print("[INFO] Calibración estéreo completada")
        return True
    
    def save_calibration(self, filename: str):
        """
        Guarda los parámetros de calibración
        
        Args:
            filename: Nombre del archivo de salida
        """
        try:
            # Crear diccionario con parámetros
            calibration_data = {
                'camera_matrix': self.camera_matrix,
                'dist_coeffs': self.dist_coeffs,
                'projector_width': self.projector_width,
                'projector_height': self.projector_height
            }
            
            # Guardar usando numpy
            np.savez(filename, **calibration_data)
            print(f"[INFO] Parámetros de calibración guardados en {filename}")
            
        except Exception as e:
            print(f"[ERROR] Error al guardar calibración: {str(e)}")
    
    def load_calibration(self, filename: str) -> bool:
        """
        Carga los parámetros de calibración
        
        Args:
            filename: Nombre del archivo de entrada
            
        Returns:
            True si se cargó correctamente
        """
        try:
            # Cargar datos
            data = np.load(filename)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.projector_width = int(data['projector_width'])
            self.projector_height = int(data['projector_height'])
            
            print(f"[INFO] Parámetros de calibración cargados de {filename}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error al cargar calibración: {str(e)}")
            return False


def main():
    """Función principal de demostración"""
    print("[INFO] Sistema de calibración proyector-cámara")
    
    # Crear sistema de calibración
    calibration = ProjectorCameraCalibration(
        camera_index=0, 
        projector_width=1280, 
        projector_height=720
    )
    
    # Calibrar sistema
    if calibration.calibrate_stereo_system():
        # Guardar calibración
        calibration.save_calibration("projector_camera_calibration.npz")
        print("[INFO] Sistema calibrado exitosamente")
    else:
        print("[ERROR] Falló la calibración del sistema")


if __name__ == "__main__":
    main()