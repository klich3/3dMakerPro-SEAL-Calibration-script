#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seal_calibration_processor.py
----------------------------
Procesa archivos de calibración SEAL y convierte entre formatos
"""

import cv2
import numpy as np
import os
import re
from typing import Optional, Tuple, Dict


class SEALCalibrationProcessor:
    def __init__(self, seal_file: str):
        """
        Inicializa el procesador de calibración SEAL
        
        Args:
            seal_file: Ruta al archivo de calibración SEAL (.txt)
        """
        self.seal_file = seal_file
        self.calibration_data = {}
        self.is_loaded = False
        
        # Cargar archivo SEAL
        self.load_seal_file()
    
    def load_seal_file(self) -> bool:
        """
        Carga y parsea un archivo de calibración SEAL
        
        Returns:
            True si se cargó correctamente
        """
        try:
            if not os.path.exists(self.seal_file):
                print(f"[ERROR] Archivo SEAL no encontrado: {self.seal_file}")
                return False
            
            with open(self.seal_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.read().splitlines()
            
            if not lines:
                print("[ERROR] Archivo SEAL vacío")
                return False
            
            # Parsear datos de calibración
            self.calibration_data = self._parse_seal_data(lines)
            self.is_loaded = True
            
            print(f"[INFO] Archivo SEAL cargado: {self.seal_file}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error al cargar archivo SEAL: {str(e)}")
            return False
    
    def _parse_seal_data(self, lines: list) -> Dict:
        """
        Parsea los datos del archivo SEAL
        
        Args:
            lines: Líneas del archivo
            
        Returns:
            Diccionario con datos de calibración
        """
        data = {}
        
        # Primera línea: resolución
        if lines:
            resolution_line = lines[0].strip()
            resolution_parts = resolution_line.split()
            if len(resolution_parts) >= 2:
                try:
                    data['width'] = int(resolution_parts[0])
                    data['height'] = int(resolution_parts[1])
                except ValueError:
                    data['width'] = 1280
                    data['height'] = 720
        
        # Buscar línea de parámetros intrínsecos (normalmente la línea 5)
        if len(lines) >= 5:
            intrinsic_line = lines[4]  # Índice 4 = línea 5
            intrinsic_values = self._extract_floats(intrinsic_line)
            
            if len(intrinsic_values) >= 9:
                # fx, fy, cx, cy, k1, k2, p1, p2, k3
                data['fx'] = intrinsic_values[0]
                data['fy'] = intrinsic_values[1]
                data['cx'] = intrinsic_values[2]
                data['cy'] = intrinsic_values[3]
                data['k1'] = intrinsic_values[4]
                data['k2'] = intrinsic_values[5]
                data['p1'] = intrinsic_values[6]
                data['p2'] = intrinsic_values[7]
                data['k3'] = intrinsic_values[8]
        
        # Buscar metadatos en la última línea
        if lines:
            metadata_line = lines[-1]
            if metadata_line.startswith('***'):
                # Extraer metadatos
                dev_id_match = re.search(r'DevID:([^*]+)', metadata_line)
                if dev_id_match:
                    data['device_id'] = dev_id_match.group(1).strip()
                
                date_match = re.search(r'CalibrateDate:([^*]+)', metadata_line)
                if date_match:
                    data['calibration_date'] = date_match.group(1).strip()
                
                type_match = re.search(r'Type:([^*]+)', metadata_line)
                if type_match:
                    data['calibration_type'] = type_match.group(1).strip()
                
                version_match = re.search(r'SoftVersion:([^*\s]+)', metadata_line)
                if version_match:
                    data['software_version'] = version_match.group(1).strip()
        
        return data
    
    def _extract_floats(self, line: str) -> list:
        """
        Extrae todos los números flotantes de una línea
        
        Args:
            line: Línea de texto
            
        Returns:
            Lista de valores flotantes
        """
        # Patrón para encontrar números flotantes
        pattern = r'[-+]?\d+(?:\.\d+)?'
        matches = re.findall(pattern, line)
        return [float(match) for match in matches]
    
    def get_camera_matrix(self) -> np.ndarray:
        """
        Obtiene la matriz de cámara K
        
        Returns:
            Matriz de cámara 3x3
        """
        if not self.is_loaded:
            return np.eye(3)
        
        fx = self.calibration_data.get('fx', 1000.0)
        fy = self.calibration_data.get('fy', 1000.0)
        cx = self.calibration_data.get('cx', 640.0)
        cy = self.calibration_data.get('cy', 360.0)
        
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        
        return K
    
    def get_distortion_coefficients(self) -> np.ndarray:
        """
        Obtiene los coeficientes de distorsión
        
        Returns:
            Vector de coeficientes de distorsión
        """
        if not self.is_loaded:
            return np.zeros(5)
        
        k1 = self.calibration_data.get('k1', 0.0)
        k2 = self.calibration_data.get('k2', 0.0)
        p1 = self.calibration_data.get('p1', 0.0)
        p2 = self.calibration_data.get('p2', 0.0)
        k3 = self.calibration_data.get('k3', 0.0)
        
        dist_coeffs = np.array([k1, k2, p1, p2, k3])
        return dist_coeffs
    
    def convert_to_npz(self, output_file: str):
        """
        Convierte el archivo SEAL a formato .npz
        
        Args:
            output_file: Ruta del archivo de salida
        """
        if not self.is_loaded:
            print("[ERROR] Archivo SEAL no cargado")
            return
        
        try:
            # Crear diccionario con datos
            npz_data = {
                'camera_matrix': self.get_camera_matrix(),
                'dist_coeffs': self.get_distortion_coefficients(),
                'width': self.calibration_data.get('width', 1280),
                'height': self.calibration_data.get('height', 720)
            }
            
            # Añadir metadatos si existen
            if 'device_id' in self.calibration_data:
                npz_data['device_id'] = self.calibration_data['device_id']
            if 'calibration_date' in self.calibration_data:
                npz_data['calibration_date'] = self.calibration_data['calibration_date']
            
            # Guardar archivo
            np.savez(output_file, **npz_data)
            print(f"[INFO] Archivo NPZ guardado: {output_file}")
            
        except Exception as e:
            print(f"[ERROR] Error al convertir a NPZ: {str(e)}")
    
    def get_calibration_info(self) -> Dict:
        """
        Obtiene información completa de la calibración
        
        Returns:
            Diccionario con toda la información de calibración
        """
        if not self.is_loaded:
            return {}
        
        info = {
            'resolution': (self.calibration_data.get('width', 1280), 
                          self.calibration_data.get('height', 720)),
            'camera_matrix': self.get_camera_matrix(),
            'distortion_coefficients': self.get_distortion_coefficients(),
            'focal_length': (self.calibration_data.get('fx', 1000.0), 
                           self.calibration_data.get('fy', 1000.0)),
            'principal_point': (self.calibration_data.get('cx', 640.0), 
                              self.calibration_data.get('cy', 360.0))
        }
        
        # Añadir metadatos si existen
        for key in ['device_id', 'calibration_date', 'calibration_type', 'software_version']:
            if key in self.calibration_data:
                info[key] = self.calibration_data[key]
        
        return info
    
    def undistort_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Corrige la distorsión de una imagen usando los parámetros de calibración SEAL
        
        Args:
            image: Imagen a corregir
            
        Returns:
            Imagen corregida o None si falla
        """
        if not self.is_loaded:
            print("[ERROR] Calibración SEAL no cargada")
            return None
        
        try:
            camera_matrix = self.get_camera_matrix()
            dist_coeffs = self.get_distortion_coefficients()
            
            # Corregir distorsión
            undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
            return undistorted
            
        except Exception as e:
            print(f"[ERROR] Error al corregir imagen: {str(e)}")
            return None


def demo_seal_processing():
    """Demostración de procesamiento de archivos SEAL"""
    print("=== Procesamiento de Archivos de Calibración SEAL ===")
    
    # Buscar archivos SEAL en el directorio actual
    seal_files = []
    for filename in os.listdir('.'):
        if filename.endswith('_seal.txt') or (filename.startswith('calib') and filename.endswith('.txt')):
            seal_files.append(filename)
    
    if not seal_files:
        print("[INFO] No se encontraron archivos SEAL en el directorio actual")
        print("[INFO] Creando archivo SEAL de ejemplo...")
        create_sample_seal_file("sample_seal_calibration.txt")
        seal_files = ["sample_seal_calibration.txt"]
    
    # Procesar cada archivo SEAL encontrado
    for seal_file in seal_files:
        print(f"\n--- Procesando {seal_file} ---")
        
        # Inicializar procesador
        processor = SEALCalibrationProcessor(seal_file)
        
        if processor.is_loaded:
            # Mostrar información de calibración
            info = processor.get_calibration_info()
            print(f"Resolución: {info['resolution']}")
            print(f"Longitud focal: {info['focal_length']}")
            print(f"Punto principal: {info['principal_point']}")
            
            # Convertir a NPZ
            npz_filename = seal_file.replace('.txt', '.npz')
            processor.convert_to_npz(npz_filename)
            
            # Si hay imágenes para corregir, hacerlo
            if os.path.exists("sample_images"):
                print("Aplicando corrección a imágenes de ejemplo...")
                # Este sería el siguiente paso en el procesamiento


def create_sample_seal_file(filename: str):
    """
    Crea un archivo SEAL de ejemplo para demostración
    
    Args:
        filename: Nombre del archivo a crear
    """
    sample_content = """1280 720
Calibración de ejemplo
Otros parámetros...
Más datos de calibración...
1000.0 1000.0 640.0 360.0 0.1 -0.2 0.01 -0.01 0.05
Datos adicionales...
***DevID:JMS1006207***CalibrateDate:2023-10-15_14-30-00***Type:Factory-12***SoftVersion:3.0.0.1116
"""
    
    with open(filename, 'w') as f:
        f.write(sample_content)
    
    print(f"[INFO] Archivo SEAL de ejemplo creado: {filename}")


def main():
    """Función principal"""
    demo_seal_processing()


if __name__ == "__main__":
    main()