#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_calibration.py
---------------------
Procesa archivos de calibración .npz y aplica corrección a imágenes
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple


class CalibrationProcessor:
    def __init__(self, calibration_file: str):
        """
        Inicializa el procesador de calibración
        
        Args:
            calibration_file: Ruta al archivo .npz de calibración
        """
        self.calibration_file = calibration_file
        self.camera_matrix = None
        self.dist_coeffs = None
        self.projector_matrix = None
        self.projector_dist_coeffs = None
        self.projector_width = None
        self.projector_height = None
        self.is_loaded = False
        
        # Cargar parámetros de calibración
        self.load_calibration()
    
    def load_calibration(self) -> bool:
        """
        Carga los parámetros de calibración desde el archivo .npz
        
        Returns:
            True si se cargó correctamente
        """
        try:
            if not os.path.exists(self.calibration_file):
                print(f"[ERROR] Archivo de calibración no encontrado: {self.calibration_file}")
                return False
            
            # Cargar datos
            data = np.load(self.calibration_file)
            
            # Extraer parámetros
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            
            # Parámetros opcionales del proyector
            if 'projector_matrix' in data:
                self.projector_matrix = data['projector_matrix']
            if 'projector_dist_coeffs' in data:
                self.projector_dist_coeffs = data['projector_dist_coeffs']
            if 'projector_width' in data:
                self.projector_width = int(data['projector_width'])
            if 'projector_height' in data:
                self.projector_height = int(data['projector_height'])
            
            self.is_loaded = True
            print(f"[INFO] Calibración cargada desde {self.calibration_file}")
            print(f"Matriz de cámara:\n{self.camera_matrix}")
            print(f"Coeficientes de distorsión:\n{self.dist_coeffs}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error al cargar calibración: {str(e)}")
            return False
    
    def undistort_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Corrige la distorsión de una imagen usando los parámetros de calibración
        
        Args:
            image: Imagen a corregir
            
        Returns:
            Imagen corregida o None si falla
        """
        if not self.is_loaded:
            print("[ERROR] Calibración no cargada")
            return None
        
        try:
            # Corregir distorsión
            undistorted = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)
            return undistorted
            
        except Exception as e:
            print(f"[ERROR] Error al corregir imagen: {str(e)}")
            return None
    
    def get_calibration_info(self) -> dict:
        """
        Obtiene información sobre la calibración cargada
        
        Returns:
            Diccionario con información de calibración
        """
        if not self.is_loaded:
            return {}
        
        info = {
            'camera_matrix': self.camera_matrix,
            'distortion_coefficients': self.dist_coeffs,
            'focal_length': (self.camera_matrix[0, 0], self.camera_matrix[1, 1]),
            'principal_point': (self.camera_matrix[0, 2], self.camera_matrix[1, 2]),
            'has_projector_calibration': self.projector_matrix is not None
        }
        
        if self.projector_width and self.projector_height:
            info['projector_resolution'] = (self.projector_width, self.projector_height)
        
        return info
    
    def apply_calibration_to_images(self, input_dir: str, output_dir: str):
        """
        Aplica la calibración a todas las imágenes en un directorio
        
        Args:
            input_dir: Directorio con imágenes de entrada
            output_dir: Directorio para imágenes corregidas
        """
        if not self.is_loaded:
            print("[ERROR] Calibración no cargada")
            return
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Procesar todas las imágenes
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, f"calibrated_{filename}")
                
                # Cargar imagen
                image = cv2.imread(input_path)
                if image is None:
                    print(f"[ERROR] No se pudo cargar {input_path}")
                    continue
                
                # Aplicar corrección
                corrected = self.undistort_image(image)
                if corrected is not None:
                    # Guardar imagen corregida
                    cv2.imwrite(output_path, corrected)
                    print(f"[INFO] Imagen corregida guardada: {output_path}")
                else:
                    print(f"[ERROR] No se pudo corregir {input_path}")


def demo_calibration_processing():
    """Demostración de procesamiento de calibración"""
    print("=== Procesamiento de Archivo de Calibración ===")
    
    # Ruta al archivo de calibración (ajusta según tu archivo)
    calibration_file = "projector_camera_calibration.npz"
    
    # Verificar si existe el archivo de calibración
    if not os.path.exists(calibration_file):
        print(f"[INFO] Archivo de calibración {calibration_file} no encontrado")
        print("[INFO] Generando un archivo de ejemplo...")
        
        # Crear un archivo de ejemplo para demostración
        create_sample_calibration_file(calibration_file)
    
    # Inicializar procesador
    processor = CalibrationProcessor(calibration_file)
    
    if processor.is_loaded:
        # Mostrar información de calibración
        info = processor.get_calibration_info()
        print("\nInformación de Calibración:")
        print(f"Longitud focal: {info['focal_length']}")
        print(f"Punto principal: {info['principal_point']}")
        print(f"Calibración de proyector: {info['has_projector_calibration']}")
        
        # Si hay un directorio de imágenes de ejemplo, procesarlas
        sample_images_dir = "tmp"
        if os.path.exists(sample_images_dir):
            print(f"\nProcesando imágenes en {sample_images_dir}...")
            processor.apply_calibration_to_images(sample_images_dir, "calibrated_images")
        else:
            print(f"\nDirectorio {sample_images_dir} no encontrado")
            print("Para procesar imágenes, crea un directorio 'sample_images' con imágenes JPG/PNG")
    else:
        print("[ERROR] No se pudo cargar la calibración")


def create_sample_calibration_file(filename: str):
    """
    Crea un archivo de calibración de ejemplo para demostración
    
    Args:
        filename: Nombre del archivo a crear
    """
    # Parámetros de ejemplo para una cámara típica
    camera_matrix = np.array([
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 360.0],
        [0.0, 0.0, 1.0]
    ])
    
    dist_coeffs = np.array([0.1, -0.2, 0.01, -0.01, 0.05])
    
    # Guardar datos
    np.savez(filename, 
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             projector_width=1280,
             projector_height=720)
    
    print(f"[INFO] Archivo de calibración de ejemplo creado: {filename}")


def main():
    """Función principal"""
    demo_calibration_processing()


if __name__ == "__main__":
    main()