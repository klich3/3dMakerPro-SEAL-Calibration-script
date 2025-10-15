#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gray_code_patterns.py
---------------------
Generación de patrones Gray Code y Phase Shifting para calibración
"""

import cv2
import numpy as np
import os
from typing import List, Tuple


def generate_gray_code_sequence(width: int, height: int, num_bits: int) -> List[np.ndarray]:
    """
    Genera una secuencia de patrones Gray Code
    
    Args:
        width: Ancho de los patrones
        height: Alto de los patrones
        num_bits: Número de bits (determina el número de patrones)
    
    Returns:
        Lista de patrones Gray Code como arrays numpy
    """
    patterns = []
    
    # Generar patrones binarios verticales
    for bit in range(num_bits):
        pattern = np.zeros((height, width), dtype=np.uint8)
        
        # Calcular el ancho de cada banda
        band_width = width // (2 ** (bit + 1))
        
        # Generar el patrón
        for col in range(width):
            # Determinar si estamos en una banda blanca o negra
            band_index = col // band_width
            is_white = (band_index % 2) == 1
            
            if is_white:
                pattern[:, col] = 255
                
        patterns.append(pattern)
    
    return patterns


def generate_phase_shift_patterns(width: int, height: int, num_patterns: int = 4) -> List[np.ndarray]:
    """
    Genera patrones sinusoidales para Phase Shifting
    
    Args:
        width: Ancho de los patrones
        height: Alto de los patrones
        num_patterns: Número de patrones (normalmente 3 o 4)
    
    Returns:
        Lista de patrones sinusoidales como arrays numpy
    """
    patterns = []
    
    # Generar patrones sinusoidales
    for i in range(num_patterns):
        pattern = np.zeros((height, width), dtype=np.uint8)
        
        # Calcular el desfase
        phase_shift = 2 * np.pi * i / num_patterns
        
        # Generar el patrón sinusoidal horizontal
        for col in range(width):
            # Calcular el valor sinusoidal (normalizado entre 0 y 1)
            x = col / width
            intensity = 0.5 * (1 + np.cos(2 * np.pi * x - phase_shift))
            
            # Convertir a escala de 0-255
            pattern[:, col] = int(intensity * 255)
            
        patterns.append(pattern)
    
    return patterns


def generate_combined_patterns(width: int, height: int, gray_bits: int = 8) -> List[np.ndarray]:
    """
    Genera una combinación de patrones Gray Code y Phase Shifting
    
    Args:
        width: Ancho de los patrones
        height: Alto de los patrones
        gray_bits: Número de bits para Gray Code
    
    Returns:
        Lista combinada de patrones
    """
    # Generar patrones Gray Code
    gray_patterns = generate_gray_code_sequence(width, height, gray_bits)
    
    # Generar patrones Phase Shifting
    phase_patterns = generate_phase_shift_patterns(width, height, 4)
    
    # Combinar patrones
    all_patterns = gray_patterns + phase_patterns
    
    return all_patterns


def save_patterns(patterns: List[np.ndarray], output_dir: str, prefix: str = "pattern"):
    """
    Guarda los patrones como archivos de imagen
    
    Args:
        patterns: Lista de patrones
        output_dir: Directorio de salida
        prefix: Prefijo para los nombres de archivo
    """
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar cada patrón
    for i, pattern in enumerate(patterns):
        filename = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
        cv2.imwrite(filename, pattern)
        print(f"[INFO] Patrón guardado: {filename}")


def display_patterns(patterns: List[np.ndarray], window_name: str = "Patrones", delay: int = 1000):
    """
    Muestra los patrones en una ventana
    
    Args:
        patterns: Lista de patrones
        window_name: Nombre de la ventana
        delay: Retraso entre patrones en milisegundos
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for i, pattern in enumerate(patterns):
        cv2.imshow(window_name, pattern)
        print(f"[INFO] Mostrando patrón {i+1}/{len(patterns)}")
        
        # Esperar la tecla o el tiempo especificado
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()


def main():
    """Función principal de demostración"""
    print("[INFO] Generando patrones Gray Code y Phase Shifting...")
    
    # Parámetros
    width, height = 1280, 720
    gray_bits = 8
    
    # Generar patrones combinados
    patterns = generate_combined_patterns(width, height, gray_bits)
    
    print(f"[INFO] Generados {len(patterns)} patrones")
    
    # Guardar patrones
    save_patterns(patterns, "calib_patterns", "calib_pattern")
    
    # Mostrar patrones (opcional)
    # display_patterns(patterns, "Patrones de Calibración", 500)
    
    print("[INFO] Generación completada")


if __name__ == "__main__":
    main()