#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_calibration.py
-------------------
Demostración completa del sistema de calibración proyector-cámara
"""

import cv2
import numpy as np
import os
import time
from typing import List
from gray_code_patterns import generate_gray_code_sequence, generate_phase_shift_patterns, save_patterns
from pattern_capture import PatternCapture, PatternDecoder
from projector_calibration import ProjectorCameraCalibration


def demo_pattern_generation():
    """Demostración de generación de patrones"""
    print("=== Demostración de Generación de Patrones ===")
    
    # Generar patrones Gray Code
    print("Generando patrones Gray Code...")
    gray_patterns = generate_gray_code_sequence(1280, 720, 8)
    print(f"Generados {len(gray_patterns)} patrones Gray Code")
    
    # Generar patrones Phase Shifting
    print("Generando patrones Phase Shifting...")
    phase_patterns = generate_phase_shift_patterns(1280, 720, 4)
    print(f"Generados {len(phase_patterns)} patrones Phase Shifting")
    
    # Guardar patrones
    print("Guardando patrones...")
    all_patterns = gray_patterns + phase_patterns
    save_patterns(all_patterns, "demo_patterns", "demo_pattern")
    
    print("Patrones guardados en directorio 'demo_patterns'")
    return all_patterns


def demo_pattern_capture():
    """Demostración de captura de patrones"""
    print("\n=== Demostración de Captura de Patrones ===")
    
    # Generar patrones para la demo
    patterns = generate_gray_code_sequence(640, 480, 4)  # Patrones más pequeños para demo
    
    # Inicializar capturador
    capture = PatternCapture(0)
    
    try:
        if capture.open_camera():
            print("Cámara abierta. Presiona Ctrl+C para detener la captura.")
            print("En una aplicación real, los patrones serían proyectados.")
            
            # Capturar secuencia (simulada)
            captured_files = capture.capture_pattern_sequence(
                patterns, "demo_captures", delay=0.5)
            
            print(f"Capturadas {len(captured_files)} imágenes")
            return captured_files
        else:
            print("No se pudo abrir la cámara")
            return []
            
    except KeyboardInterrupt:
        print("Captura interrumpida")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        capture.close_camera()


def demo_pattern_decoding(captured_files: List[str]):
    """Demostración de decodificación de patrones"""
    print("\n=== Demostración de Decodificación de Patrones ===")
    
    if len(captured_files) == 0:
        print("No hay archivos para decodificar")
        return
    
    # Cargar imágenes
    images = []
    for filename in captured_files[:4]:  # Usar solo las primeras 4
        img = cv2.imread(filename)
        if img is not None:
            images.append(img)
    
    if len(images) == 0:
        print("No se pudieron cargar las imágenes")
        return
    
    # Inicializar decodificador
    decoder = PatternDecoder()
    
    # Generar patrones de referencia
    patterns = generate_gray_code_sequence(640, 480, len(images))
    
    try:
        # Decodificar patrones Gray Code
        print("Decodificando patrones Gray Code...")
        decoded_map = decoder.decode_gray_code(images, patterns)
        print(f"Mapa decodificado de tamaño: {decoded_map.shape}")
        
        # Decodificar Phase Shifting (si hay suficientes imágenes)
        if len(images) >= 3:
            print("Decodificando patrones Phase Shifting...")
            phase_map = decoder.decode_phase_shift(images)
            print(f"Mapa de fase de tamaño: {phase_map.shape}")
            
    except Exception as e:
        print(f"Error en decodificación: {e}")


def demo_full_calibration():
    """Demostración completa de calibración"""
    print("\n=== Demostración Completa de Calibración ===")
    
    # Crear sistema de calibración
    calibration = ProjectorCameraCalibration(
        camera_index=0,
        projector_width=1280,
        projector_height=720
    )
    
    print("Sistema de calibración creado")
    print("Nota: Esta demostración requiere hardware real (cámara y proyector)")
    print("Para una ejecución completa, conecta el hardware y ejecuta:")
    print("python projector_calibration.py")


def main():
    """Función principal de demostración"""
    print("Sistema de Calibración Proyector-Cámara")
    print("======================================")
    
    # Ejecutar demos
    patterns = demo_pattern_generation()
    captured_files = demo_pattern_capture()
    demo_pattern_decoding(captured_files)
    demo_full_calibration()
    
    print("\n=== Demostración Completada ===")
    print("Archivos generados:")
    print("- Patrones en directorio 'demo_patterns'")
    print("- Capturas en directorio 'demo_captures'")
    print("- Sistema listo para calibración completa")


if __name__ == "__main__":
    main()