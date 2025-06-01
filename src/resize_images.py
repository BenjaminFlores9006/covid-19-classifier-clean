# ===================================================================
# SCRIPT DE PREPROCESAMIENTO - resize_images.py
# COVID-19 Classifier - Optimizado para hardware limitado
# ===================================================================

import os
import cv2
import numpy as np
from pathlib import Path
import logging
import time
import argparse

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_images(input_dir, output_dir, target_size=(150, 150)):
    """
    Redimensionar imágenes para el clasificador COVID-19
    
    Args:
        input_dir: Directorio de imágenes originales
        output_dir: Directorio de salida
        target_size: Tamaño objetivo (150x150 para optimizar RAM)
    """
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Contador de imágenes procesadas
    processed = 0
    errors = 0
    
    # Formatos de imagen soportados
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    logger.info(f"🚀 Iniciando preprocesamiento...")
    logger.info(f"📁 Input: {input_dir}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"📐 Target size: {target_size}")
    
    start_time = time.time()
    
    # Procesar cada imagen
    for img_file in Path(input_dir).rglob('*'):
        if img_file.suffix.lower() in image_extensions:
            try:
                # Leer imagen
                img = cv2.imread(str(img_file))
                if img is None:
                    logger.warning(f"⚠️ No se pudo leer: {img_file}")
                    errors += 1
                    continue
                
                # Redimensionar
                img_resized = cv2.resize(img, target_size)
                
                # Crear ruta de salida manteniendo estructura
                relative_path = img_file.relative_to(input_dir)
                output_file = output_path / relative_path.with_suffix('.png')
                
                # Crear directorio padre si no existe
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Guardar imagen procesada
                cv2.imwrite(str(output_file), img_resized)
                
                processed += 1
                
                # Mostrar progreso cada 100 imágenes
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed
                    logger.info(f"📸 Procesadas: {processed} ({rate:.1f} img/s)")
                    
            except Exception as e:
                logger.error(f"❌ Error procesando {img_file}: {e}")
                errors += 1
    
    # Estadísticas finales
    total_time = time.time() - start_time
    logger.info("=" * 50)
    logger.info("📊 PROCESAMIENTO COMPLETADO")
    logger.info("=" * 50)
    logger.info(f"✅ Imágenes procesadas: {processed}")
    logger.info(f"❌ Errores: {errors}")
    logger.info(f"⏱️ Tiempo total: {total_time:.2f} segundos")
    logger.info(f"📈 Velocidad promedio: {processed/total_time:.2f} img/s")

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Preprocesador de imágenes COVID-19')
    parser.add_argument('--input', '-i', required=True, help='Directorio de entrada')
    parser.add_argument('--output', '-o', required=True, help='Directorio de salida')
    parser.add_argument('--size', '-s', nargs=2, type=int, default=[150, 150], 
                       help='Tamaño objetivo (ancho alto)')
    
    args = parser.parse_args()
    
    # Ejecutar preprocesamiento
    resize_images(args.input, args.output, tuple(args.size))

if __name__ == "__main__":
    main()