import os
import cv2
import numpy as np
from pathlib import Path
import logging
import time
import argparse
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def organize_dataset(input_dir, output_dir):
    """
    Organizar dataset en estructura COVID/NORMAL/PNEUMONIA
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Crear directorios de salida
    for class_name in ['COVID', 'NORMAL', 'PNEUMONIA']:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Mapeo de nombres de archivo a clases
    class_mappings = {
        'covid': 'COVID',
        'normal': 'NORMAL', 
        'pneumonia': 'PNEUMONIA',
        'viral': 'PNEUMONIA'
    }
    
    processed = 0
    
    for img_file in input_path.rglob('*'):
        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
            filename_lower = img_file.name.lower()
            
            # Determinar clase basada en el nombre del archivo
            detected_class = None
            for keyword, class_name in class_mappings.items():
                if keyword in filename_lower:
                    detected_class = class_name
                    break
            
            if detected_class:
                # Copiar archivo a la carpeta correspondiente
                output_file = output_path / detected_class / f"{detected_class}_{processed:04d}.png"
                
                # Leer y guardar imagen
                img = cv2.imread(str(img_file))
                if img is not None:
                    cv2.imwrite(str(output_file), img)
                    processed += 1
                    
                    if processed % 100 == 0:
                        logger.info(f"Organizadas: {processed} imágenes")
    
    logger.info(f"✅ Dataset organizado: {processed} imágenes")

def resize_images(input_dir, output_dir, target_size=(150, 150)):
    """
    Redimensionar imágenes para el clasificador COVID-19
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    errors = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    logger.info(f"🚀 Iniciando preprocesamiento...")
    logger.info(f"📁 Input: {input_dir}")
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"📐 Target size: {target_size}")
    
    start_time = time.time()
    
    # Obtener lista de todas las imágenes
    all_images = []
    for img_file in Path(input_dir).rglob('*'):
        if img_file.suffix.lower() in image_extensions:
            all_images.append(img_file)
    
    logger.info(f"📊 Total de imágenes encontradas: {len(all_images)}")
    
    # Procesar con barra de progreso
    for img_file in tqdm(all_images, desc="Procesando imágenes"):
        try:
            # Leer imagen
            img = cv2.imread(str(img_file))
            if img is None:
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
    parser.add_argument('--organize', action='store_true', 
                       help='Organizar dataset en carpetas por clase')
    
    args = parser.parse_args()
    
    # Organizar dataset si se solicita
    if args.organize:
        organize_dataset(args.input, args.output + "_organized")
        args.input = args.output + "_organized"
    
    # Ejecutar preprocesamiento
    resize_images(args.input, args.output, tuple(args.size))

if __name__ == "__main__":
    main()