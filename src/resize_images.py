import os
import cv2
import numpy as np
from pathlib import Path
import logging
import time
import argparse
import shutil
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def organize_covid_radiography_dataset(input_dir, output_dir):
    """
    Organizar específicamente el COVID-19 Radiography Dataset
    Mapea las 4 carpetas originales a las 3 clases del concurso
    Separa images y masks en carpetas distintas
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    logger.info(f"📦 Organizando COVID-19 Radiography Dataset...")
    logger.info(f"📁 Origen: {input_path}")
    logger.info(f"📁 Destino: {output_path}")
    
    # Crear directorios de salida organizados por clase médica
    for class_name in ['COVID', 'NORMAL', 'PNEUMONIA']:
        (output_path / class_name / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / class_name / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Mapeo específico para COVID-19 Radiography Dataset
    covid_radiography_mapping = {
        'COVID': 'COVID',                    # COVID → COVID
        'Normal': 'NORMAL',                  # Normal → NORMAL  
        'Lung_Opacity': 'PNEUMONIA',        # Lung_Opacity → PNEUMONIA
        'Viral Pneumonia': 'PNEUMONIA'      # Viral Pneumonia → PNEUMONIA
    }
    
    logger.info("🗂️ MAPEO DE CLASES:")
    for origen, destino in covid_radiography_mapping.items():
        logger.info(f"   {origen} → {destino}")
    
    # Contadores separados para images y masks
    contadores_images = {clase: 0 for clase in ['COVID', 'NORMAL', 'PNEUMONIA']}
    contadores_masks = {clase: 0 for clase in ['COVID', 'NORMAL', 'PNEUMONIA']}
    total_processed = 0
    
    # Procesar cada carpeta origen
    for carpeta_origen_nombre, clase_destino in covid_radiography_mapping.items():
        carpeta_clase_origen = input_path / carpeta_origen_nombre
        
        if not carpeta_clase_origen.exists():
            logger.warning(f"⚠️ No se encuentra carpeta: {carpeta_origen_nombre}")
            continue
        
        logger.info(f"📁 Procesando: {carpeta_origen_nombre} → {clase_destino}")
        
        # Procesar IMAGES
        carpeta_images = carpeta_clase_origen / 'images'
        if carpeta_images.exists():
            imagenes = list(carpeta_images.glob('*.png'))
            logger.info(f"   📷 Images encontradas: {len(imagenes)}")
            
            for i, archivo_origen in enumerate(imagenes):
                try:
                    nuevo_nombre = f"{clase_destino}_{contadores_images[clase_destino]:05d}.png"
                    archivo_destino = output_path / clase_destino / 'images' / nuevo_nombre
                    
                    shutil.copy2(archivo_origen, archivo_destino)
                    contadores_images[clase_destino] += 1
                    total_processed += 1
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"      📸 Images copiadas: {i + 1}")
                        
                except Exception as e:
                    logger.error(f"❌ Error copiando image {archivo_origen.name}: {e}")
        
        # Procesar MASKS
        carpeta_masks = carpeta_clase_origen / 'masks'
        if carpeta_masks.exists():
            masks = list(carpeta_masks.glob('*.png'))
            logger.info(f"   🎭 Masks encontradas: {len(masks)}")
            
            for i, archivo_origen in enumerate(masks):
                try:
                    nuevo_nombre = f"{clase_destino}_{contadores_masks[clase_destino]:05d}.png"
                    archivo_destino = output_path / clase_destino / 'masks' / nuevo_nombre
                    
                    shutil.copy2(archivo_origen, archivo_destino)
                    contadores_masks[clase_destino] += 1
                    total_processed += 1
                    
                    if (i + 1) % 1000 == 0:
                        logger.info(f"      🎭 Masks copiadas: {i + 1}")
                        
                except Exception as e:
                    logger.error(f"❌ Error copiando mask {archivo_origen.name}: {e}")
        
        logger.info(f"   ✅ Completado: {carpeta_origen_nombre} → {clase_destino}")
    
    # Mostrar resumen separado
    logger.info("📊 RESUMEN DE ORGANIZACIÓN:")
    logger.info("📷 IMAGES:")
    for clase, cantidad in contadores_images.items():
        logger.info(f"   {clase}: {cantidad:,} imágenes")
    
    logger.info("🎭 MASKS:")
    for clase, cantidad in contadores_masks.items():
        logger.info(f"   {clase}: {cantidad:,} máscaras")
    
    logger.info(f"🎉 Total organizado: {total_processed:,} archivos")
    
    return total_processed

def organize_generic_dataset(input_dir, output_dir):
    """
    Organizar dataset genérico basado en nombres de archivo
    (Función original para otros tipos de dataset)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    logger.info(f"📦 Organizando dataset genérico...")
    
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
    return processed

def resize_images(input_dir, output_dir, target_size=(150, 150)):
    """
    Redimensionar imágenes para el clasificador COVID-19
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    errors = 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    logger.info(f"🚀 Iniciando redimensionado...")
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
    for img_file in tqdm(all_images, desc="Redimensionando imágenes"):
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
    logger.info("📊 REDIMENSIONADO COMPLETADO")
    logger.info("=" * 50)
    logger.info(f"✅ Imágenes procesadas: {processed}")
    logger.info(f"❌ Errores: {errors}")
    logger.info(f"⏱️ Tiempo total: {total_time:.2f} segundos")
    if total_time > 0:
        logger.info(f"📈 Velocidad promedio: {processed/total_time:.2f} img/s")

def detect_dataset_type(input_dir):
    """
    Detectar automáticamente el tipo de dataset
    """
    input_path = Path(input_dir)
    
    # Buscar carpetas específicas del COVID-19 Radiography Dataset
    covid_radiography_folders = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
    found_folders = []
    
    for folder in covid_radiography_folders:
        if (input_path / folder).exists():
            found_folders.append(folder)
    
    if len(found_folders) >= 3:  # Si encuentra al menos 3 de las 4 carpetas
        logger.info("🔍 Detectado: COVID-19 Radiography Dataset")
        return "covid_radiography"
    else:
        logger.info("🔍 Detectado: Dataset genérico")
        return "generic"

def main():
    """Función principal con argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(description='Organizador y redimensionador COVID-19 TODO EN UNO')
    parser.add_argument('--input', '-i', required=True, help='Directorio de entrada')
    parser.add_argument('--output', '-o', required=True, help='Directorio de salida')
    parser.add_argument('--size', '-s', nargs=2, type=int, default=[150, 150], 
                       help='Tamaño objetivo (ancho alto)')
    parser.add_argument('--organize', action='store_true', 
                       help='Organizar dataset en carpetas por clase')
    parser.add_argument('--covid-radiography', action='store_true',
                       help='Forzar modo COVID-19 Radiography Dataset')
    parser.add_argument('--generic', action='store_true',
                       help='Forzar modo dataset genérico')
    
    args = parser.parse_args()
    
    logger.info("🦠 COVID-19 Dataset Processor - TODO EN UNO")
    logger.info("=" * 60)
    
    # Verificar directorio de entrada
    if not Path(args.input).exists():
        logger.error(f"❌ No se encuentra: {args.input}")
        return
    
    if args.organize:
        logger.info("📋 PASO 1: Organizando dataset...")
        
        # Detectar tipo de dataset
        if args.covid_radiography:
            dataset_type = "covid_radiography"
        elif args.generic:
            dataset_type = "generic"
        else:
            dataset_type = detect_dataset_type(args.input)
        
        # Organizar según el tipo
        if dataset_type == "covid_radiography":
            total = organize_covid_radiography_dataset(args.input, args.output + "_organized")
        else:
            total = organize_generic_dataset(args.input, args.output + "_organized")
        
        if total > 0:
            logger.info("📋 PASO 2: Redimensionando solo las IMAGES...")
            # Solo redimensionar las images de cada clase
            resize_images(args.output + "_organized", args.output, tuple(args.size))
            
            # Limpiar carpeta temporal
            logger.info("🧹 Limpiando archivos temporales...")
            import shutil
            shutil.rmtree(args.output + "_organized")
            logger.info("✅ Archivos temporales eliminados")
        else:
            logger.error("❌ No se organizaron imágenes")
    else:
        logger.info("📋 Solo redimensionando imágenes...")
        resize_images(args.input, args.output, tuple(args.size))
    
    logger.info("🎉 ¡Procesamiento completado!")
    logger.info("🚀 Siguiente paso: python covid_classifier.py")

if __name__ == "__main__":
    main()