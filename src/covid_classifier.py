import os
import numpy as np
import pandas as pd
import cv2
import time
import json
import logging
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import requests  # Para la integración con Make

# Configurar TensorFlow para CPU optimizado
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class COVID19Classifier:
    """
    Clasificador COVID-19 optimizado para hardware limitado
    """
    
    def __init__(self, img_size=(150, 150), batch_size=8):
        self.img_size = img_size
        self.batch_size = batch_size
        self.classes = ['COVID', 'NORMAL', 'PNEUMONIA']
        self.model = None
        self.history = None
        
        # Crear directorio de resultados
        self.results_dir = Path('results')
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"🧠 COVID-19 Classifier inicializado")
        logger.info(f"📐 Tamaño de imagen: {self.img_size}")
        logger.info(f"📦 Batch size: {self.batch_size}")
    
    def load_dataset(self, data_dir):
        """Cargar dataset de imágenes organizadas por carpetas"""
        logger.info(f"📁 Cargando dataset desde: {data_dir}")
        
        image_paths = []
        labels = []
        class_mapping = {'COVID': 0, 'NORMAL': 1, 'PNEUMONIA': 2}
        data_path = Path(data_dir)
        
        for class_name, class_idx in class_mapping.items():
            class_dir = data_path / class_name
            
            if not class_dir.exists():
                logger.warning(f"⚠️ Directorio {class_name} no encontrado")
                continue
            
            # Buscar imágenes
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_path in class_dir.glob(ext):
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
        
        image_paths = np.array(image_paths)
        labels = np.array(labels)
        
        logger.info(f"📊 Dataset cargado:")
        logger.info(f"   Total imágenes: {len(image_paths)}")
        
        # Mostrar distribución por clase
        unique, counts = np.unique(labels, return_counts=True)
        for i, (class_idx, count) in enumerate(zip(unique, counts)):
            class_name = self.classes[class_idx]
            logger.info(f"   {class_name}: {count} imágenes")
        
        return image_paths, labels
    
    def preprocess_image(self, image_path):
        """Preprocesar imagen individual"""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return None
                
            # Convertir a RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar
            img = cv2.resize(img, self.img_size)
            
            # Normalizar
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"❌ Error procesando {image_path}: {e}")
            return None
    
    def create_data_generator(self, image_paths, labels, shuffle=True, augment=False):
        """Crear generador de datos eficiente en memoria"""
        def generator():
            indices = np.arange(len(image_paths))
            if shuffle:
                np.random.shuffle(indices)
            
            batch_images = []
            batch_labels = []
            
            for idx in indices:
                img = self.preprocess_image(image_paths[idx])
                if img is not None:
                    # Data augmentation básico solo para entrenamiento
                    if augment and np.random.random() > 0.5:
                        # Flip horizontal simple
                        if np.random.random() > 0.5:
                            img = cv2.flip(img, 1)
                    
                    batch_images.append(img)
                    batch_labels.append(labels[idx])
                    
                    if len(batch_images) == self.batch_size:
                        yield np.array(batch_images), np.array(batch_labels)
                        batch_images = []
                        batch_labels = []
            
            # Yield el último batch si no está vacío
            if batch_images:
                yield np.array(batch_images), np.array(batch_labels)
        
        return generator
    
    def create_model(self):
        """Crear modelo optimizado usando MobileNetV2"""
        logger.info("🏗️ Creando modelo MobileNetV2...")
        
        # Base model MobileNetV2 (eficiente para CPU)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            alpha=0.75,  # Reducir parámetros
            weights='imagenet',
            include_top=False
        )
        
        # Congelar capas base inicialmente
        base_model.trainable = False
        
        # Añadir capas de clasificación
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(len(self.classes), activation='softmax', name='predictions')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compilar
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        total_params = model.count_params()
        logger.info(f"✅ Modelo creado:")
        logger.info(f"   Parámetros totales: {total_params:,}")
        logger.info(f"   Memoria estimada: ~{total_params*4/1024/1024:.1f} MB")
        
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=15):
        """Entrenar el modelo"""
        logger.info("🚀 Iniciando entrenamiento...")
        start_time = time.time()
        
        # Crear generadores
        train_gen = self.create_data_generator(X_train, y_train, shuffle=True, augment=True)
        val_gen = self.create_data_generator(X_val, y_val, shuffle=False, augment=False)
        
        # Calcular steps
        steps_per_epoch = max(1, len(X_train) // self.batch_size)
        validation_steps = max(1, len(X_val) // self.batch_size)
        
        logger.info(f"📊 Steps por época: {steps_per_epoch}")
        logger.info(f"📊 Validation steps: {validation_steps}")
        
        # Callbacks simplificados
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=5, 
                restore_best_weights=True,
                monitor='val_accuracy'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=3, 
                factor=0.5,
                monitor='val_accuracy'
            )
        ]
        
        try:
            self.history = self.model.fit(
                train_gen(),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_gen(),
                validation_steps=validation_steps,
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = time.time() - start_time
            logger.info(f"✅ Entrenamiento completado en {training_time:.2f} segundos")
            
            # Guardar modelo
            model_path = self.results_dir / 'covid_model.h5'
            self.model.save(model_path)
            logger.info(f"💾 Modelo guardado en: {model_path}")
            
            return self.history
            
        except Exception as e:
            logger.error(f"❌ Error durante entrenamiento: {e}")
            raise
    
    def evaluate_model(self, X_test, y_test):
        """Evaluar modelo en conjunto de prueba"""
        logger.info("📊 Evaluando modelo...")
        
        test_gen = self.create_data_generator(X_test, y_test, shuffle=False, augment=False)
        
        predictions = []
        true_labels = []
        
        for batch_images, batch_labels in test_gen():
            batch_predictions = self.model.predict(batch_images, verbose=0)
            predictions.extend(np.argmax(batch_predictions, axis=1))
            true_labels.extend(batch_labels)
        
        predictions = np.array(predictions)
        true_labels = np.array(true_labels)
        
        accuracy = np.mean(predictions == true_labels)
        
        # Reporte detallado
        report = classification_report(
            true_labels, predictions, 
            target_names=self.classes, 
            output_dict=True
        )
        
        cm = confusion_matrix(true_labels, predictions)
        
        logger.info("=" * 50)
        logger.info("📊 RESULTADOS DE EVALUACIÓN")
        logger.info("=" * 50)
        logger.info(f"🎯 Precisión total: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Métricas por clase
        for i, class_name in enumerate(self.classes):
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            logger.info(f"📈 {class_name}:")
            logger.info(f"   Precision: {precision:.3f}")
            logger.info(f"   Recall: {recall:.3f}")
            logger.info(f"   F1-Score: {f1:.3f}")
        
        # Verificar criterio del concurso
        criteria_met = accuracy >= 0.85
        if criteria_met:
            logger.info("✅ CRITERIO DE PRECISIÓN CUMPLIDO (≥85%)")
        else:
            logger.warning("⚠️ Precisión por debajo del criterio mínimo (85%)")
        
        # Visualizar resultados
        self.plot_results(cm, report)
        
        return accuracy, report, cm, criteria_met
    
    def plot_results(self, confusion_matrix, classification_report):
        """Crear visualizaciones de resultados"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Matriz de confusión
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes, ax=ax1)
        ax1.set_title('Matriz de Confusión')
        ax1.set_ylabel('Etiqueta Real')
        ax1.set_xlabel('Etiqueta Predicha')
        
        # Métricas por clase
        metrics_df = pd.DataFrame(classification_report).T
        metrics_df = metrics_df.drop(['accuracy', 'macro avg', 'weighted avg'])
        metrics_df[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax2)
        ax2.set_title('Métricas por Clase')
        ax2.set_ylabel('Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Historia de entrenamiento
        if self.history:
            ax3.plot(self.history.history['accuracy'], label='Entrenamiento', linewidth=2)
            ax3.plot(self.history.history['val_accuracy'], label='Validación', linewidth=2)
            ax3.set_title('Precisión del Modelo')
            ax3.set_xlabel('Época')
            ax3.set_ylabel('Precisión')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4.plot(self.history.history['loss'], label='Entrenamiento', linewidth=2)
            ax4.plot(self.history.history['val_loss'], label='Validación', linewidth=2)
            ax4.set_title('Pérdida del Modelo')
            ax4.set_xlabel('Época')
            ax4.set_ylabel('Pérdida')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.results_dir / 'model_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Gráficos guardados en: {plot_path}")
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predicción para una imagen individual"""
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        img_batch = np.expand_dims(img, axis=0)
        prediction = self.model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx])
        predicted_class = self.classes[predicted_class_idx]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                self.classes[i]: float(prediction[0][i]) 
                for i in range(len(self.classes))
            }
        }
    
    def send_results_to_make(self, results, webhook_url=None):
        """Enviar resultados a Make.com para notificación por email"""
        if not webhook_url:
            logger.warning("⚠️ No se proporcionó URL de webhook para Make.com")
            return False
        
        try:
            # Preparar datos para enviar
            payload = {
                'accuracy': results['accuracy'],
                'criteria_met': results['criteria_met']['accuracy_85_percent'],
                'timestamp': results['timestamp'],
                'model_params': results['model_params'],
                'dataset_size': results['dataset_size'],
                'message': f"Clasificador COVID-19 completado con {results['accuracy']:.2%} de precisión"
            }
            
            # Enviar a Make.com
            response = requests.post(webhook_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info("✅ Resultados enviados correctamente a Make.com")
                return True
            else:
                logger.error(f"❌ Error enviando a Make.com: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error conectando con Make.com: {e}")
            return False

def main():
    """Función principal del entrenamiento"""
    logger.info("🦠 COVID-19 Classifier - Iniciando...")
    
    # Configurar rutas (ajusta según tu sistema)
    data_dir = Path('data/processed')  # Cambiar por tu ruta
    
    # Si no existe, intentar con la estructura en C:
    if not data_dir.exists():
        data_dir = Path('C:/Dataset_COVID')  # Ajustar según tu estructura
        
    if not data_dir.exists():
        logger.error(f"❌ Directorio de datos no encontrado: {data_dir}")
        logger.info("💡 Opciones:")
        logger.info("   1. Crear carpetas: COVID, NORMAL, PNEUMONIA en data/processed/")
        logger.info("   2. Ejecutar: python resize_images.py --input C:/tu_carpeta --output data/processed")
        return
    
    try:
        # Inicializar clasificador
        classifier = COVID19Classifier()
        
        # Cargar dataset
        image_paths, labels = classifier.load_dataset(data_dir)
        
        if len(image_paths) == 0:
            logger.error("❌ No se encontraron imágenes en el dataset")
            return
        
        # Split train/validation/test (simplificado)
        X_temp, X_test, y_temp, y_test = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
        )
        
        logger.info(f"📊 División del dataset:")
        logger.info(f"   Entrenamiento: {len(X_train)} imágenes")
        logger.info(f"   Validación: {len(X_val)} imágenes")
        logger.info(f"   Prueba: {len(X_test)} imágenes")
        
        # Crear y entrenar modelo
        classifier.create_model()
        classifier.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluar modelo
        accuracy, report, cm, criteria_met = classifier.evaluate_model(X_test, y_test)
        
        # Guardar métricas finales (JSON corregido)
        results = {
            'accuracy': float(accuracy),
            'timestamp': datetime.now().isoformat(),
            'model_params': int(classifier.model.count_params()),
            'dataset_size': int(len(image_paths)),
            'criteria_met': {
                'accuracy_85_percent': bool(criteria_met)
            },
            'class_metrics': {
                class_name: {
                    'precision': float(report[class_name]['precision']),
                    'recall': float(report[class_name]['recall']),
                    'f1_score': float(report[class_name]['f1-score'])
                }
                for class_name in classifier.classes
            }
        }
        
        results_file = classifier.results_dir / 'final_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Resultados finales guardados en: {results_file}")
        
        # Integración con Make.com (opcional)
        webhook_url = "https://hook.eu1.make.com/tu-webhook-aqui"  # Cambiar por tu webhook
        # classifier.send_results_to_make(results, webhook_url)
        
        logger.info("🎉 ¡Entrenamiento completado exitosamente!")
        
    except Exception as e:
        logger.error(f"❌ Error durante ejecución: {e}")
        raise

if __name__ == "__main__":
    main()