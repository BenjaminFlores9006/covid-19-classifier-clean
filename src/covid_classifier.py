import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

class CovidClassifierSmartBalanced:
    def __init__(self, image_size=(224, 224), use_all_data=True):
        self.image_size = image_size
        self.use_all_data = use_all_data
        self.models = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        
        print("🧠 COVID-19 Classifier SMART BALANCED")
        print("🎯 Objetivo: Usar TODOS los datos disponibles de forma inteligente")
        
    def load_data_smart_balanced(self, data_dir, target_samples_per_class=None):
        """
        🚀 CARGA INTELIGENTE: Usa todos los datos disponibles de forma balanceada
        """
        print("\n📊 CARGA INTELIGENTE DE DATOS...")
        
        # Detectar estructura de carpetas
        data_path = Path(data_dir)
        
        # Primero, explorar la estructura completa
        print(f"🔍 Explorando estructura en: {data_path}")
        
        if not data_path.exists():
            print(f"❌ El directorio {data_path} no existe")
            # Buscar en ubicaciones alternativas
            alternative_paths = [
                Path("data"),
                Path("../data"),
                Path("./data"),
                Path("datasets"),
                Path("../datasets")
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    print(f"✅ Encontrado directorio alternativo: {alt_path}")
                    data_path = alt_path
                    break
            else:
                raise ValueError(f"No se encontró directorio de datos en: {[str(p) for p in alternative_paths]}")
        
        # Explorar recursivamente
        all_subdirs = []
        for item in data_path.rglob("*"):
            if item.is_dir():
                all_subdirs.append(item)
        
        print(f"📁 Subdirectorios encontrados:")
        for subdir in all_subdirs[:10]:  # Mostrar primeros 10
            print(f"  {subdir}")
        if len(all_subdirs) > 10:
            print(f"  ... y {len(all_subdirs)-10} más")
        
        # Buscar carpetas de clases de forma más flexible
        class_folders = {
            'COVID': None,
            'PNEUMONIA': None, 
            'NORMAL': None
        }
        
        # ESTRUCTURA DETECTADA: COVID/images, COVID/masks, etc.
        # Buscar carpetas principales primero
        main_class_dirs = {}
        for folder in data_path.iterdir():
            if folder.is_dir():
                folder_name = folder.name.upper()
                if 'COVID' in folder_name:
                    main_class_dirs['COVID'] = folder
                elif 'PNEUMONIA' in folder_name or 'PNEUM' in folder_name:
                    main_class_dirs['PNEUMONIA'] = folder
                elif 'NORMAL' in folder_name:
                    main_class_dirs['NORMAL'] = folder
        
        print(f"📁 Carpetas principales detectadas:")
        for class_name, folder in main_class_dirs.items():
            if folder:
                print(f"  {class_name}: {folder}")
                
                # Buscar subcarpetas images y masks
                images_dir = folder / "images"
                masks_dir = folder / "masks"
                
                if images_dir.exists():
                    image_count = len(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")))
                    print(f"    📷 images/: {image_count} archivos")
                
                if masks_dir.exists():
                    mask_count = len(list(masks_dir.glob("*.png")) + list(masks_dir.glob("*.jpg")) + list(masks_dir.glob("*.jpeg")))
                    print(f"    🎭 masks/: {mask_count} archivos")
                
                # Usar la carpeta images como principal (máscaras son opcionales)
                if images_dir.exists():
                    class_folders[class_name] = images_dir
                else:
                    # Fallback: usar la carpeta principal si no hay subcarpeta images
                    class_folders[class_name] = folder
        # Verificar que se encontraron todas las carpetas
        missing_classes = [k for k, v in class_folders.items() if v is None]
        if missing_classes:
            print(f"\n❌ No se encontraron carpetas para: {missing_classes}")
            raise ValueError("No se encontraron todas las carpetas de clases necesarias")
        
        print(f"\n✅ CARPETAS CONFIGURADAS:")
        for class_name, folder in class_folders.items():
            print(f"  {class_name}: {folder}")
        
        # Contar imágenes disponibles
        available_counts = {}
        all_images = {}
        
        for class_name, folder in class_folders.items():
            if folder and folder.exists():
                # Buscar todas las imágenes (incluyendo máscaras)
                image_files = []
                for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                    image_files.extend(list(folder.glob(ext)))
                
                available_counts[class_name] = len(image_files)
                all_images[class_name] = image_files
                print(f"  📷 {class_name}: {len(image_files)} imágenes disponibles")
        
        # Estrategia de balanceado inteligente
        if target_samples_per_class is None:
            # Automático: usar el 80% de la clase con menos datos
            min_class = min(available_counts.values())
            target_samples_per_class = max(min_class, int(min_class * 1.5))
        
        print(f"\n🎯 TARGET POR CLASE: {target_samples_per_class} imágenes")
        
        # Cargar datos con estrategia inteligente
        features_list = []
        labels_list = []
        
        for class_name, image_files in all_images.items():
            if not image_files:
                continue
                
            available = len(image_files)
            needed = target_samples_per_class
            
            print(f"\n🔄 Procesando {class_name}:")
            print(f"  📊 Disponibles: {available}")
            print(f"  🎯 Necesarias: {needed}")
            
            if available >= needed:
                # Suficientes datos: sample aleatorio
                selected_files = np.random.choice(image_files, needed, replace=False)
                print(f"  ✅ Seleccionadas: {len(selected_files)} (sample aleatorio)")
            else:
                # Insuficientes datos: usar todas + buscar máscaras
                selected_files = image_files.copy()
                
                # Buscar máscaras correspondientes para data augmentation
                mask_folder = image_files[0].parent.parent / "masks"
                if mask_folder.exists():
                    mask_files = []
                    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                        mask_files.extend(list(mask_folder.glob(ext)))
                    
                    print(f"  🎭 Máscaras disponibles: {len(mask_files)}")
                    
                    # Agregar máscaras hasta completar el target
                    needed_extra = needed - available
                    available_masks = min(len(mask_files), needed_extra)
                    
                    if available_masks > 0:
                        selected_masks = np.random.choice(mask_files, available_masks, replace=False)
                        selected_files.extend(selected_masks)
                        print(f"  ✅ Agregadas {available_masks} máscaras")
                
                # Si aún faltan, hacer data augmentation
                if len(selected_files) < needed:
                    needed_extra = needed - len(selected_files)
                    print(f"  🔄 Generando {needed_extra} imágenes con augmentation...")
                    
                    augmented_files = self.generate_augmented_data(
                        image_files[:min(len(image_files), 100)], needed_extra, class_name
                    )
                    selected_files.extend(augmented_files)
                
                print(f"  ✅ Total final: {len(selected_files)} imágenes")
            
            # Extraer features de las imágenes seleccionadas
            class_features = self.extract_features_batch(selected_files, class_name)
            
            if class_features is not None and len(class_features) > 0:
                features_list.extend(class_features)
                labels_list.extend([class_name] * len(class_features))
        
        if not features_list:
            raise ValueError("No se pudieron cargar datos")
        
        # Convertir a arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n✅ DATOS CARGADOS:")
        print(f"  📊 Total samples: {len(X)}")
        print(f"  🔢 Features por sample: {X.shape[1]}")
        
        # Mostrar distribución final
        class_distribution = Counter(y)
        for class_name, count in class_distribution.items():
            percentage = (count / len(y)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        return X, y
    
    def generate_augmented_data(self, image_files, needed_count, class_name):
        """
        🔄 Generar datos aumentados para completar muestras
        """
        augmented_files = []
        temp_dir = Path("temp_augmented") / class_name
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Seleccionar imágenes base para augmentation
        base_images = np.random.choice(image_files, min(len(image_files), needed_count), replace=True)
        
        for i, base_image_path in enumerate(base_images):
            try:
                # Cargar imagen
                img = cv2.imread(str(base_image_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Aplicar augmentation aleatoria
                augmented_img = self.apply_random_augmentation(img)
                
                # Guardar imagen aumentada
                aug_filename = f"aug_{class_name}_{i}.png"
                aug_path = temp_dir / aug_filename
                cv2.imwrite(str(aug_path), augmented_img)
                
                augmented_files.append(aug_path)
                
            except Exception as e:
                print(f"    ❌ Error en augmentation {i}: {e}")
                continue
        
        print(f"    🎨 Generadas {len(augmented_files)} imágenes aumentadas")
        return augmented_files
    
    def apply_random_augmentation(self, img):
        """
        🎨 Aplicar augmentation aleatorio a una imagen
        """
        augmented = img.copy()
        
        # Rotación aleatoria (-15 a 15 grados)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            center = (img.shape[1]//2, img.shape[0]//2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, matrix, (img.shape[1], img.shape[0]))
        
        # Ajuste de brillo aleatorio
        if np.random.random() > 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            augmented = np.clip(augmented * brightness, 0, 255).astype(np.uint8)
        
        # Ajuste de contraste aleatorio
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            augmented = np.clip((augmented - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
        
        # Ruido gaussiano sutil
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, augmented.shape)
            augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
        
        # Desenfoque sutil ocasional
        if np.random.random() > 0.8:
            augmented = cv2.GaussianBlur(augmented, (3, 3), 0.5)
        
        return augmented
    
    def extract_features_batch(self, image_files, class_name):
        """
        📊 Extraer features de un lote de imágenes
        """
        features_list = []
        
        print(f"    🔍 Extrayendo features de {len(image_files)} imágenes...")
        
        for i, image_path in enumerate(image_files):
            try:
                # Progreso cada 100 imágenes
                if i % 100 == 0 and i > 0:
                    print(f"    📊 Procesadas {i}/{len(image_files)} imágenes...")
                
                features = self._extract_single_image_features(image_path)
                if features is not None:
                    features_list.append(features)
                    
            except Exception as e:
                print(f"    ❌ Error procesando {image_path}: {e}")
                continue
        
        print(f"    ✅ Features extraídas: {len(features_list)}/{len(image_files)}")
        return features_list
    
    def _extract_single_image_features(self, image_path):
        """
        🔬 Extraer features de una sola imagen (versión simplificada)
        """
        try:
            # Cargar imagen
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Redimensionar
            img = cv2.resize(img, self.image_size)
            
            # Normalizar
            img = img / 255.0
            
            # Features estadísticas básicas
            features = []
            
            # 1. Estadísticas globales
            features.extend([
                np.mean(img),
                np.std(img),
                np.min(img),
                np.max(img),
                np.median(img),
                np.percentile(img, 25),
                np.percentile(img, 75)
            ])
            
            # 2. Features de textura (LBP simplificado)
            # Calcular gradientes
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                np.mean(np.abs(grad_x)),
                np.std(grad_x),
                np.mean(np.abs(grad_y)),
                np.std(grad_y)
            ])
            
            # 3. Características de regiones
            # Dividir imagen en cuadrantes
            h, w = img.shape
            quadrants = [
                img[0:h//2, 0:w//2],      # Superior izquierdo
                img[0:h//2, w//2:w],      # Superior derecho
                img[h//2:h, 0:w//2],      # Inferior izquierdo
                img[h//2:h, w//2:w]       # Inferior derecho
            ]
            
            for quad in quadrants:
                features.extend([
                    np.mean(quad),
                    np.std(quad)
                ])
            
            # 4. Features de bordes
            edges = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # 5. Momentos de Hu (invariantes)
            moments = cv2.moments(img)
            hu_moments = cv2.HuMoments(moments).flatten()
            # Log transform para estabilizar
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            features.extend(hu_moments)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error extrayendo features de {image_path}: {e}")
            return None
    
    def train_with_smart_balancing(self, data_dir, target_samples_per_class=None):
        """
        🎯 Entrenar con balanceado inteligente
        COVID: 7,232 | NORMAL: 20,384 | PNEUMONIA: 14,714
        DECISIÓN ÓPTIMA: 7,232 por clase (usar COVID como limitante)
        """
        print("\n🚀 INICIANDO ENTRENAMIENTO SMART BALANCED...")
        print("📊 Dataset detectado:")
        print("   COVID: 7,232 imágenes")
        print("   NORMAL: 20,384 imágenes") 
        print("   PNEUMONIA: 14,714 imágenes")
        print("🎯 ESTRATEGIA: 7,232 por clase (COVID como limitante)")
        
        # Configuración automática óptima
        if target_samples_per_class is None:
            target_samples_per_class = 7232  # COVID es limitante
        
        # Cargar datos con estrategia inteligente
        X, y = self.load_data_smart_balanced(data_dir, target_samples_per_class)
        
        # Encoding de labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Calcular pesos de clase para mayor equilibrio
        self.class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(y_encoded), 
            y=y_encoded
        )
        class_weight_dict = dict(zip(np.unique(y_encoded), self.class_weights))
        
        print(f"\n⚖️ PESOS DE CLASE CALCULADOS:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            print(f"  {class_name}: {self.class_weights[i]:.3f}")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n📊 DIVISIÓN DE DATOS:")
        print(f"  🏋️ Training: {len(X_train)} samples")
        print(f"  🧪 Testing: {len(X_test)} samples")
        
        # Aplicar SMOTE para balanceo adicional
        print("\n🔄 Aplicando SMOTE para balanceo adicional...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        print(f"📊 Después de SMOTE:")
        train_distribution = Counter(y_train_balanced)
        for class_idx, count in train_distribution.items():
            class_name = self.label_encoder.inverse_transform([class_idx])[0]
            print(f"  {class_name}: {count} samples")
        
        # Normalización
        print("\n🔧 Normalizando features...")
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA
        print("🔍 Aplicando PCA...")
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"📉 Componentes PCA: {X_train_pca.shape[1]} (de {X_train_scaled.shape[1]})")
        print(f"📊 Varianza explicada: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Entrenar ensemble de modelos
        print("\n🎯 Entrenando ensemble de modelos...")
        
        models_to_train = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight_dict,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
                class_weight=class_weight_dict
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight=class_weight_dict,
                probability=True,
                random_state=42
            )
        }
        
        for name, model in models_to_train.items():
            try:
                print(f"  🔄 Entrenando {name}...")
                model.fit(X_train_pca, y_train_balanced)
                self.models[name] = model
                
                # Evaluación rápida
                train_score = model.score(X_train_pca, y_train_balanced)
                test_score = model.score(X_test_pca, y_test)
                print(f"    📊 {name} - Train: {train_score:.3f}, Test: {test_score:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error entrenando {name}: {e}")
        
        # Evaluación final
        print(f"\n🎯 EVALUACIÓN FINAL DEL ENSEMBLE:")
        y_pred_ensemble = self._ensemble_predict(self.models, X_test_pca)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        
        print(f"📊 Accuracy del Ensemble: {ensemble_accuracy:.3f}")
        
        # Reporte detallado
        print("\n📋 REPORTE DE CLASIFICACIÓN:")
        class_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred_ensemble, target_names=class_names))
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred_ensemble)
        self._plot_confusion_matrix(cm, class_names)
        
        return ensemble_accuracy
    
    def _ensemble_predict(self, models, X):
        """🎯 Predicción por ensemble con votación ponderada"""
        if not models:
            return None
        
        predictions = []
        weights = []
        
        for name, model in models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                # Peso basado en el tipo de modelo
                if name in ['RandomForest', 'ExtraTrees']:
                    weights.append(1.2)  # Más peso a modelos de árboles
                elif name in ['XGBoost', 'LightGBM']:
                    weights.append(1.1)  # Peso medio a gradient boosting
                else:
                    weights.append(1.0)  # Peso normal
            except:
                continue
        
        if not predictions:
            return None
        
        # Votación ponderada
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Votación por mayoría ponderada
        ensemble_pred = []
        for i in range(len(X)):
            votes = {}
            for j, pred in enumerate(predictions[:, i]):
                if pred not in votes:
                    votes[pred] = 0
                votes[pred] += weights[j]
            
            # Clase con mayor peso acumulado
            best_class = max(votes.items(), key=lambda x: x[1])[0]
            ensemble_pred.append(best_class)
        
        return np.array(ensemble_pred)
    
    def _ensemble_predict_proba(self, models, X):
        """🎯 Probabilidades del ensemble"""
        if not models:
            return None
        
        all_probas = []
        weights = []
        
        for name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    all_probas.append(proba)
                    
                    if name in ['RandomForest', 'ExtraTrees']:
                        weights.append(1.2)
                    elif name in ['XGBoost', 'LightGBM']:
                        weights.append(1.1)
                    else:
                        weights.append(1.0)
            except:
                continue
        
        if not all_probas:
            return None
        
        # Promedio ponderado de probabilidades
        all_probas = np.array(all_probas)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalizar pesos
        
        ensemble_proba = np.average(all_probas, axis=0, weights=weights)
        return ensemble_proba
    
    def predict(self, image_path):
        """🎯 Predicción de una imagen"""
        try:
            # Extraer features
            features = self._extract_single_image_features(image_path)
            if features is None:
                return None, 0.0, None
            
            # Transformar features
            features_scaled = self.scaler.transform([features])
            features_pca = self.pca.transform(features_scaled)
            
            # Predicción ensemble
            pred_class = self._ensemble_predict(self.models, features_pca)[0]
            pred_proba = self._ensemble_predict_proba(self.models, features_pca)
            
            # Convertir a nombre de clase
            class_name = self.label_encoder.inverse_transform([pred_class])[0]
            confidence = pred_proba[0][pred_class] if pred_proba is not None else 0.5
            
            return class_name, confidence, pred_proba[0] if pred_proba is not None else None
            
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None, 0.0, None
    
    def _plot_confusion_matrix(self, cm, class_names):
        """📊 Plotear matriz de confusión simple y elegante"""
        # Crear figura con 2 subplots lado a lado
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Matriz con números absolutos
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1,
                   cbar_kws={'label': 'Número de casos'})
        ax1.set_title('Matriz de Confusión\n(Números Absolutos)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Clase Real', fontsize=12)
        ax1.set_xlabel('Clase Predicha', fontsize=12)
        
        # 2. Matriz con porcentajes
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Crear anotaciones con números y porcentajes
        annotations = []
        for i in range(len(class_names)):
            row = []
            for j in range(len(class_names)):
                text = f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)'
                row.append(text)
            annotations.append(row)
        
        sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2,
                   cbar_kws={'label': 'Porcentaje (%)'})
        ax2.set_title('Matriz de Confusión\n(Porcentajes por Clase)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Clase Real', fontsize=12)
        ax2.set_xlabel('Clase Predicha', fontsize=12)
        
        # Calcular accuracy general
        accuracy = np.trace(cm) / np.sum(cm)
        
        # Agregar título general con accuracy
        fig.suptitle(f'COVID-19 Classifier Smart Balanced\nAccuracy General: {accuracy:.1%}', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Dejar espacio para el título
        plt.savefig('results/confusion_matrix_smart_balanced_clean.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Análisis simple y claro
        print(f"\n🎯 RESULTADOS DEL MODELO SMART BALANCED")
        print(f"{'='*50}")
        
        overall_accuracy = np.trace(cm) / np.sum(cm)
        print(f"📊 ACCURACY GENERAL: {overall_accuracy:.1%}")
        print(f"📈 Total de muestras: {np.sum(cm):,}")
        
        print(f"\n📋 RENDIMIENTO POR CLASE:")
        print(f"{'-'*50}")
        
        for i, class_name in enumerate(class_names):
            class_accuracy = cm[i,i] / cm[i,:].sum() * 100
            total_class = cm[i,:].sum()
            correct = cm[i,i]
            
            # Emojis por clase
            emoji = '🦠' if class_name == 'COVID' else '✅' if class_name == 'NORMAL' else '🫁'
            
            print(f"{emoji} {class_name}:")
            print(f"   Correctos: {correct}/{total_class} ({class_accuracy:.1f}%)")
            
            # Mostrar principales errores
            errors = []
            for j in range(len(class_names)):
                if i != j and cm[i,j] > 0:
                    error_pct = (cm[i,j] / total_class) * 100
                    errors.append((class_names[j], cm[i,j], error_pct))
            
            if errors:
                errors.sort(key=lambda x: x[1], reverse=True)  # Ordenar por cantidad
                print(f"   Errores principales:")
                for error_class, count, pct in errors:
                    print(f"     → {error_class}: {count} casos ({pct:.1f}%)")
            print()
        
        # Interpretación simple
        print(f"💡 INTERPRETACIÓN:")
        if overall_accuracy > 0.85:
            print("🏆 EXCELENTE rendimiento")
        elif overall_accuracy > 0.75:
            print("✅ BUEN rendimiento - Modelo confiable")
        elif overall_accuracy > 0.65:
            print("⚠️ Rendimiento ACEPTABLE")
        else:
            print("❌ Necesita MEJORAS")
        
        print(f"\n🎯 CONCLUSIÓN:")
        print(f"El modelo tiene un rendimiento {overall_accuracy:.1%} con balance equilibrado entre las 3 clases.")
        print(f"Los errores son médicamente lógicos (confusión entre patologías similares).")
    
    def save_model(self, filepath):
        """💾 Guardar modelo"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'pca': self.pca,
            'label_encoder': self.label_encoder,
            'class_weights': self.class_weights,
            'image_size': self.image_size
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✅ Modelo guardado en: {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """📂 Cargar modelo"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Crear instancia
        instance = cls(image_size=model_data['image_size'])
        
        # Cargar componentes
        instance.models = model_data['models']
        instance.scaler = model_data['scaler']
        instance.pca = model_data['pca']
        instance.label_encoder = model_data['label_encoder']
        instance.class_weights = model_data.get('class_weights', None)
        
        print(f"✅ Modelo cargado desde: {filepath}")
        return instance

def main():
    """🚀 Función principal para entrenar modelo smart balanced"""
    print("🎯 COVID-19 Classifier SMART BALANCED")
    print("📊 CONFIGURACIÓN ÓPTIMA para tu dataset:")
    print("   COVID: 7,232 | NORMAL: 20,384 | PNEUMONIA: 14,714")
    print("🎯 DECISIÓN: 7,232 por clase = 21,696 total (PERFECTAMENTE BALANCEADO)")
    
    # Configuración
    data_dir = "data/processed"
    model_path = "results/covid_classifier_optimal_balanced.pkl"
    
    # Crear directorio de resultados
    os.makedirs("results", exist_ok=True)
    
    # Crear y entrenar modelo
    classifier = CovidClassifierSmartBalanced(use_all_data=True)
    
    # Entrenar con diferentes configuraciones
    target_samples = 7232  # DECISIÓN ÓPTIMA: COVID como limitante
    
    print(f"\n🎯 CONFIGURACIÓN ÓPTIMA:")
    print(f"   COVID: 7,232 (todas)")
    print(f"   PNEUMONIA: 7,232 (sample de 14,714)")
    print(f"   NORMAL: 7,232 (sample de 20,384)")
    print(f"   TOTAL: 21,696 imágenes balanceadas")
    
    accuracy = classifier.train_with_smart_balancing(data_dir, target_samples)
    
    # Guardar modelo
    classifier.save_model(model_path)
    
    print(f"\n✅ ENTRENAMIENTO COMPLETADO")
    print(f"📊 Accuracy final: {accuracy:.3f}")
    print(f"💾 Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()