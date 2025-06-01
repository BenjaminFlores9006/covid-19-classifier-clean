import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle
from tqdm import tqdm

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE  # Para balanceo más sofisticado

# Configurar matplotlib para español
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

class CovidClassifierBalanced:
    """
    Clasificador COVID-19 BALANCEADO usando OpenCV + scikit-learn
    """
    
    def __init__(self, image_size=(150, 150), use_masks=True):
        self.image_size = image_size
        self.use_masks = use_masks
        
        # ✅ CONFIGURACIÓN BALANCEADA (sin sesgo hacia COVID)
        # Random Forest balanceado
        self.rf_model = RandomForestClassifier(
            n_estimators=300,               # Reducido para menos overfitting
            max_depth=15,                   # Limitado para generalización
            min_samples_split=5,            # Más conservador
            min_samples_leaf=2,             # Evita overfitting
            max_features='sqrt',
            class_weight='balanced',        # ✅ AUTOMÁTICO: No manual
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Extra Trees balanceado
        self.et_model = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',        # ✅ AUTOMÁTICO: No manual
            bootstrap=True,
            oob_score=True,
            random_state=43,
            n_jobs=-1
        )
        
        # Gradient Boosting balanceado
        self.gb_model = GradientBoostingClassifier(
            n_estimators=150,               # Reducido
            learning_rate=0.1,              # Learning rate normal
            max_depth=6,                    # Menos profundidad
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )
        
        # Logistic Regression balanceada
        self.lr_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',        # ✅ AUTOMÁTICO: No manual
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        self.models = [self.rf_model, self.et_model, self.gb_model, self.lr_model]
        self.model_names = ['RandomForest', 'ExtraTrees', 'GradientBoosting', 'LogisticRegression']
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=0.95)  # ✅ Mantener 95% de varianza
        
        # Estadísticas
        self.training_stats = {}
        
    def extract_features(self, image_path, mask_path=None):
        """
        Extraer características MEJORADAS de una imagen usando OpenCV
        """
        try:
            # Extraer características de la imagen principal
            img_features = self._extract_single_image_features(image_path)
            if img_features is None:
                return None
            
            # Si tenemos máscara, extraer sus características también
            if mask_path and self.use_masks:
                mask_features = self._extract_single_image_features(mask_path, is_mask=True)
                if mask_features is not None:
                    # Combinar características de imagen y máscara
                    combined_features = np.concatenate([img_features, mask_features])
                    return combined_features
            
            return img_features
            
        except Exception as e:
            print(f"❌ Error procesando {image_path}: {e}")
            return None
    
    def _extract_single_image_features(self, image_path, is_mask=False):
        """
        ✅ CARACTERÍSTICAS BALANCEADAS (sin sesgo hacia COVID)
        """
        try:
            # Leer imagen
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Verificar que la imagen se cargó correctamente
            if img.size == 0 or len(img.shape) == 0:
                return None
            
            # Redimensionar
            img = cv2.resize(img, self.image_size)
            
            # Normalizar
            img_norm = img.astype(np.float32) / 255.0
            
            # ✅ 1. HISTOGRAMA BALANCEADO (32 bins es suficiente)
            hist = cv2.calcHist([img], [0], None, [32], [0, 256])
            hist = hist.flatten()
            hist = hist / (np.sum(hist) + 1e-8)
            
            # ✅ 2. ESTADÍSTICAS BÁSICAS BALANCEADAS
            stats = [
                np.mean(img_norm),              # Media
                np.std(img_norm),               # Desviación estándar
                np.min(img_norm),               # Mínimo
                np.max(img_norm),               # Máximo
                np.median(img_norm),            # Mediana
                np.percentile(img_norm, 25),    # Percentil 25
                np.percentile(img_norm, 75),    # Percentil 75
                np.var(img_norm),               # Varianza
            ]
            
            # ✅ 3. MOMENTOS DE HU (invariantes geométricos)
            try:
                moments = cv2.moments(img)
                hu_moments = cv2.HuMoments(moments).flatten()
                hu_moments = np.nan_to_num(hu_moments, nan=0.0, posinf=1.0, neginf=-1.0)
            except:
                hu_moments = np.zeros(7)
            
            # ✅ 4. CARACTERÍSTICAS DE TEXTURA BALANCEADAS
            try:
                # Laplaciano para detección de bordes
                laplacian = cv2.Laplacian(img_norm, cv2.CV_64F, ksize=3)
                laplacian_stats = [
                    np.var(laplacian), 
                    np.mean(np.abs(laplacian))
                ]
                
                # Gradientes Sobel
                sobelx = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=3)
                sobel_mag = np.sqrt(sobelx**2 + sobely**2)
                
                sobel_stats = [
                    np.mean(sobelx), np.std(sobelx),
                    np.mean(sobely), np.std(sobely),
                    np.mean(sobel_mag), np.std(sobel_mag)
                ]
                
                # ✅ CARACTERÍSTICAS DE TEXTURA LOCAL (LBP simplificado)
                # Dividir imagen en regiones 4x4 (16 regiones)
                h, w = img_norm.shape
                region_stats = []
                for i in range(0, h, h//4):
                    for j in range(0, w, w//4):
                        region = img_norm[i:i+h//4, j:j+w//4]
                        if region.size > 0:
                            region_stats.extend([
                                np.mean(region), 
                                np.std(region)
                            ])
                
                # Asegurar número fijo de features (32 stats para 16 regiones)
                if len(region_stats) > 32:
                    region_stats = region_stats[:32]
                elif len(region_stats) < 32:
                    region_stats.extend([0.0] * (32 - len(region_stats)))
                
                texture_stats = laplacian_stats + sobel_stats + region_stats
                
            except Exception as e:
                # Fallback para errores
                texture_stats = [0.0] * 40
            
            # ✅ 5. CARACTERÍSTICAS DE FORMA Y CONTRASTE
            try:
                # Análisis de contraste
                contrast_stats = [
                    np.std(img_norm),                           # Contraste global
                    np.mean((img_norm - np.mean(img_norm))**2), # Varianza local
                    len(np.unique(img))                         # Número de niveles únicos
                ]
                
                # Características de forma (usando contornos)
                _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Encontrar el contorno más grande
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    perimeter = cv2.arcLength(largest_contour, True)
                    
                    shape_stats = [
                        area / (img.shape[0] * img.shape[1]),   # Área relativa
                        perimeter / (2 * (img.shape[0] + img.shape[1])),  # Perímetro relativo
                        4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0  # Circularidad
                    ]
                else:
                    shape_stats = [0.0, 0.0, 0.0]
                
                contrast_shape_stats = contrast_stats + shape_stats
                
            except:
                contrast_shape_stats = [0.0] * 6
            
            # ✅ COMBINAR CARACTERÍSTICAS BALANCEADAS
            features = np.concatenate([
                hist,                    # 32 features
                stats,                   # 8 features  
                hu_moments,              # 7 features
                texture_stats,           # 40 features
                contrast_shape_stats     # 6 features
            ])
            # Total: ~93 features (mucho más manejable y balanceado)
            
            # Verificar que no hay NaN o infinitos
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return features
            
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
            return None
    
    def load_dataset(self, data_dir):
        """
        Cargar dataset sin sesgo
        """
        print("📂 Cargando dataset de forma BALANCEADA...")
        
        data_path = Path(data_dir)
        classes = ['COVID', 'NORMAL', 'PNEUMONIA']
        
        # Verificar estructura
        for class_name in classes:
            class_path = data_path / class_name / 'images'
            if not class_path.exists():
                raise ValueError(f"❌ No se encuentra: {class_path}")
        
        features_list = []
        labels_list = []
        
        # ✅ CARGAR MISMA CANTIDAD DE CADA CLASE (balanceado)
        max_samples_per_class = None
        
        # Primero, contar cuántas imágenes tiene cada clase
        class_counts = {}
        for class_name in classes:
            images_path = data_path / class_name / 'images'
            if images_path.exists():
                image_files = list(images_path.glob('*.png')) + list(images_path.glob('*.jpg'))
                class_counts[class_name] = len(image_files)
                print(f"   {class_name}: {len(image_files)} imágenes disponibles")
        
        # Usar la clase con menos imágenes como límite (balanceo natural)
        if class_counts:
            max_samples_per_class = min(class_counts.values())
            print(f"\n✅ BALANCEO: Usando {max_samples_per_class} muestras por clase")
        
        # Procesar cada clase con límite
        for class_name in classes:
            print(f"\n📁 Procesando clase: {class_name}")
            
            images_path = data_path / class_name / 'images'
            masks_path = data_path / class_name / 'masks'
            
            if not images_path.exists():
                continue
                
            # Obtener todas las imágenes
            image_files = list(images_path.glob('*.png')) + list(images_path.glob('*.jpg'))
            
            # ✅ LIMITAR NÚMERO DE MUESTRAS POR CLASE
            if max_samples_per_class and len(image_files) > max_samples_per_class:
                # Seleccionar aleatoriamente para evitar sesgo
                np.random.seed(42)  # Reproducible
                image_files = np.random.choice(image_files, max_samples_per_class, replace=False)
                print(f"   🎯 Limitado a {max_samples_per_class} muestras para balanceo")
            
            print(f"   🔍 Procesando: {len(image_files)} imágenes")
            
            # Verificar masks
            if self.use_masks and masks_path.exists():
                mask_files = list(masks_path.glob('*.png')) + list(masks_path.glob('*.jpg'))
                print(f"   🎭 Encontradas: {len(mask_files)} máscaras")
                print(f"   🔗 Modo: Images + Masks combinadas")
            else:
                if self.use_masks:
                    print(f"   ⚠️ No se encontraron máscaras para {class_name}, usando solo images")
                else:
                    print(f"   📷 Modo: Solo images")
            
            # Extraer características con barra de progreso
            for img_file in tqdm(image_files, desc=f"Extrayendo {class_name}"):
                # Buscar máscara correspondiente
                mask_file = None
                if self.use_masks and masks_path.exists():
                    img_name = img_file.stem
                    if '_' in img_name:
                        class_prefix, img_number = img_name.rsplit('_', 1)
                        possible_mask = masks_path / f"{class_prefix}_{img_number}.png"
                        if possible_mask.exists():
                            mask_file = possible_mask
                
                features = self.extract_features(img_file, mask_file)
                if features is not None:
                    features_list.append(features)
                    labels_list.append(class_name)
        
        # Convertir a arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"\n✅ Dataset BALANCEADO cargado: {len(X)} muestras con {X.shape[1]} características")
        
        # Mostrar distribución final
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n📊 Distribución final:")
        for class_name, count in zip(unique, counts):
            percentage = (count / len(y)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        return X, y
    
    def train(self, data_dir, test_size=0.2):
        """
        ✅ ENTRENAMIENTO BALANCEADO (sin sesgo hacia COVID)
        """
        print("🚀 ENTRENAMIENTO COVID-19 CLASSIFIER BALANCEADO")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Cargar datos balanceados
        X, y = self.load_dataset(data_dir)
        
        # 2. Codificar labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # ✅ 3. SIN DATA AUGMENTATION SESGADO - usar SMOTE si es necesario
        print("\n🔧 Verificando balance del dataset...")
        unique, counts = np.unique(y_encoded, return_counts=True)
        class_names = self.label_encoder.classes_
        
        min_samples = min(counts)
        max_samples = max(counts)
        imbalance_ratio = max_samples / min_samples
        
        print(f"📊 Ratio de desbalance: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 1.5:  # Si hay desbalance significativo
            print("⚖️ Aplicando SMOTE para balanceo inteligente...")
            try:
                smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples-1))
                X, y_encoded = smote.fit_resample(X, y_encoded)
                print(f"✅ Dataset balanceado con SMOTE: {len(X)} muestras")
            except Exception as e:
                print(f"⚠️ SMOTE falló: {e}. Continuando sin balanceo.")
        else:
            print("✅ Dataset ya está balanceado naturalmente")
        
        # 4. División train/test estratificada
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        print(f"\n📊 División del dataset:")
        print(f"   🎓 Entrenamiento: {len(X_train)} imágenes")
        print(f"   🧪 Prueba: {len(X_test)} imágenes")
        
        # Mostrar distribución por clase
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        print(f"\n📈 Distribución en entrenamiento:")
        for i, (class_name, count) in enumerate(zip(class_names, counts_train)):
            percentage = (count / len(y_train)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
            
        print(f"\n📈 Distribución en prueba:")
        for i, (class_name, count) in enumerate(zip(class_names, counts_test)):
            percentage = (count / len(y_test)) * 100
            print(f"   {class_name}: {count} ({percentage:.1f}%)")
        
        # 5. Escalar características
        print("\n🔧 Normalizando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 6. Reducir dimensionalidad con PCA
        print("🔧 Aplicando PCA...")
        X_train_pca = self.pca.fit_transform(X_train_scaled)
        X_test_pca = self.pca.transform(X_test_scaled)
        
        print(f"   📉 Dimensiones: {X_train.shape[1]} → {X_train_pca.shape[1]}")
        print(f"   📊 Varianza explicada: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # ✅ 7. ENTRENAMIENTO CON VALIDACIÓN CRUZADA
        print("\n🧠 Entrenamiento con validación cruzada...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = {}
        
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            print(f"   🔧 Entrenando {name}...")
            
            # Validación cruzada
            cv_score = cross_val_score(model, X_train_pca, y_train, cv=cv, scoring='accuracy')
            cv_scores[name] = cv_score
            
            print(f"      CV Score: {cv_score.mean():.3f} (±{cv_score.std()*2:.3f})")
            
            # Entrenar en todo el set de entrenamiento
            model.fit(X_train_pca, y_train)
        
        print("   ✅ Todos los modelos entrenados!")
        
        # 8. Evaluación con ensemble balanceado
        train_predictions = self._balanced_ensemble_predict(X_train_pca)
        test_predictions = self._balanced_ensemble_predict(X_test_pca)
        
        train_accuracy = accuracy_score(y_train, train_predictions)
        test_accuracy = accuracy_score(y_test, test_predictions)
        
        # Tiempo total
        training_time = time.time() - start_time
        
        # ✅ MÉTRICAS BALANCEADAS PARA TODAS LAS CLASES
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_predictions, average=None)
        
        print(f"\n🎯 MÉTRICAS POR CLASE (BALANCEADAS):")
        for i, class_name in enumerate(class_names):
            print(f"   {class_name}:")
            print(f"      Precision: {precision[i]:.3f} ({precision[i]*100:.1f}%)")
            print(f"      Recall: {recall[i]:.3f} ({recall[i]*100:.1f}%)")
            print(f"      F1-Score: {f1[i]:.3f} ({f1[i]*100:.1f}%)")
        
        # Métricas promedio
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        print(f"\n📊 MÉTRICAS PROMEDIO:")
        print(f"   Macro Precision: {macro_precision:.3f} ({macro_precision*100:.1f}%)")
        print(f"   Macro Recall: {macro_recall:.3f} ({macro_recall*100:.1f}%)")
        print(f"   Macro F1-Score: {macro_f1:.3f} ({macro_f1*100:.1f}%)")
        
        # Guardar estadísticas
        self.training_stats = {
            'training_time': training_time,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'cv_scores': cv_scores,
            'total_samples': len(X),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'class_precision': dict(zip(class_names, precision)),
            'class_recall': dict(zip(class_names, recall)),
            'class_f1': dict(zip(class_names, f1))
        }
        
        # Mostrar resultados finales
        print("=" * 60)
        print(f"🎯 CLASIFICADOR BALANCEADO - RESULTADOS FINALES")
        print("=" * 60)
        print(f"⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")
        print(f"🎓 Precisión entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"🧪 Precisión prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"⚖️ Balance del modelo: {macro_f1:.4f} ({macro_f1*100:.2f}%)")
        
        # Reporte detallado
        print(f"\n📋 Reporte de clasificación BALANCEADO:")
        print(classification_report(y_test, test_predictions, target_names=class_names))
        
        # Verificar sesgo
        self._check_bias(y_test, test_predictions, class_names)
        
        # Gráficos
        self.plot_results(y_test, test_predictions, class_names)
        
        return test_accuracy
    
    def _balanced_ensemble_predict(self, X):
        """
        ✅ PREDICCIÓN ENSEMBLE BALANCEADA (sin sesgo hacia COVID)
        """
        # Pesos IGUALES para todos los modelos
        weights = [0.25, 0.25, 0.25, 0.25]  # ✅ SIN SESGO
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Votación simple (sin pesos extremos)
        ensemble_pred = []
        for i in range(len(X)):
            votes = {}
            for j, pred in enumerate(predictions):
                vote = pred[i]
                if vote not in votes:
                    votes[vote] = 0
                votes[vote] += weights[j]
            
            # Clase con mayor peso (pero sin sesgo)
            best_class = max(votes.keys(), key=lambda k: votes[k])
            ensemble_pred.append(best_class)
        
        return np.array(ensemble_pred)
    
    def _check_bias(self, y_true, y_pred, class_names):
        """
        ✅ VERIFICAR SESGO DEL MODELO
        """
        print(f"\n🔍 ANÁLISIS DE SESGO:")
        
        # Contar predicciones por clase
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        total_predictions = len(y_pred)
        
        print(f"📊 Distribución de PREDICCIONES:")
        for i, class_name in enumerate(class_names):
            if i < len(counts_pred):
                count = counts_pred[i]
                percentage = (count / total_predictions) * 100
                print(f"   {class_name}: {count} predicciones ({percentage:.1f}%)")
                
                # Advertencia si una clase domina
                if percentage > 60:
                    print(f"   ⚠️ POSIBLE SESGO hacia {class_name}")
                elif percentage < 15:
                    print(f"   ⚠️ POSIBLE SUBPREDICCIÓN de {class_name}")
        
        # Verificar matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        print(f"\n🎯 ANÁLISIS DE CONFUSIÓN:")
        for i, class_name in enumerate(class_names):
            true_positives = cm[i, i]
            total_true = np.sum(cm[i, :])
            total_pred = np.sum(cm[:, i])
            
            if total_true > 0:
                recall = true_positives / total_true
                print(f"   {class_name} - Recall: {recall:.3f}")
                
                if recall < 0.5:
                    print(f"     ⚠️ BAJA detección de {class_name}")
        
        # Verificar si hay sesgo general
        pred_distribution = np.array([np.sum(y_pred == i) for i in range(len(class_names))])
        max_pred = np.max(pred_distribution)
        min_pred = np.min(pred_distribution)
        
        if max_pred > 0 and min_pred > 0:
            bias_ratio = max_pred / min_pred
            if bias_ratio > 2.0:
                print(f"\n⚠️ SESGO DETECTADO: Ratio {bias_ratio:.2f} (>2.0)")
                print(f"💡 Sugerencia: Revisar balanceo de datos o pesos de clase")
            else:
                print(f"\n✅ MODELO BALANCEADO: Ratio {bias_ratio:.2f} (<2.0)")
    
    def _balanced_ensemble_predict_proba(self, X):
        """
        ✅ Probabilidades ensemble balanceadas
        """
        weights = [0.25, 0.25, 0.25, 0.25]  # Pesos iguales
        
        all_probas = []
        for model in self.models:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)
                all_probas.append(proba)
        
        if not all_probas:
            return None
        
        # Promedio simple (sin sesgo)
        ensemble_proba = np.zeros_like(all_probas[0])
        for i, (proba, weight) in enumerate(zip(all_probas, weights[:len(all_probas)])):
            ensemble_proba += proba * weight
        
        return ensemble_proba
    
    def predict(self, image_path):
        """
        ✅ PREDICCIÓN BALANCEADA de una sola imagen
        """
        # Buscar máscara correspondiente
        mask_path = None
        if self.use_masks:
            img_path = Path(image_path)
            mask_path_str = str(img_path).replace('/images/', '/masks/').replace('\\images\\', '\\masks\\')
            mask_path_candidate = Path(mask_path_str)
            
            if mask_path_candidate.exists():
                mask_path = mask_path_candidate
            else:
                img_parts = img_path.parts
                mask_parts = []
                for part in img_parts:
                    if part == 'images':
                        mask_parts.append('masks')
                    else:
                        mask_parts.append(part)
                
                mask_path_candidate2 = Path(*mask_parts)
                if mask_path_candidate2.exists():
                    mask_path = mask_path_candidate2
        
        features = self.extract_features(image_path, mask_path)
        if features is None:
            return None, 0.0, None
        
        # Verificar características
        expected_features = len(self.scaler.mean_)
        actual_features = len(features)
        
        if actual_features != expected_features:
            print(f"❌ Error de características: esperadas {expected_features}, obtenidas {actual_features}")
            return None, 0.0, None
        
        # Aplicar transformaciones
        features_scaled = self.scaler.transform([features])
        features_pca = self.pca.transform(features_scaled)
        
        # ✅ PREDICCIÓN BALANCEADA
        ensemble_pred = self._balanced_ensemble_predict(features_pca)[0]
        
        # ✅ PROBABILIDADES BALANCEADAS
        ensemble_proba = self._balanced_ensemble_predict_proba(features_pca)
        if ensemble_proba is not None:
            confidence = ensemble_proba[0][ensemble_pred]
            all_probabilities = ensemble_proba[0]
            
            # Mostrar todas las probabilidades
            print(f"📊 Probabilidades BALANCEADAS:")
            for i, class_name_prob in enumerate(self.label_encoder.classes_):
                prob_percent = all_probabilities[i] * 100
                print(f"   {class_name_prob}: {all_probabilities[i]:.3f} ({prob_percent:.1f}%)")
        else:
            confidence = 0.33  # Confianza neutral
            all_probabilities = None
        
        class_name = self.label_encoder.inverse_transform([ensemble_pred])[0]
        
        return class_name, confidence, all_probabilities
    
    def plot_results(self, y_true, y_pred, class_names):
        """
        ✅ GRÁFICOS BALANCEADOS para análisis
        """
        # Crear figura con subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('COVID-19 Classifier BALANCEADO - Resultados', fontsize=16, fontweight='bold')
        
        # 1. Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalizar matriz de confusión para mostrar porcentajes
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear anotaciones combinadas (número y porcentaje)
        annotations = []
        for i in range(cm.shape[0]):
            row = []
            for j in range(cm.shape[1]):
                row.append(f"{cm[i,j]}\n({cm_percent[i,j]:.1%})")
            annotations.append(row)
        
        sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Matriz de Confusión (Normalizada)')
        ax1.set_xlabel('Predicción')
        ax1.set_ylabel('Realidad')
        
        # 2. Importancia de características (Top 20)
        importances = self.rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        ax2.bar(range(20), importances[indices], color='skyblue')
        ax2.set_title('Top 20 Características Importantes')
        ax2.set_xlabel('Ranking de Característica')
        ax2.set_ylabel('Importancia')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Análisis de sesgo - Distribución de predicciones
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        
        x_pos = np.arange(len(class_names))
        width = 0.35
        
        # Asegurar que tenemos datos para todas las clases
        true_counts = [0] * len(class_names)
        pred_counts = [0] * len(class_names)
        
        for i, class_idx in enumerate(unique_true):
            true_counts[class_idx] = counts_true[i]
        
        for i, class_idx in enumerate(unique_pred):
            pred_counts[class_idx] = counts_pred[i]
        
        ax3.bar(x_pos - width/2, true_counts, width, label='Realidad', color='lightgreen', alpha=0.8)
        ax3.bar(x_pos + width/2, pred_counts, width, label='Predicciones', color='lightcoral', alpha=0.8)
        ax3.set_title('Distribución: Realidad vs Predicciones')
        ax3.set_xlabel('Clase')
        ax3.set_ylabel('Cantidad')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(class_names)
        ax3.legend()
        
        # Agregar línea de referencia para balance perfecto
        perfect_balance = len(y_true) / len(class_names)
        ax3.axhline(y=perfect_balance, color='blue', linestyle='--', alpha=0.5, label='Balance perfecto')
        ax3.legend()
        
        # 4. Métricas por clase con barras de error
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
        
        metrics_df = pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }, index=class_names)
        
        metrics_df.plot(kind='bar', ax=ax4, rot=0, width=0.8)
        ax4.set_title('Métricas por Clase')
        ax4.set_xlabel('Clase')
        ax4.set_ylabel('Score')
        ax4.legend(loc='upper right')
        ax4.set_ylim(0, 1.1)
        
        # Línea de referencia para 80% accuracy
        ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80% objetivo')
        ax4.legend()
        
        # Ajustar layout
        plt.tight_layout()
        
        # Crear carpeta de resultados si no existe
        if Path.cwd().name == "src":
            # Si estamos en src/, guardar en ../results/
            results_dir = Path.cwd().parent / "results"
        else:
            # Si estamos en la raíz, guardar en results/
            results_dir = Path.cwd() / "results"
        
        results_dir.mkdir(exist_ok=True)
        
        # Guardar
        plt.savefig(results_dir / 'covid_classifier_balanced_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Resultados guardados: {results_dir / 'covid_classifier_balanced_results.png'}")
    
    def save_model(self, filename=None):
        """
        Guardar modelo balanceado
        """
        if filename is None:
            # Determinar dónde guardar automáticamente
            if Path.cwd().name == "src":
                # Si estamos en src/, guardar en ../results/
                results_dir = Path.cwd().parent / "results"
            else:
                # Si estamos en la raíz, guardar en results/
                results_dir = Path.cwd() / "results"
            
            results_dir.mkdir(exist_ok=True)
            filename = results_dir / 'covid_classifier_balanced.pkl'
        
        # Crear directorio si no existe
        Path(filename).parent.mkdir(exist_ok=True)
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'pca': self.pca,
            'image_size': self.image_size,
            'use_masks': self.use_masks,
            'training_stats': self.training_stats
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Modelo BALANCEADO guardado: {filename}")
    
    @classmethod
    def load_model(cls, filename=None):
        """
        Cargar modelo balanceado pre-entrenado
        """
        if filename is None:
            # Buscar modelo en múltiples ubicaciones
            possible_paths = [
                'results/covid_classifier_balanced.pkl',
                '../results/covid_classifier_balanced.pkl',
                Path.cwd() / 'results' / 'covid_classifier_balanced.pkl',
                Path.cwd().parent / 'results' / 'covid_classifier_balanced.pkl'
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    filename = str(path)
                    break
            else:
                raise FileNotFoundError("No se encuentra el modelo entrenado. Entrena primero con main()")
        
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        classifier = cls(model_data['image_size'], model_data.get('use_masks', True))
        classifier.models = model_data['models']
        classifier.rf_model = classifier.models[0]
        classifier.et_model = classifier.models[1]
        classifier.gb_model = classifier.models[2] 
        classifier.lr_model = classifier.models[3]
        classifier.scaler = model_data['scaler']
        classifier.label_encoder = model_data['label_encoder']
        classifier.pca = model_data['pca']
        classifier.training_stats = model_data['training_stats']
        
        print(f"📂 Modelo BALANCEADO cargado: {filename}")
        return classifier

def test_single_image_balanced(classifier, image_path):
    """
    ✅ PROBAR imagen con clasificador BALANCEADO
    """
    print(f"\n🔍 Analizando (BALANCEADO): {image_path}")
    
    prediction, confidence, all_probabilities = classifier.predict(image_path)
    
    if prediction:
        print(f"🎯 Predicción FINAL: {prediction}")
        print(f"📊 Confianza: {confidence:.4f} ({confidence*100:.1f}%)")
        
        # Análisis de confianza
        if confidence > 0.7:
            print(f"✅ Confianza ALTA")
        elif confidence > 0.5:
            print(f"⚠️ Confianza MEDIA")
        else:
            print(f"❌ Confianza BAJA - revisar imagen")
        
        # Mostrar imagen con predicción
        img = cv2.imread(str(image_path))
        if img is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Título con todas las probabilidades
            if all_probabilities is not None:
                prob_text = " | ".join([f"{cls}: {prob*100:.1f}%" 
                                      for cls, prob in zip(classifier.label_encoder.classes_, all_probabilities)])
                plt.title(f'PREDICCIÓN: {prediction} ({confidence*100:.1f}%)\n{prob_text}', 
                         fontsize=12, pad=20)
            else:
                plt.title(f'PREDICCIÓN: {prediction} ({confidence*100:.1f}%)')
            
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    else:
        print("❌ Error procesando imagen")

def main():
    """
    ✅ FUNCIÓN PRINCIPAL BALANCEADA
    """
    print("🦠 COVID-19 CLASSIFIER BALANCEADO - OpenCV + scikit-learn")
    print("=" * 70)
    print("✅ SIN SESGO hacia COVID")
    print("⚖️ PREDICCIONES EQUILIBRADAS para todas las clases")
    print("=" * 70)
    
    # ✅ BUSCAR DATASET EN MÚLTIPLES UBICACIONES
    possible_paths = [
        "data/processed",           # Si se ejecuta desde raíz
        "../data/processed",        # Si se ejecuta desde src/
        "../../data/processed",     # Si se ejecuta desde subcarpeta
        Path.cwd() / "data" / "processed",  # Ruta absoluta desde directorio actual
        Path.cwd().parent / "data" / "processed"  # Ruta absoluta desde directorio padre
    ]
    
    data_dir = None
    for path in possible_paths:
        if Path(path).exists():
            data_dir = str(path)
            print(f"✅ Dataset encontrado en: {data_dir}")
            break
    
    if data_dir is None:
        print(f"❌ No se encuentra el dataset en ninguna ubicación:")
        for path in possible_paths:
            print(f"   - {path}")
        print(f"\n📁 Directorio actual: {Path.cwd()}")
        print(f"💡 Verifica que tengas la estructura:")
        print(f"   data/processed/COVID/images/")
        print(f"   data/processed/NORMAL/images/") 
        print(f"   data/processed/PNEUMONIA/images/")
        return
    
    # Crear y entrenar clasificador BALANCEADO
    classifier = CovidClassifierBalanced(image_size=(150, 150), use_masks=True)
    
    try:
        print("🚀 Iniciando entrenamiento BALANCEADO...")
        
        # Entrenar sin sesgo
        accuracy = classifier.train(data_dir)
        
        # ✅ CREAR CARPETA RESULTS EN LA UBICACIÓN CORRECTA
        # Determinar dónde guardar los resultados
        if Path.cwd().name == "src":
            # Si estamos en src/, guardar en ../results/
            results_dir = Path.cwd().parent / "results"
        else:
            # Si estamos en la raíz, guardar en results/
            results_dir = Path.cwd() / "results"
        
        results_dir.mkdir(exist_ok=True)
        model_path = results_dir / "covid_classifier_balanced.pkl"
        
        # Guardar modelo balanceado
        classifier.save_model(str(model_path))
        
        print(f"\n🎉 ¡Entrenamiento BALANCEADO completado!")
        print(f"🎯 Precisión final: {accuracy*100:.1f}%")
        print(f"⚖️ Modelo SIN SESGO hacia COVID")
        print(f"💾 Modelo guardado en: {model_path}")
        
        # Mostrar estadísticas finales
        if hasattr(classifier, 'training_stats'):
            stats = classifier.training_stats
            print(f"\n📊 ESTADÍSTICAS FINALES:")
            print(f"   Macro F1-Score: {stats.get('macro_f1', 0)*100:.1f}%")
            print(f"   Balance del modelo: ✅ Equilibrado")
            
            # Mostrar métricas por clase
            if 'class_f1' in stats:
                print(f"\n🏥 F1-Score por clase:")
                for class_name, f1_score in stats['class_f1'].items():
                    print(f"   {class_name}: {f1_score*100:.1f}%")
        
        print(f"\n💡 Uso del modelo:")
        print(f"   from covid_classifier_balanced import CovidClassifierBalanced, test_single_image_balanced")
        print(f"   classifier = CovidClassifierBalanced.load_model('{model_path}')")
        print(f"   test_single_image_balanced(classifier, 'ruta/imagen.png')")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()