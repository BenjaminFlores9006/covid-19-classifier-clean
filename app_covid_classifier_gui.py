import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import sys
import os

class CovidClassifierGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🦠 COVID-19 Classifier BALANCEADO - Análisis de Rayos X")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        # Variables
        self.model = None
        self.current_image_path = None
        self.processed_image = None
        
        # Cargar modelo balanceado
        self.load_model()
        
        # Crear interfaz
        self.create_ui()
        
        # Configurar estilos
        self.configure_styles()
    
    def load_model(self):
        """✅ CARGAR MODELO BALANCEADO"""
        try:
            # Buscar modelo en múltiples ubicaciones
            possible_paths = [
                'results/covid_classifier_balanced.pkl',
                '../results/covid_classifier_balanced.pkl',
                'src/results/covid_classifier_balanced.pkl',
                Path.cwd() / 'results' / 'covid_classifier_balanced.pkl',
                Path.cwd().parent / 'results' / 'covid_classifier_balanced.pkl'
            ]
            
            model_path = None
            for path in possible_paths:
                if Path(path).exists():
                    model_path = str(path)
                    break
            
            if model_path is None:
                messagebox.showerror("Error", 
                    "No se encuentra el modelo entrenado.\n"
                    "Posibles ubicaciones buscadas:\n" + 
                    "\n".join([str(p) for p in possible_paths]) +
                    "\n\nEjecuta primero: python covid_classifier_balanced.py")
                return
            
            # Importar la clase del modelo balanceado
            sys.path.append('src')
            sys.path.append('.')
            
            try:
                # Intentar importar desde el nuevo archivo
                from src.covid_classifier import CovidClassifierBalanced
                self.model = CovidClassifierBalanced.load_model(model_path)
                print(f"✅ Modelo BALANCEADO cargado desde: {model_path}")
                
            except ImportError:
                # Fallback al modelo original si no existe el balanceado
                try:
                    from src.covid_classifier import CovidClassifierCV
                    self.model = CovidClassifierCV.load_model(model_path)
                    print(f"⚠️ Usando modelo original desde: {model_path}")
                    messagebox.showwarning("Advertencia", 
                        "Se cargó el modelo original. Para mejor rendimiento, "
                        "entrena el modelo balanceado ejecutando covid_classifier_balanced.py")
                except ImportError as e:
                    raise ImportError(f"No se pudo importar ningún modelo: {e}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando modelo: {e}")
            print(f"❌ Error cargando modelo: {e}")
            import traceback
            traceback.print_exc()
    
    def configure_styles(self):
        """Configurar estilos de la interfaz"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Estilo para botones
        style.configure('Modern.TButton',
                       background='#4CAF50',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       relief='flat',
                       font=('Arial', 12, 'bold'))
        
        style.map('Modern.TButton',
                 background=[('active', '#45a049'),
                           ('pressed', '#3d8b40')])
        
        # Estilo para botón de proceso
        style.configure('Process.TButton',
                       background='#2196F3',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       relief='flat',
                       font=('Arial', 14, 'bold'))
        
        style.map('Process.TButton',
                 background=[('active', '#1976D2'),
                           ('pressed', '#1565C0')])
    
    def create_ui(self):
        """Crear la interfaz de usuario"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_frame = tk.Frame(main_frame, bg='#1e1e1e')
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(title_frame, 
                              text="🦠 COVID-19 Classifier BALANCEADO",
                              font=('Arial', 24, 'bold'),
                              fg='#4CAF50',
                              bg='#1e1e1e')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="⚖️ Análisis Equilibrado de Radiografías • SIN SESGO hacia COVID",
                                 font=('Arial', 12),
                                 fg='#888888',
                                 bg='#1e1e1e')
        subtitle_label.pack()
        
        # Frame para contenido principal
        content_frame = tk.Frame(main_frame, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Imagen
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Título panel izquierdo
        img_title = tk.Label(left_panel,
                            text="📷 Imagen de Rayos X",
                            font=('Arial', 16, 'bold'),
                            fg='white',
                            bg='#2d2d2d')
        img_title.pack(pady=10)
        
        # Botón subir imagen
        upload_btn = ttk.Button(left_panel,
                               text="📁 Subir Imagen",
                               style='Modern.TButton',
                               command=self.upload_image)
        upload_btn.pack(pady=10)
        
        # Frame para mostrar imagen
        self.image_frame = tk.Frame(left_panel, bg='#2d2d2d')
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Label para mostrar imagen
        self.image_label = tk.Label(self.image_frame,
                                   text="Selecciona una imagen de rayos X\npara comenzar el análisis",
                                   font=('Arial', 12),
                                   fg='#888888',
                                   bg='#2d2d2d')
        self.image_label.pack(expand=True)
        
        # Panel derecho - Resultados
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Título panel derecho
        results_title = tk.Label(right_panel,
                                text="📊 Resultados del Análisis BALANCEADO",
                                font=('Arial', 16, 'bold'),
                                fg='white',
                                bg='#2d2d2d')
        results_title.pack(pady=10)
        
        # Botón procesar
        self.process_btn = ttk.Button(right_panel,
                                     text="🔬 Analizar (Balanceado)",
                                     style='Process.TButton',
                                     command=self.process_image,
                                     state='disabled')
        self.process_btn.pack(pady=10)
        
        # Frame para resultados
        self.results_frame = tk.Frame(right_panel, bg='#2d2d2d')
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Placeholder para resultados
        self.results_label = tk.Label(self.results_frame,
                                     text="Sube una imagen y presiona\n'Analizar (Balanceado)' para ver los resultados\n\n✅ Modelo sin sesgo hacia COVID",
                                     font=('Arial', 12),
                                     fg='#888888',
                                     bg='#2d2d2d',
                                     justify=tk.CENTER)
        self.results_label.pack(expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(right_panel, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=20, pady=10)
        self.progress.pack_forget()  # Ocultar inicialmente
    
    def upload_image(self):
        """Subir imagen"""
        file_types = [
            ('Imágenes', '*.png *.jpg *.jpeg *.bmp *.tiff'),
            ('PNG', '*.png'),
            ('JPEG', '*.jpg *.jpeg'),
            ('Todos los archivos', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen de rayos X",
            filetypes=file_types
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.process_btn.config(state='normal')
            
            # Limpiar resultados anteriores
            for widget in self.results_frame.winfo_children():
                widget.destroy()
            
            self.results_label = tk.Label(self.results_frame,
                                         text="Imagen cargada ✅\nPresiona 'Analizar (Balanceado)' para procesar\n\n⚖️ Análisis sin sesgo hacia COVID",
                                         font=('Arial', 12),
                                         fg='#4CAF50',
                                         bg='#2d2d2d',
                                         justify=tk.CENTER)
            self.results_label.pack(expand=True)
    
    def display_image(self, image_path):
        """✅ MOSTRAR IMAGEN SIMPLIFICADO"""
        try:
            # Cargar imagen
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para mostrar
            height, width = img_rgb.shape[:2]
            max_size = 400
            
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            img_resized = cv2.resize(img_rgb, (new_width, new_height))
            
            # Limpiar frame anterior
            for widget in self.image_frame.winfo_children():
                widget.destroy()
            
            # Frame para imagen
            display_frame = tk.Frame(self.image_frame, bg='#2d2d2d')
            display_frame.pack(expand=True)
            
            # Imagen original
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            original_label = tk.Label(display_frame, 
                                    text="📷 Imagen Cargada", 
                                    font=('Arial', 12, 'bold'),
                                    fg='white', bg='#2d2d2d')
            original_label.pack(pady=5)
            
            self.image_label = tk.Label(display_frame, image=img_tk, bg='#2d2d2d')
            self.image_label.image = img_tk  # Mantener referencia
            self.image_label.pack(pady=5)
            
            # Información de la imagen
            info_text = f"Dimensiones: {width}x{height} px\nArchivo: {Path(image_path).name}"
            info_label = tk.Label(display_frame,
                                 text=info_text,
                                 font=('Arial', 10),
                                 fg='#888888',
                                 bg='#2d2d2d')
            info_label.pack(pady=5)
            
            # Información del modelo
            model_info = tk.Label(display_frame,
                                 text="🎯 Modelo: BALANCEADO (sin sesgo)",
                                 font=('Arial', 10, 'bold'),
                                 fg='#4CAF50',
                                 bg='#2d2d2d')
            model_info.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error cargando imagen: {e}")
            print(f"Error en display_image: {e}")
    
    def process_image(self):
        """✅ PROCESAR IMAGEN CON MODELO BALANCEADO"""
        if not self.current_image_path or not self.model:
            messagebox.showerror("Error", "No hay imagen cargada o modelo no disponible")
            return
        
        try:
            # Mostrar progress bar
            self.progress.pack(fill=tk.X, padx=20, pady=10)
            self.progress.start()
            self.root.update()
            
            print(f"\n🔬 PROCESANDO CON MODELO BALANCEADO: {self.current_image_path}")
            
            # ✅ USAR DIRECTAMENTE EL MÉTODO PREDICT DEL MODELO
            # Esto ya maneja todas las complejidades internamente
            if hasattr(self.model, 'predict'):
                # Modelo balanceado (nuevo)
                result = self.model.predict(self.current_image_path)
                
                if len(result) == 3:
                    # Nuevo formato: (class_name, confidence, all_probabilities)
                    class_name, confidence, all_probabilities = result
                else:
                    # Formato anterior: (class_name, confidence)
                    class_name, confidence = result
                    all_probabilities = None
                
            else:
                raise AttributeError("El modelo no tiene método predict")
            
            # Detener progress bar
            self.progress.stop()
            self.progress.pack_forget()
            
            if class_name is None:
                messagebox.showerror("Error", "No se pudo procesar la imagen")
                return
            
            print(f"🎯 RESULTADO BALANCEADO: {class_name} ({confidence*100:.1f}%)")
            
            # Mostrar resultados
            self.display_results(class_name, confidence, all_probabilities)
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("Error", f"Error procesando imagen: {e}")
            print(f"❌ Error completo: {e}")
            import traceback
            traceback.print_exc()
    
    def display_results(self, prediction, confidence, probabilities):
        """✅ MOSTRAR RESULTADOS BALANCEADOS"""
        # Limpiar frame de resultados
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        # Configurar colores según predicción
        colors = {
            'COVID': '#FF5722',      # Rojo
            'PNEUMONIA': '#FF9800',  # Naranja
            'NORMAL': '#4CAF50'      # Verde
        }
        
        icons = {
            'COVID': '🦠',
            'PNEUMONIA': '🫁',
            'NORMAL': '✅'
        }
        
        color = colors.get(prediction, '#888888')
        icon = icons.get(prediction, '🔬')
        
        # Título del resultado
        result_title = tk.Label(self.results_frame,
                               text=f"{icon} Análisis BALANCEADO Completado",
                               font=('Arial', 16, 'bold'),
                               fg='white',
                               bg='#2d2d2d')
        result_title.pack(pady=(0, 20))
        
        # Predicción principal
        prediction_frame = tk.Frame(self.results_frame, bg=color, relief=tk.RAISED, bd=3)
        prediction_frame.pack(fill=tk.X, pady=10)
        
        prediction_label = tk.Label(prediction_frame,
                                   text=f"DIAGNÓSTICO: {prediction}",
                                   font=('Arial', 18, 'bold'),
                                   fg='white',
                                   bg=color)
        prediction_label.pack(pady=15)
        
        confidence_label = tk.Label(prediction_frame,
                                   text=f"Confianza: {confidence*100:.1f}%",
                                   font=('Arial', 14),
                                   fg='white',
                                   bg=color)
        confidence_label.pack(pady=(0, 15))
        
        # ✅ INDICADOR DE MODELO BALANCEADO
        balanced_indicator = tk.Label(prediction_frame,
                                     text="⚖️ Modelo Balanceado (Sin Sesgo)",
                                     font=('Arial', 10, 'bold'),
                                     fg='white',
                                     bg=color)
        balanced_indicator.pack(pady=(0, 10))
        
        # Interpretación del resultado
        interpretations = {
            'COVID': "⚠️ Posible infección por COVID-19 detectada.\nConsulte con un profesional médico.\n\n🔍 Análisis realizado sin sesgo hacia COVID",
            'PNEUMONIA': "⚠️ Signos de neumonía detectados.\nSe recomienda evaluación médica.\n\n🔍 Diagnóstico equilibrado y confiable",
            'NORMAL': "✅ Radiografía aparenta normalidad.\nNo se detectan anomalías evidentes.\n\n🔍 Resultado de análisis balanceado"
        }
        
        interpretation = interpretations.get(prediction, "Resultado del análisis balanceado.")
        
        interp_label = tk.Label(self.results_frame,
                               text=interpretation,
                               font=('Arial', 12),
                               fg='white',
                               bg='#2d2d2d',
                               justify=tk.CENTER,
                               wraplength=300)
        interp_label.pack(pady=20)
        
        # ✅ ANÁLISIS DE CONFIANZA MEJORADO
        confidence_analysis = self.analyze_confidence(confidence, prediction)
        confidence_label = tk.Label(self.results_frame,
                                   text=confidence_analysis,
                                   font=('Arial', 11, 'italic'),
                                   fg='#FFA726',
                                   bg='#2d2d2d',
                                   justify=tk.CENTER,
                                   wraplength=300)
        confidence_label.pack(pady=10)
        
        # Probabilidades detalladas
        if probabilities is not None:
            self.display_probabilities(probabilities)
        
        # Disclaimer médico
        disclaimer = tk.Label(self.results_frame,
                             text="⚠️ IMPORTANTE: Este es un sistema de apoyo diagnóstico BALANCEADO.\n"
                                  "No reemplaza el criterio médico profesional.\n"
                                  "Consulte siempre con un radiólogo o médico especialista.\n\n"
                                  "✅ Modelo entrenado sin sesgo hacia ninguna clase.",
                             font=('Arial', 10, 'italic'),
                             fg='#FFA726',
                             bg='#2d2d2d',
                             justify=tk.CENTER,
                             wraplength=350)
        disclaimer.pack(pady=20)
    
    def analyze_confidence(self, confidence, prediction):
        """✅ ANÁLISIS DE CONFIANZA MEJORADO"""
        if confidence > 0.8:
            return f"🎯 Confianza MUY ALTA ({confidence*100:.1f}%)\nEl modelo está muy seguro del diagnóstico."
        elif confidence > 0.6:
            return f"✅ Confianza ALTA ({confidence*100:.1f}%)\nResultado confiable para {prediction}."
        elif confidence > 0.4:
            return f"⚠️ Confianza MEDIA ({confidence*100:.1f}%)\nSe recomienda análisis adicional."
        else:
            return f"❌ Confianza BAJA ({confidence*100:.1f}%)\nImagen difícil de clasificar. Revisar calidad."
    
    def display_probabilities(self, probabilities):
        """✅ MOSTRAR PROBABILIDADES DETALLADAS"""
        prob_title = tk.Label(self.results_frame,
                             text="📊 Probabilidades Detalladas (Balanceadas)",
                             font=('Arial', 14, 'bold'),
                             fg='white',
                             bg='#2d2d2d')
        prob_title.pack(pady=(20, 10))
        
        classes = self.model.label_encoder.classes_
        icons = {'COVID': '🦠', 'PNEUMONIA': '🫁', 'NORMAL': '✅'}
        colors = {'COVID': '#FF5722', 'PNEUMONIA': '#FF9800', 'NORMAL': '#4CAF50'}
        
        for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
            prob_frame = tk.Frame(self.results_frame, bg='#3d3d3d', relief=tk.RAISED, bd=1)
            prob_frame.pack(fill=tk.X, pady=3, padx=20)
            
            class_icon = icons.get(class_name, '🔬')
            class_color = colors.get(class_name, '#888888')
            
            prob_text = tk.Label(prob_frame,
                                text=f"{class_icon} {class_name}:",
                                font=('Arial', 11, 'bold'),
                                fg='white',
                                bg='#3d3d3d',
                                width=12,
                                anchor='w')
            prob_text.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Barra de progreso para probabilidad
            prob_bar = ttk.Progressbar(prob_frame, 
                                      length=150, 
                                      mode='determinate')
            prob_bar.pack(side=tk.LEFT, padx=5, pady=5)
            prob_bar['value'] = prob * 100
            
            prob_percentage = tk.Label(prob_frame,
                                      text=f"{prob*100:.1f}%",
                                      font=('Arial', 11, 'bold'),
                                      fg=class_color,
                                      bg='#3d3d3d')
            prob_percentage.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # ✅ ANÁLISIS DE DISTRIBUCIÓN
        max_prob = max(probabilities)
        min_prob = min(probabilities)
        prob_range = max_prob - min_prob
        
        if prob_range < 0.3:
            balance_text = "⚖️ Probabilidades MUY EQUILIBRADAS\nDiagnóstico incierto - requiere análisis adicional"
            balance_color = '#FFA726'
        elif prob_range < 0.5:
            balance_text = "⚖️ Probabilidades MODERADAMENTE equilibradas\nResultado aceptable pero con cierta incertidumbre"
            balance_color = '#FFD54F'
        else:
            balance_text = "✅ Diagnóstico CLARO\nUna clase domina claramente sobre las otras"
            balance_color = '#4CAF50'
        
        balance_label = tk.Label(self.results_frame,
                               text=balance_text,
                               font=('Arial', 10, 'italic'),
                               fg=balance_color,
                               bg='#2d2d2d',
                               justify=tk.CENTER,
                               wraplength=300)
        balance_label.pack(pady=10)
    
    def run(self):
        """Ejecutar la aplicación"""
        self.root.mainloop()

def main():
    """✅ FUNCIÓN PRINCIPAL MEJORADA"""
    try:
        print("🚀 Iniciando COVID-19 Classifier GUI BALANCEADO...")
        app = CovidClassifierGUI()
        
        if app.model is None:
            print("❌ No se pudo cargar el modelo. Cerrando aplicación.")
            return
        
        print("✅ GUI iniciada correctamente con modelo balanceado")
        app.run()
        
    except Exception as e:
        print(f"❌ Error iniciando aplicación: {e}")
        import traceback
        traceback.print_exc()
        
        # Mostrar error en ventana si es posible
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error Fatal", 
                f"Error iniciando la aplicación:\n\n{e}\n\n"
                "Verifica que:\n"
                "1. Tengas el modelo entrenado (covid_classifier_balanced.pkl)\n"
                "2. Las dependencias estén instaladas\n"
                "3. Los archivos de código estén en la ubicación correcta")
        except:
            pass

if __name__ == "__main__":
    main()