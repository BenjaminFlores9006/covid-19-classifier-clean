import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
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
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import threading
from datetime import datetime

class ScrollableFrame(tk.Frame):
    """Frame con scroll personalizado"""
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        
        # Canvas y scrollbar
        self.canvas = tk.Canvas(self, bg='#2d2d2d', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#2d2d2d')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Bind mouse wheel
        self.bind_mousewheel()
    
    def bind_mousewheel(self):
        """Habilitar scroll con rueda del mouse"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind("<MouseWheel>", _on_mousewheel)
        self.scrollable_frame.bind("<MouseWheel>", _on_mousewheel)

class EmailDialog:
    """Diálogo mejorado para envío de correos con Outlook SMTP"""
    def __init__(self, parent, diagnosis, confidence, image_path=None):
        self.parent = parent
        self.diagnosis = diagnosis
        self.confidence = confidence
        self.image_path = image_path
        self.result = None
        
        self.create_dialog()

    def create_dialog(self):
        """Crear ventana de diálogo"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("📤 Enviar Resultados por Correo")
        self.dialog.geometry("450x520")
        self.dialog.resizable(False, False)
        self.dialog.configure(bg='#2d2d2d')
        self.dialog.transient(self.parent)
        self.dialog.grab_set()

        self.dialog.geometry("+%d+%d" % (
            self.parent.winfo_rootx() + 50,
            self.parent.winfo_rooty() + 50
        ))

        main_frame = tk.Frame(self.dialog, bg='#2d2d2d')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        title_label = tk.Label(main_frame,
                               text="📧 Configurar Envío de Resultados",
                               font=('Arial', 14, 'bold'),
                               fg='#4CAF50',
                               bg='#2d2d2d')
        title_label.pack(pady=(0, 15))

        info_frame = tk.Frame(main_frame, bg='#3d3d3d', relief=tk.RAISED, bd=2)
        info_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(info_frame,
                 text=f"Diagnóstico: {self.diagnosis} ({self.confidence*100:.1f}%)",
                 font=('Arial', 11, 'bold'),
                 fg='white',
                 bg='#3d3d3d').pack(pady=8)

        fields_frame = tk.Frame(main_frame, bg='#2d2d2d')
        fields_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(fields_frame, text="👤 Nombre del Paciente:",
                 font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor='w')
        self.name_entry = tk.Entry(fields_frame, width=45, font=('Arial', 10))
        self.name_entry.pack(fill=tk.X, pady=(3, 8))

        tk.Label(fields_frame, text="👨‍⚕️ Correo del Doctor:",
                 font=('Arial', 10, 'bold'), fg='white', bg='#2d2d2d').pack(anchor='w')
        self.email_entry = tk.Entry(fields_frame, width=45, font=('Arial', 10))
        self.email_entry.pack(fill=tk.X, pady=(3, 8))
        self.email_entry.insert(0, "doctor@ejemplo.com")

        sender_frame = tk.LabelFrame(fields_frame, text="🔐 Configuración del Remitente",
                                     font=('Arial', 9, 'bold'), fg='#4CAF50', bg='#2d2d2d')
        sender_frame.pack(fill=tk.X, pady=(8, 0))

        tk.Label(sender_frame, text="Correo del remitente:",
                 font=('Arial', 8), fg='white', bg='#2d2d2d').pack(anchor='w', padx=8, pady=(3, 0))
        self.sender_email_entry = tk.Entry(sender_frame, width=45, font=('Arial', 8))
        self.sender_email_entry.pack(fill=tk.X, padx=8, pady=(1, 2))
        self.sender_email_entry.insert(0, "tucorreo@outlook.com")

        tk.Label(sender_frame, text="Contraseña:",
                 font=('Arial', 8), fg='white', bg='#2d2d2d').pack(anchor='w', padx=8)
        self.password_entry = tk.Entry(sender_frame, width=45, font=('Arial', 8), show="*")
        self.password_entry.pack(fill=tk.X, padx=8, pady=(1, 2))

        help_label = tk.Label(sender_frame, text="💡 Usa tu contraseña de Correo (o clave de app si tienes 2FA)",
                              font=('Arial', 7), fg='#FFA726', bg='#2d2d2d')
        help_label.pack(padx=8, pady=(0, 5))

        options_frame = tk.Frame(main_frame, bg='#2d2d2d')
        options_frame.pack(fill=tk.X, pady=(5, 0))

        self.include_image_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="📎 Adjuntar imagen",
                       variable=self.include_image_var,
                       font=('Arial', 8), fg='white', bg='#2d2d2d',
                       selectcolor='#4CAF50').pack(anchor='w')

        buttons_frame = tk.Frame(main_frame, bg='#2d2d2d')
        buttons_frame.pack(fill=tk.X, pady=(15, 5))

        send_btn = tk.Button(buttons_frame,
                             text="📤 ENVIAR CORREO",
                             command=self.send_email,
                             bg='#4CAF50',
                             fg='white',
                             font=('Arial', 10, 'bold'),
                             relief=tk.RAISED,
                             bd=2,
                             padx=15,
                             pady=5)
        send_btn.pack(fill=tk.X, pady=(0, 8))

        cancel_btn = tk.Button(buttons_frame,
                                text="❌ CANCELAR",
                                command=self.cancel,
                                bg='#f44336',
                                fg='white',
                                font=('Arial', 10, 'bold'),
                                relief=tk.RAISED,
                                bd=2,
                                padx=15,
                                pady=5)
        cancel_btn.pack(fill=tk.X)

        tk.Label(main_frame, text=" ", bg='#2d2d2d').pack()

        self.name_entry.focus()

    def cancel(self):
        """Cancelar diálogo"""
        self.result = False
        self.dialog.destroy()

    def send_email(self):
        """Enviar correo electrónico"""
        patient_name = self.name_entry.get().strip()
        patient_email = self.email_entry.get().strip()
        sender_email = self.sender_email_entry.get().strip()
        sender_password = self.password_entry.get().strip()

        if not all([patient_name, patient_email, sender_email, sender_password]):
            messagebox.showerror("Error", "❌ Por favor complete todos los campos")
            return

        if "@" not in patient_email or "@" not in sender_email:
            messagebox.showerror("Error", "❌ Formato de correo inválido")
            return

        progress_window = self.show_progress()

        thread = threading.Thread(target=self._send_email_thread,
                                  args=(patient_name, patient_email, sender_email, sender_password, progress_window))
        thread.daemon = True
        thread.start()

    def show_progress(self):
        """Mostrar ventana de progreso"""
        progress_window = tk.Toplevel(self.dialog)
        progress_window.title("Enviando...")
        progress_window.geometry("350x120")
        progress_window.resizable(False, False)
        progress_window.configure(bg='#2d2d2d')
        progress_window.transient(self.dialog)
        progress_window.grab_set()

        progress_window.geometry("+%d+%d" % (
            self.dialog.winfo_rootx() + 100,
            self.dialog.winfo_rooty() + 100
        ))

        tk.Label(progress_window, text="📤 Enviando correo electrónico...",
                 font=('Arial', 12), fg='white', bg='#2d2d2d').pack(pady=(15, 5))

        self.progress_label = tk.Label(progress_window, text="Conectando al servidor...",
                                      font=('Arial', 10), fg='#FFA726', bg='#2d2d2d')
        self.progress_label.pack(pady=5)

        progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
        progress_bar.pack(fill=tk.X, padx=20, pady=(5, 15))
        progress_bar.start()

        return progress_window

    def update_progress(self, message):
        """Actualizar mensaje de progreso"""
        if hasattr(self, 'progress_label'):
            self.progress_label.config(text=message)

    def _send_email_thread(self, patient_name, patient_email, sender_email, sender_password, progress_window):
        """Enviar correo en hilo separado usando Outlook SMTP con múltiples intentos"""
        try:
            self.dialog.after(0, lambda: self.update_progress("Preparando mensaje..."))
            
            # Crear mensaje
            msg = MIMEMultipart('alternative')
            msg['From'] = sender_email
            msg['To'] = patient_email
            msg['Subject'] = f"🩺 Resultado de Diagnóstico COVID-19 - {patient_name}"

            # Cuerpo del correo
            html_body = self.create_email_body(patient_name)
            msg.attach(MIMEText(html_body, 'html'))

            # Adjuntar imagen si está seleccionado
            if self.include_image_var.get() and self.image_path and os.path.exists(self.image_path):
                self.dialog.after(0, lambda: self.update_progress("Adjuntando imagen..."))
                try:
                    with open(self.image_path, "rb") as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())

                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= radiografia_{patient_name.replace(" ", "_")}.jpg'
                    )
                    msg.attach(part)
                except Exception as e:
                    print(f"Error adjuntando imagen: {e}")

            # Intentar envío con múltiples servidores SMTP
            success = False
            error_messages = []
            
            smtp_configs = [
                # Outlook/Hotmail (más compatible)
                {"server": "smtp-mail.outlook.com", "port": 587, "name": "Outlook TLS"},
                {"server": "smtp.office365.com", "port": 587, "name": "Office365 TLS"},
                # Gmail como respaldo
                {"server": "smtp.gmail.com", "port": 587, "name": "Gmail TLS"},
                {"server": "smtp.gmail.com", "port": 465, "name": "Gmail SSL", "use_ssl": True},
            ]
            
            for config in smtp_configs:
                if success:
                    break
                    
                try:
                    self.dialog.after(0, lambda c=config: self.update_progress(f"Conectando a {c['name']}..."))
                    
                    if config.get("use_ssl", False):
                        # Conexión SSL directa
                        with smtplib.SMTP_SSL(config["server"], config["port"]) as server:
                            server.login(sender_email, sender_password)
                            server.sendmail(sender_email, patient_email, msg.as_string())
                    else:
                        # Conexión TLS
                        with smtplib.SMTP(config["server"], config["port"]) as server:
                            server.starttls()
                            server.login(sender_email, sender_password)
                            server.sendmail(sender_email, patient_email, msg.as_string())
                    
                    success = True
                    print(f"✅ Correo enviado exitosamente usando {config['name']}")
                    
                except smtplib.SMTPAuthenticationError as e:
                    error_msg = f"{config['name']}: Error de autenticación"
                    error_messages.append(error_msg)
                    print(f"❌ {error_msg}: {e}")
                    
                except Exception as e:
                    error_msg = f"{config['name']}: {str(e)}"
                    error_messages.append(error_msg)
                    print(f"❌ {error_msg}")

            if success:
                # Éxito
                self.dialog.after(0, lambda: self._email_sent_success(progress_window, patient_email))
            else:
                # Fallo en todos los métodos
                error_detail = "\n\n".join(error_messages)
                self.dialog.after(0, lambda: self._email_sent_error(progress_window, 
                    f"❌ No se pudo enviar el correo con ningún servidor.\n\n"
                    f"Intentos realizados:\n{error_detail}\n\n"
                    f"💡 Sugerencias:\n"
                    f"• Verifica tu email y contraseña\n"
                    f"• Para Outlook: usa tu contraseña normal\n"
                    f"• Para Gmail: activa 'aplicaciones menos seguras'\n"
                    f"• Verifica tu conexión a internet\n"
                    f"• Si tienes 2FA, usa contraseña de aplicación"))

        except Exception as e:
            self.dialog.after(0, lambda: self._email_sent_error(progress_window, f"❌ Error inesperado: {str(e)}"))

    def _email_sent_success(self, progress_window, email):
        """Manejar envío exitoso"""
        progress_window.destroy()
        messagebox.showinfo("¡Éxito! 🎉", f"✅ Correo enviado exitosamente a:\n\n📧 {email}\n\n🩺 El doctor recibirá el diagnóstico completo con los resultados del análisis.")
        self.result = True
        self.dialog.destroy()

    def _email_sent_error(self, progress_window, error_msg):
        """Manejar error en envío"""
        progress_window.destroy()
        messagebox.showerror("Error de Envío", error_msg)

    def create_email_body(self, patient_name):
        """Crear cuerpo del correo en HTML"""
        date_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        # Obtener recomendación
        recommendations = {
            'COVID': "🚨 Se detectaron posibles signos de COVID-19. Se recomienda:\n• Aislamiento inmediato\n• Prueba PCR confirmatoria\n• Consulta médica urgente",
            'PNEUMONIA': "⚠️ Hallazgos compatibles con neumonía. Se recomienda:\n• Evaluación por neumólogo\n• Análisis complementarios\n• Seguimiento médico",
            'NORMAL': "✅ No se detectaron anomalías significativas. Se recomienda:\n• Continuar controles rutinarios\n• Mantener medidas preventivas\n• Consulta si aparecen síntomas"
        }
        
        recommendation = recommendations.get(self.diagnosis, "Consulte con su médico para interpretación detallada.")
        
        # Colores según diagnóstico
        colors = {
            'COVID': '#FF5722',
            'PNEUMONIA': '#FF9800', 
            'NORMAL': '#4CAF50'
        }
        color = colors.get(self.diagnosis, '#2196F3')
        
        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, {color}, #1976D2); color: white; padding: 20px; text-align: center; border-radius: 10px; }}
                .content {{ background: #f9f9f9; padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .result-box {{ background: {color}; color: white; padding: 15px; border-radius: 8px; text-align: center; margin: 20px 0; }}
                .recommendation {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
                .footer {{ background: #e9ecef; padding: 15px; border-radius: 8px; font-size: 12px; color: #666; }}
                .emoji {{ font-size: 1.2em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🩺 Resultado de Diagnóstico Radiológico</h1>
                    <p>Sistema de Análisis COVID-19 BALANCEADO</p>
                </div>
                
                <div class="content">
                    <h2>📋 Información del Paciente</h2>
                    <p><strong>Nombre:</strong> {patient_name}</p>
                    <p><strong>Fecha del análisis:</strong> {date_str}</p>
                    
                    <div class="result-box">
                        <h2>🎯 RESULTADO DEL ANÁLISIS</h2>
                        <h3>{self.diagnosis}</h3>
                        <p>Nivel de confianza: {self.confidence*100:.1f}%</p>
                        <p><small>⚖️ Análisis realizado con modelo balanceado</small></p>
                    </div>
                    
                    <div class="recommendation">
                        <h3>📋 Recomendaciones Médicas</h3>
                        <p>{recommendation.replace('• ', '<br>• ')}</p>
                    </div>
                    
                    <div style="background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 20px 0;">
                        <h3>🔬 Detalles Técnicos</h3>
                        <ul>
                            <li>Modelo utilizado: COVID-19 Classifier Balanceado</li>
                            <li>Tecnología: Inteligencia Artificial (Deep Learning)</li>
                            <li>Precisión del sistema: Optimizado para evitar sesgos</li>
                            <li>Tipo de imagen: Radiografía de tórax</li>
                        </ul>
                    </div>
                </div>
                
                <div class="footer">
                    <p><strong>⚠️ AVISO MÉDICO IMPORTANTE:</strong></p>
                    <p>Este resultado es generado por un sistema de apoyo diagnóstico basado en inteligencia artificial. 
                    <strong>NO sustituye el criterio médico profesional.</strong> Es fundamental que consulte con un 
                    radiólogo o médico especialista para la interpretación definitiva de los resultados.</p>
                    
                    <hr style="margin: 15px 0;">
                    
                    <p><small>
                        🏥 Sistema desarrollado para apoyo diagnóstico<br>
                        📧 Este correo fue generado automáticamente<br>
                        🕒 Fecha de generación: {date_str}
                    </small></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_body

class CovidClassifierGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🦠 COVID-19 Classifier BALANCEADO - Análisis de Rayos X")
        self.root.geometry("1200x750")
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
        
        # Estilo para botón de correo
        style.configure('Email.TButton',
                       background='#FF9800',
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       relief='flat',
                       font=('Arial', 12, 'bold'))
        
        style.map('Email.TButton',
                 background=[('active', '#F57C00'),
                           ('pressed', '#E65100')])
    
    def create_ui(self):
        """Crear la interfaz de usuario mejorada con scroll"""
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1e1e1e')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        title_frame = tk.Frame(main_frame, bg='#1e1e1e')
        title_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(title_frame, 
                              text="🦠 COVID-19 Classifier BALANCEADO",
                              font=('Arial', 22, 'bold'),
                              fg='#4CAF50',
                              bg='#1e1e1e')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame,
                                 text="⚖️ Análisis Equilibrado • SIN SESGO • 📧 Envío por Outlook",
                                 font=('Arial', 12),
                                 fg='#888888',
                                 bg='#1e1e1e')
        subtitle_label.pack()
        
        # Frame para contenido principal con scroll
        content_frame = tk.Frame(main_frame, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Imagen (con scroll)
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Título panel izquierdo
        img_title = tk.Label(left_panel,
                            text="📷 Imagen de Rayos X",
                            font=('Arial', 14, 'bold'),
                            fg='white',
                            bg='#2d2d2d')
        img_title.pack(pady=10)
        
        # Botones de control
        buttons_frame = tk.Frame(left_panel, bg='#2d2d2d')
        buttons_frame.pack(pady=10)
        
        upload_btn = ttk.Button(buttons_frame,
                               text="📁 Subir Imagen",
                               style='Modern.TButton',
                               command=self.upload_image)
        upload_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(buttons_frame,
                              text="🗑️ Limpiar",
                              style='Modern.TButton',
                              command=self.clear_image)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Frame scrollable para imagen
        self.image_scroll_frame = ScrollableFrame(left_panel)
        self.image_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Label para mostrar imagen
        self.image_label = tk.Label(self.image_scroll_frame.scrollable_frame,
                                   text="Selecciona una imagen de rayos X\npara comenzar el análisis\n\n🖼️ Formatos soportados: PNG, JPG, JPEG, BMP, TIFF",
                                   font=('Arial', 14),
                                   fg='#888888',
                                   bg='#2d2d2d',
                                   justify=tk.CENTER)
        self.image_label.pack(expand=True, pady=50)
        
        # Panel derecho - Resultados (con scroll)
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Título panel derecho
        results_title = tk.Label(right_panel,
                                text="📊 Resultados del Análisis BALANCEADO",
                                font=('Arial', 14, 'bold'),
                                fg='white',
                                bg='#2d2d2d')
        results_title.pack(pady=10)
        
        # Botones de control
        control_frame = tk.Frame(right_panel, bg='#2d2d2d')
        control_frame.pack(pady=10)
        
        self.process_btn = ttk.Button(control_frame,
                                     text="🔬 Analizar",
                                     style='Process.TButton',
                                     command=self.process_image,
                                     state='disabled')
        self.process_btn.pack(side=tk.LEFT, padx=3)
        
        self.email_btn = ttk.Button(control_frame,
                                   text="📧 Enviar Correo",
                                   style='Email.TButton',
                                   command=self.show_email_dialog,
                                   state='disabled')
        self.email_btn.pack(side=tk.LEFT, padx=3)
        
        # Frame scrollable para resultados
        self.results_scroll_frame = ScrollableFrame(right_panel)
        self.results_scroll_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Placeholder para resultados
        self.results_label = tk.Label(self.results_scroll_frame.scrollable_frame,
                                     text="Sube una imagen y presiona\n'Analizar' para ver los resultados\n\n✅ Modelo sin sesgo hacia COVID\n📧 Envío automático con Outlook",
                                     font=('Arial', 14),
                                     fg='#888888',
                                     bg='#2d2d2d',
                                     justify=tk.CENTER)
        self.results_label.pack(expand=True, pady=50)
        
        # Progress bar
        self.progress = ttk.Progressbar(right_panel, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=20, pady=10)
        self.progress.pack_forget()  # Ocultar inicialmente
        
        # Variables para resultados
        self.last_prediction = None
        self.last_confidence = None
        self.last_probabilities = None
    
    def clear_image(self):
        """Limpiar imagen cargada"""
        self.current_image_path = None
        self.last_prediction = None
        self.last_confidence = None
        self.last_probabilities = None
        
        # Limpiar display de imagen
        for widget in self.image_scroll_frame.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.image_label = tk.Label(self.image_scroll_frame.scrollable_frame,
                                   text="Selecciona una imagen de rayos X\npara comenzar el análisis\n\n🖼️ Formatos soportados: PNG, JPG, JPEG, BMP, TIFF",
                                   font=('Arial', 14),
                                   fg='#888888',
                                   bg='#2d2d2d',
                                   justify=tk.CENTER)
        self.image_label.pack(expand=True, pady=50)
        
        # Limpiar resultados
        for widget in self.results_scroll_frame.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.results_label = tk.Label(self.results_scroll_frame.scrollable_frame,
                                     text="Sube una imagen y presiona\n'Analizar' para ver los resultados\n\n✅ Modelo sin sesgo hacia COVID\n📧 Envío automático con Outlook",
                                     font=('Arial', 14),
                                     fg='#888888',
                                     bg='#2d2d2d',
                                     justify=tk.CENTER)
        self.results_label.pack(expand=True, pady=50)
        
        # Deshabilitar botones
        self.process_btn.config(state='disabled')
        self.email_btn.config(state='disabled')
    
    def upload_image(self):
        """Subir imagen"""
        file_types = [
            ('Imágenes', '*.png *.jpg *.jpeg *.bmp *.tiff *.gif'),
            ('PNG', '*.png'),
            ('JPEG', '*.jpg *.jpeg'),
            ('BMP', '*.bmp'),
            ('TIFF', '*.tiff'),
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
            for widget in self.results_scroll_frame.scrollable_frame.winfo_children():
                widget.destroy()
            
            self.results_label = tk.Label(self.results_scroll_frame.scrollable_frame,
                                         text="✅ Imagen cargada exitosamente\n\nPresiona 'Analizar' para procesar\n\n⚖️ Análisis sin sesgo hacia COVID\n🔬 Modelo de IA entrenado",
                                         font=('Arial', 14),
                                         fg='#4CAF50',
                                         bg='#2d2d2d',
                                         justify=tk.CENTER)
            self.results_label.pack(expand=True, pady=50)
    
    def display_image(self, image_path):
        """✅ MOSTRAR IMAGEN MEJORADO CON SCROLL"""
        try:
            # Cargar imagen
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("No se pudo cargar la imagen")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar para mostrar
            height, width = img_rgb.shape[:2]
            max_size = 500
            
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            img_resized = cv2.resize(img_rgb, (new_width, new_height))
            
            # Limpiar frame anterior
            for widget in self.image_scroll_frame.scrollable_frame.winfo_children():
                widget.destroy()
            
            # Frame para contenido de imagen
            content_frame = tk.Frame(self.image_scroll_frame.scrollable_frame, bg='#2d2d2d')
            content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Título de imagen
            img_title = tk.Label(content_frame, 
                                text="📷 Imagen Cargada", 
                                font=('Arial', 16, 'bold'),
                                fg='#4CAF50', bg='#2d2d2d')
            img_title.pack(pady=(0, 15))
            
            # Frame para la imagen
            img_frame = tk.Frame(content_frame, bg='#3d3d3d', relief=tk.RAISED, bd=2)
            img_frame.pack(pady=10)
            
            # Imagen
            img_pil = Image.fromarray(img_resized)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            self.image_label = tk.Label(img_frame, image=img_tk, bg='#3d3d3d')
            self.image_label.image = img_tk  # Mantener referencia
            self.image_label.pack(padx=10, pady=10)
            
            # Información detallada de la imagen
            info_frame = tk.Frame(content_frame, bg='#3d3d3d', relief=tk.RAISED, bd=2)
            info_frame.pack(fill=tk.X, pady=15)
            
            tk.Label(info_frame, text="📋 Información de la Imagen",
                    font=('Arial', 14, 'bold'), fg='#4CAF50', bg='#3d3d3d').pack(pady=(10, 5))
            
            # Información técnica
            file_size = os.path.getsize(image_path) / 1024  # KB
            file_name = Path(image_path).name
            
            info_text = [
                f"📄 Archivo: {file_name}",
                f"📐 Dimensiones: {width} × {height} píxeles",
                f"💾 Tamaño: {file_size:.1f} KB",
                f"🖼️ Formato: {Path(image_path).suffix.upper()[1:]}",
                f"📊 Canales: {len(img_rgb.shape)} ({'Color' if len(img_rgb.shape) == 3 else 'Escala de grises'})"
            ]
            
            for info in info_text:
                tk.Label(info_frame, text=info, font=('Arial', 11),
                        fg='white', bg='#3d3d3d').pack(anchor='w', padx=15, pady=2)
            
            # Estado del modelo
            model_frame = tk.Frame(content_frame, bg='#4CAF50', relief=tk.RAISED, bd=2)
            model_frame.pack(fill=tk.X, pady=15)
            
            tk.Label(model_frame, text="🤖 Estado del Modelo",
                    font=('Arial', 14, 'bold'), fg='white', bg='#4CAF50').pack(pady=(10, 5))
            
            model_info = [
                "✅ Modelo cargado correctamente",
                "⚖️ Configuración balanceada (sin sesgo)",
                "🎯 Listo para análisis de COVID/Pneumonia/Normal",
                "🔬 IA entrenada con datos equilibrados"
            ]
            
            for info in model_info:
                tk.Label(model_frame, text=info, font=('Arial', 11),
                        fg='white', bg='#4CAF50').pack(anchor='w', padx=15, pady=2)
            
            tk.Label(model_frame, text=" ", bg='#4CAF50').pack(pady=5)  # Espaciado
            
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
            if hasattr(self.model, 'predict'):
                result = self.model.predict(self.current_image_path)
                
                if len(result) == 3:
                    class_name, confidence, all_probabilities = result
                else:
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
            
            # Guardar resultados
            self.last_prediction = class_name
            self.last_confidence = confidence
            self.last_probabilities = all_probabilities
            
            # Mostrar resultados
            self.display_results(class_name, confidence, all_probabilities)
            
            # Habilitar botón de correo
            self.email_btn.config(state='normal')
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            messagebox.showerror("Error", f"Error procesando imagen: {e}")
            print(f"❌ Error completo: {e}")
            import traceback
            traceback.print_exc()
    
    def display_results(self, prediction, confidence, probabilities):
        """✅ MOSTRAR RESULTADOS BALANCEADOS CON SCROLL"""
        # Limpiar frame de resultados
        for widget in self.results_scroll_frame.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Frame principal para resultados
        results_container = tk.Frame(self.results_scroll_frame.scrollable_frame, bg='#2d2d2d')
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
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
        result_title = tk.Label(results_container,
                               text=f"{icon} Análisis BALANCEADO Completado",
                               font=('Arial', 20, 'bold'),
                               fg='white',
                               bg='#2d2d2d')
        result_title.pack(pady=(0, 25))
        
        # Predicción principal - Frame destacado
        prediction_frame = tk.Frame(results_container, bg=color, relief=tk.RAISED, bd=4)
        prediction_frame.pack(fill=tk.X, pady=15)
        
        # Contenido de predicción
        pred_content = tk.Frame(prediction_frame, bg=color)
        pred_content.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(pred_content, text=f"{icon} DIAGNÓSTICO DETECTADO",
                font=('Arial', 14, 'bold'), fg='white', bg=color).pack()
        
        tk.Label(pred_content, text=prediction,
                font=('Arial', 24, 'bold'), fg='white', bg=color).pack(pady=(5, 10))
        
        tk.Label(pred_content, text=f"Nivel de Confianza: {confidence*100:.1f}%",
                font=('Arial', 16), fg='white', bg=color).pack()
        
        tk.Label(pred_content, text="⚖️ Resultado de Modelo Balanceado (Sin Sesgo)",
                font=('Arial', 12, 'bold'), fg='white', bg=color).pack(pady=(10, 0))
        
        # Análisis de confianza
        confidence_frame = tk.Frame(results_container, bg='#3d3d3d', relief=tk.RAISED, bd=2)
        confidence_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(confidence_frame, text="📊 Análisis de Confianza",
                font=('Arial', 16, 'bold'), fg='#4CAF50', bg='#3d3d3d').pack(pady=(15, 10))
        
        confidence_analysis = self.analyze_confidence(confidence, prediction)
        tk.Label(confidence_frame, text=confidence_analysis,
                font=('Arial', 13), fg='white', bg='#3d3d3d',
                justify=tk.CENTER, wraplength=400).pack(pady=(0, 15))
        
        # Interpretación médica
        interpretation_frame = tk.Frame(results_container, bg='#3d3d3d', relief=tk.RAISED, bd=2)
        interpretation_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(interpretation_frame, text="🩺 Interpretación Médica",
                font=('Arial', 16, 'bold'), fg='#4CAF50', bg='#3d3d3d').pack(pady=(15, 10))
        
        interpretations = {
            'COVID': "⚠️ Se detectaron patrones compatibles con infección por COVID-19.\n\n🔍 Recomendaciones:\n• Aislamiento preventivo inmediato\n• Confirmación con prueba PCR/antígenos\n• Consulta médica urgente\n• Monitoreo de síntomas\n\n📋 Análisis realizado sin sesgo hacia COVID",
            'PNEUMONIA': "⚠️ Se identificaron signos radiológicos de neumonía.\n\n🔍 Recomendaciones:\n• Evaluación por especialista en neumología\n• Estudios complementarios (laboratorio, cultivos)\n• Tratamiento según protocolo médico\n• Seguimiento radiológico\n\n📋 Diagnóstico equilibrado y confiable",
            'NORMAL': "✅ La radiografía presenta características dentro de parámetros normales.\n\n🔍 Recomendaciones:\n• Continuar con controles médicos rutinarios\n• Mantener medidas preventivas de salud\n• Consultar si aparecen síntomas respiratorios\n• Seguimiento según criterio médico\n\n📋 Resultado de análisis balanceado"
        }
        
        interpretation = interpretations.get(prediction, "Consulte con un profesional médico para interpretación detallada.")
        
        tk.Label(interpretation_frame, text=interpretation,
                font=('Arial', 12), fg='white', bg='#3d3d3d',
                justify=tk.LEFT, wraplength=450).pack(padx=20, pady=(0, 15))
        
        # Probabilidades detalladas
        if probabilities is not None:
            self.display_probabilities_enhanced(results_container, probabilities)
        
        # Disclaimer médico mejorado
        disclaimer_frame = tk.Frame(results_container, bg='#FF6B35', relief=tk.RAISED, bd=3)
        disclaimer_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(disclaimer_frame, text="⚠️ AVISO MÉDICO IMPORTANTE",
                font=('Arial', 16, 'bold'), fg='white', bg='#FF6B35').pack(pady=(15, 5))
        
        disclaimer_text = ("Este sistema constituye una herramienta de apoyo diagnóstico basada en "
                          "inteligencia artificial con modelo BALANCEADO. Los resultados NO sustituyen "
                          "el criterio médico profesional ni el diagnóstico clínico definitivo.\n\n"
                          "SIEMPRE consulte con un radiólogo o médico especialista para la "
                          "interpretación final de los estudios radiológicos.\n\n"
                          "✅ Modelo entrenado sin sesgo hacia ninguna clase específica.")
        
        tk.Label(disclaimer_frame, text=disclaimer_text,
                font=('Arial', 11, 'italic'), fg='white', bg='#FF6B35',
                justify=tk.CENTER, wraplength=450).pack(padx=20, pady=(0, 15))
        
        # Información técnica del análisis
        tech_frame = tk.Frame(results_container, bg='#1565C0', relief=tk.RAISED, bd=2)
        tech_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(tech_frame, text="🔬 Información Técnica del Análisis",
                font=('Arial', 14, 'bold'), fg='white', bg='#1565C0').pack(pady=(10, 5))
        
        tech_info = [
            f"🤖 Modelo: COVID-19 Classifier Balanceado",
            f"📊 Tecnología: Deep Learning (Redes Neuronales)",
            f"⚖️ Características: Sin sesgo hacia clases específicas",
            f"🎯 Clases detectables: COVID-19, Neumonía, Normal",
            f"📷 Imagen procesada: {Path(self.current_image_path).name}",
            f"🕒 Procesamiento: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        ]
        
        for info in tech_info:
            tk.Label(tech_frame, text=info, font=('Arial', 10),
                    fg='white', bg='#1565C0').pack(anchor='w', padx=20, pady=1)
        
        tk.Label(tech_frame, text=" ", bg='#1565C0').pack(pady=5)
    
    def display_probabilities_enhanced(self, parent, probabilities):
        """✅ MOSTRAR PROBABILIDADES DETALLADAS MEJORADAS"""
        prob_frame = tk.Frame(parent, bg='#3d3d3d', relief=tk.RAISED, bd=2)
        prob_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(prob_frame, text="📊 Distribución de Probabilidades (Balanceadas)",
                font=('Arial', 16, 'bold'), fg='#4CAF50', bg='#3d3d3d').pack(pady=(15, 20))
        
        classes = self.model.label_encoder.classes_
        icons = {'COVID': '🦠', 'PNEUMONIA': '🫁', 'NORMAL': '✅'}
        colors = {'COVID': '#FF5722', 'PNEUMONIA': '#FF9800', 'NORMAL': '#4CAF50'}
        
        # Crear barras de probabilidad mejoradas
        for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
            class_frame = tk.Frame(prob_frame, bg='#4d4d4d', relief=tk.RAISED, bd=1)
            class_frame.pack(fill=tk.X, padx=20, pady=5)
            
            # Frame para contenido de clase
            content_frame = tk.Frame(class_frame, bg='#4d4d4d')
            content_frame.pack(fill=tk.X, padx=15, pady=10)
            
            # Información de clase
            class_icon = icons.get(class_name, '🔬')
            class_color = colors.get(class_name, '#888888')
            
            info_frame = tk.Frame(content_frame, bg='#4d4d4d')
            info_frame.pack(fill=tk.X)
            
            tk.Label(info_frame, text=f"{class_icon} {class_name}",
                    font=('Arial', 13, 'bold'), fg='white', bg='#4d4d4d').pack(side=tk.LEFT)
            
            tk.Label(info_frame, text=f"{prob*100:.2f}%",
                    font=('Arial', 13, 'bold'), fg=class_color, bg='#4d4d4d').pack(side=tk.RIGHT)
            
            # Barra de progreso visual
            bar_frame = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.SUNKEN, bd=1)
            bar_frame.pack(fill=tk.X, pady=(5, 0))
            
            bar_width = int(300 * prob)  # Ancho proporcional
            if bar_width > 0:
                progress_bar = tk.Frame(bar_frame, bg=class_color, height=20)
                progress_bar.pack(side=tk.LEFT, padx=2, pady=2)
                progress_bar.configure(width=bar_width)
        
        # Análisis de distribución mejorado
        analysis_frame = tk.Frame(prob_frame, bg='#2d2d2d', relief=tk.SUNKEN, bd=2)
        analysis_frame.pack(fill=tk.X, padx=20, pady=(15, 0))
        
        max_prob = max(probabilities)
        min_prob = min(probabilities)
        prob_range = max_prob - min_prob
        
        if prob_range < 0.2:
            balance_text = "⚖️ DISTRIBUCIÓN MUY EQUILIBRADA\n\nLas probabilidades están muy cerca entre sí.\nEsto indica incertidumbre en el diagnóstico.\nSe recomienda análisis adicional o segunda opinión."
            balance_color = '#FFA726'
        elif prob_range < 0.4:
            balance_text = "⚖️ DISTRIBUCIÓN MODERADAMENTE EQUILIBRADA\n\nExiste una tendencia clara pero con cierta incertidumbre.\nResultado aceptable pero considerar factores clínicos adicionales."
            balance_color = '#FFD54F'
        else:
            balance_text = "✅ DIAGNÓSTICO CLARO Y DEFINIDO\n\nUna clase domina significativamente sobre las otras.\nAlta confianza en el resultado del análisis."
            balance_color = '#4CAF50'
        
        tk.Label(analysis_frame, text="📈 Análisis de Distribución",
                font=('Arial', 13, 'bold'), fg=balance_color, bg='#2d2d2d').pack(pady=(10, 5))
        
        tk.Label(analysis_frame, text=balance_text,
                font=('Arial', 11), fg='white', bg='#2d2d2d',
                justify=tk.CENTER, wraplength=400).pack(pady=(0, 10))
    
    def analyze_confidence(self, confidence, prediction):
        """✅ ANÁLISIS DE CONFIANZA MEJORADO"""
        if confidence > 0.85:
            return f"🎯 CONFIANZA MUY ALTA ({confidence*100:.1f}%)\n\nEl modelo está muy seguro del diagnóstico de {prediction}.\nResultado altamente confiable para apoyo diagnóstico."
        elif confidence > 0.70:
            return f"✅ CONFIANZA ALTA ({confidence*100:.1f}%)\n\nBuen nivel de certeza en el diagnóstico de {prediction}.\nResultado confiable, considerar correlación clínica."
        elif confidence > 0.55:
            return f"⚠️ CONFIANZA MODERADA ({confidence*100:.1f}%)\n\nNivel aceptable de certeza para {prediction}.\nSe recomienda análisis complementario y evaluación clínica."
        else:
            return f"❌ CONFIANZA BAJA ({confidence*100:.1f}%)\n\nEl modelo tiene dificultades para clasificar esta imagen.\nRevisar calidad de imagen y considerar nueva captura."
    
    def show_email_dialog(self):
        """Mostrar diálogo para envío de correo"""
        if not self.last_prediction or not self.last_confidence:
            messagebox.showerror("Error", "No hay resultados para enviar.\nPrimero procesa una imagen.")
            return
        
        # Crear y mostrar diálogo
        email_dialog = EmailDialog(
            self.root, 
            self.last_prediction, 
            self.last_confidence, 
            self.current_image_path
        )
        
        # Esperar a que se cierre el diálogo
        self.root.wait_window(email_dialog.dialog)
    
    def run(self):
        """Ejecutar la aplicación"""
        self.root.mainloop()

def main():
    """✅ FUNCIÓN PRINCIPAL MEJORADA"""
    try:
        print("🚀 Iniciando COVID-19 Classifier GUI BALANCEADO con OUTLOOK SMTP...")
        
        # Verificar dependencias críticas
        required_modules = ['cv2', 'PIL', 'numpy', 'matplotlib']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            error_msg = f"Módulos faltantes: {', '.join(missing_modules)}\n\nInstala con: pip install opencv-python pillow numpy matplotlib"
            print(f"❌ {error_msg}")
            
            try:
                root = tk.Tk()
                root.withdraw()
                messagebox.showerror("Dependencias Faltantes", error_msg)
            except:
                pass
            return
        
        app = CovidClassifierGUI()
        
        if app.model is None:
            print("❌ No se pudo cargar el modelo. Cerrando aplicación.")
            return
        
        print("✅ GUI iniciada correctamente con modelo balanceado")
        print("🔧 Características habilitadas:")
        print("   • Interfaz con scroll en ambos paneles")
        print("   • Sistema de envío de correo con Outlook SMTP") 
        print("   • Múltiples servidores SMTP de respaldo")
        print("   • Análisis detallado de resultados")
        print("   • Modelo balanceado sin sesgo")
        
        app.run()
        
    except Exception as e:
        print(f"❌ Error iniciando aplicación: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Error Fatal", 
                f"Error iniciando la aplicación:\n\n{e}\n\n"
                "Verifica que:\n"
                "1. Tengas el modelo entrenado (covid_classifier_balanced.pkl)\n"
                "2. Las dependencias estén instaladas correctamente\n"
                "3. Los archivos de código estén en la ubicación correcta\n"
                "4. Tengas permisos de lectura/escritura en el directorio")
        except:
            pass

if __name__ == "__main__":
    main()