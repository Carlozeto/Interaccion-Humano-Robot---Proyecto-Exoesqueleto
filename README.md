# PoseDetection - Sistema de Análisis Postural con OAK-D y Webcam

## Descripción del Proyecto

Este proyecto implementa un sistema de análisis postural en tiempo real utilizando una cámara OAK-D (OAK-Depth) y una webcam convencional. El sistema combina la detección de pose mediante MediaPipe con análisis de profundidad para generar modelos 3D y métricas de postura.

## Características Principales

- **Detección de pose en tiempo real** usando MediaPipe
- **Análisis de profundidad** con cámara OAK-D
- **Cálculo de ángulos articulares** de la espalda (superior e inferior)
- **Generación de modelos 3D** en formato PLY
- **Interfaz gráfica** con PyQt5
- **Control remoto** vía web con Flask
- **Grabación de sesiones** con metadatos

## Requisitos de Hardware

### Cámara OAK-D
- **Modelo**: OAK-D, OAK-D-Lite, o similar
- **Conectividad**: USB-C
- **Resolución**: 1080p (RGB) + 400p (estéreo)
- **Profundidad**: Hasta 10 metros

### Webcam
- **Resolución mínima**: 640x480
- **FPS**: 30 o superior
- **Conectividad**: USB 2.0/3.0

### Computadora
- **Sistema Operativo**: Windows 10/11, Linux, macOS
- **RAM**: 8GB mínimo, 16GB recomendado
- **GPU**: No requerida (procesamiento CPU)
- **USB**: Puerto USB-C para OAK-D + USB para webcam

## Requisitos de Software

### Sistema Operativo
- Windows 10/11
- Ubuntu 18.04+ (recomendado)
- macOS 10.15+

### Python
- **Versión**: Python 3.8 - 3.12
- **Entorno virtual**: Recomendado

### Dependencias Principales
```
depthai>=2.30.0
mediapipe>=0.10.0
opencv-python>=4.8.0
PyQt5>=5.15.0
numpy>=1.21.0
scipy>=1.7.0
flask>=2.0.0
blobconverter>=1.4.0
```

## Instalación

### 1. Clonar el Repositorio
```bash
git clone <url-del-repositorio>
cd PoseDetection
```

### 2. Crear Entorno Virtual
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar OAK-D
```bash
# Instalar DepthAI
pip install depthai

# Verificar conexión
python -c "import depthai as dai; print('OAK-D conectado')"
```

## Estructura del Código

### Archivos Principales

#### `oakd_webcam_gui.py`
Archivo principal que contiene:
- **MainWindow**: Interfaz gráfica principal
- **OakdPoseThread**: Thread para procesamiento OAK-D
- **WebcamCapture**: Thread para captura de webcam
- **Funciones de análisis**: Cálculo de ángulos y reconstrucción 3D

#### `pose.py`
Contiene funciones auxiliares para detección de pose:
- `getKeypoints()`: Extracción de puntos clave
- `getValidPairs()`: Validación de conexiones
- `getPersonwiseKeypoints()`: Agrupación de landmarks

### Componentes Principales

#### 1. Captura de Video
```python
class OakdPoseThread(threading.Thread):
    # Maneja captura RGB, profundidad y inferencia neural
```

#### 2. Análisis de Pose
```python
def calculate_back_angles(spine_points):
    # Calcula ángulos superior e inferior de la espalda
```

#### 3. Reconstrucción 3D
```python
def create_3d_model(self):
    # Combina datos de profundidad y landmarks para modelo 3D
```

#### 4. Interfaz Web
```python
def run_flask_server(self):
    # Servidor Flask para control remoto en puerto 5000
```

## Funcionamiento del Sistema

### 1. Inicialización
- Configuración de pipeline OAK-D
- Inicialización de MediaPipe
- Configuración de threads de captura
- Inicio del servidor web

### 2. Procesamiento en Tiempo Real
- **OAK-D**: Captura RGB + profundidad + inferencia pose
- **Webcam**: Captura + MediaPipe pose detection
- **Análisis**: Cálculo de ángulos articulares
- **Visualización**: 4 ventanas (RGB, Depth, Webcam, Máscara)

### 3. Grabación
- Guarda video combinado (RGB + Depth + Webcam)
- Almacena frames de profundidad
- Guarda landmarks 3D
- Genera metadatos CSV

### 4. Generación 3D
- Segmentación de persona
- Reconstrucción de nube de puntos
- Integración de landmarks
- Exportación a PLY

## Contenido Generado

### Archivos de Video
- **Formato**: AVI (XVID codec)
- **Contenido**: RGB + Depth + Webcam combinados
- **FPS**: 20
- **Nombre**: `recording_YYYYMMDD_HHMMSS.avi`

### Datos de Profundidad
- **Formato**: NPZ (NumPy comprimido)
- **Contenido**: Frames de profundidad raw
- **Ubicación**: `recording_*/depth_data.npz`

### Landmarks 3D
- **Formato**: JSON
- **Contenido**: Coordenadas 3D de landmarks MediaPipe
- **Estructura**: Lista de frames con landmarks por frame
- **Ubicación**: `recording_*/landmarks_3d.json`

### Modelos 3D
- **Formato**: PLY (ASCII)
- **Tipos**:
  - `enhanced_3d_model_*.ply`: Nube de puntos con color
  - `spine_curve_*.ply`: Curva de columna vertebral
- **Contenido**: Puntos 3D + colores RGB

### Metadatos
- **Formato**: CSV
- **Campos**: PatientID, Posture, Filename, Timestamp
- **Archivo**: `recordings_metadata.csv`

## Procesamiento del Contenido Generado

### 1. Análisis de Video
```python
import cv2
import numpy as np

# Cargar video
cap = cv2.VideoCapture('recording_20250101_120000.avi')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Procesar frame
cap.release()
```

### 2. Análisis de Profundidad
```python
# Cargar datos de profundidad
depth_data = np.load('recording_*/depth_data.npz')
depth_frames = depth_data['depth_frames']

# Procesar cada frame
for depth_frame in depth_frames:
    # Análisis de profundidad
    pass
```

### 3. Análisis de Landmarks
```python
import json

# Cargar landmarks
with open('recording_*/landmarks_3d.json', 'r') as f:
    landmarks = json.load(f)

# Procesar landmarks por frame
for frame_landmarks in landmarks:
    for landmark in frame_landmarks:
        x, y, z = landmark['x'], landmark['y'], landmark['z']
        # Análisis de movimiento
```

### 4. Visualización 3D
```python
import open3d as o3d

# Cargar modelo PLY
pcd = o3d.io.read_point_cloud('enhanced_3d_model_*.ply')

# Visualizar
o3d.visualization.draw_geometries([pcd])
```

### 5. Análisis de Ángulos
```python
# Calcular estadísticas de ángulos
angles_data = []
for frame in landmarks:
    # Extraer landmarks de columna
    spine_landmarks = extract_spine_landmarks(frame)
    angles = calculate_back_angles(spine_landmarks)
    angles_data.append(angles)

# Análisis temporal
mean_upper_angle = np.mean([a['upper'] for a in angles_data])
std_upper_angle = np.std([a['upper'] for a in angles_data])
```

## Scripts de Análisis

### Análisis de Tendencias Posturales
```python
def analyze_postural_trends(landmarks_file):
    """Analiza tendencias posturales a lo largo del tiempo"""
    with open(landmarks_file, 'r') as f:
        landmarks = json.load(f)
    
    angles_over_time = []
    for frame_landmarks in landmarks:
        angles = calculate_back_angles_from_landmarks(frame_landmarks)
        angles_over_time.append(angles)
    
    return analyze_trends(angles_over_time)
```

### Generación de Reportes
```python
def generate_posture_report(recording_path):
    """Genera reporte completo de postura"""
    # Cargar datos
    depth_data = np.load(f'{recording_path}/depth_data.npz')
    with open(f'{recording_path}/landmarks_3d.json', 'r') as f:
        landmarks = json.load(f)
    
    # Análisis
    posture_metrics = analyze_posture_metrics(landmarks)
    depth_analysis = analyze_depth_patterns(depth_data)
    
    # Generar reporte
    create_pdf_report(posture_metrics, depth_analysis)
```

## Uso del Sistema

### 1. Ejecución Local
```bash
python oakd_webcam_gui.py
```

### 2. Control Remoto
- Abrir navegador en `http://localhost:5000`
- Usar botones para iniciar/detener grabación
- Generar modelos 3D remotamente

### 3. Configuración de Cámaras
- **OAK-D**: Conectar vía USB-C
- **Webcam**: Conectar vía USB
- **Verificar**: Las cámaras aparecen en la interfaz

### 4. Grabación de Sesión
1. Ingresar Patient ID
2. Seleccionar tipo de postura
3. Hacer clic en "Start Recording"
4. Realizar ejercicios/movimientos
5. Hacer clic en "Stop Recording"

## Troubleshooting

### Problemas Comunes

#### OAK-D no detectada
```bash
# Verificar conexión USB
lsusb | grep OAK

# Reinstalar DepthAI
pip uninstall depthai
pip install depthai
```

#### Webcam no funciona
```python
# Cambiar índice de cámara en WebcamCapture
self.cap = cv2.VideoCapture(0)  # Probar 0, 1, 2...
```

#### Errores de memoria
- Reducir resolución de captura
- Aumentar intervalo de frames
- Cerrar otras aplicaciones

#### Problemas de rendimiento
- Verificar que no hay otros procesos usando las cámaras
- Reducir FPS objetivo
- Usar SSD para almacenamiento

## Limitaciones Actuales

1. **Sincronización**: Las cámaras pueden no estar perfectamente sincronizadas
2. **Precisión**: Los ángulos dependen de la calidad de detección de pose
3. **Profundidad**: Limitada a ~10 metros con OAK-D
4. **Rendimiento**: Procesamiento en CPU puede ser lento en hardware antiguo

## Mejoras Futuras

1. **GPU Acceleration**: Usar CUDA para procesamiento más rápido
2. **Multi-persona**: Detección de múltiples personas
3. **Machine Learning**: Clasificación automática de posturas
4. **Cloud Integration**: Almacenamiento y análisis en la nube
5. **Mobile App**: Aplicación móvil para control remoto

## Conclusiones

Este sistema proporciona una solución completa para el análisis postural utilizando tecnología de visión por computadora moderna. La combinación de OAK-D y MediaPipe permite:

### Ventajas
- **Análisis en tiempo real** de postura
- **Datos 3D** para análisis más profundo
- **Interfaz intuitiva** para usuarios no técnicos
- **Almacenamiento estructurado** de datos
- **Control remoto** vía web

### Aplicaciones
- **Fisioterapia**: Análisis de movimientos
- **Deportes**: Optimización de técnica
- **Ergonomía**: Evaluación de posturas de trabajo
- **Investigación**: Estudios de biomecánica

### Impacto
El sistema democratiza el acceso a tecnología de análisis postural avanzada, permitiendo evaluaciones objetivas y cuantificables de la postura y movimiento humano.

## Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Crear un Pull Request

## Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## Contacto

Para preguntas o soporte:
- Crear un issue en GitHub
- Contactar al equipo de desarrollo

---

**Nota**: Este sistema requiere hardware específico (OAK-D) y puede no funcionar en todos los entornos. Se recomienda probar en un entorno controlado antes de uso en producción. 