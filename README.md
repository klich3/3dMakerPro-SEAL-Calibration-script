# SEAL_CALIB_Builder

## Descripción
Herramientas para calibración estéreo de cámaras y reconstrucción 3D con láser y UV.

## Cámaras USB Detectadas

```text
KYT Camera A:

  N.º de modelo:	UVC Camera VendorID_3141 ProductID_25450
  Identificador único:	0x145110000c45636a

KYT Camera B:

  N.º de modelo:	UVC Camera VendorID_3141 ProductID_25451
  Identificador único:	0x145120000c45636b
```

## Configuración de cámaras

- Cámara láser (frontal): Índice 0
- Cámara UV (inclinada): Índice 1

## Solución para el problema de permisos de cámara en macOS

### Problema identificado
El error "OpenCV: not authorized to capture video (status 0)" indica que la aplicación Python no tiene permiso para acceder a las cámaras del sistema.

### Solución

#### 1. Otorgar permisos de cámara a Terminal/Python

1. Abre **Preferencias del Sistema** (System Preferences)
2. Ve a **Seguridad y privacidad** (Security & Privacy)
3. Selecciona la pestaña **Privacidad** (Privacy)
4. En el panel izquierdo, selecciona **Cámara** (Camera)
5. Haz clic en el candado en la esquina inferior izquierda y proporciona tu contraseña
6. Activa la casilla junto a:
   - **Terminal** (si corres el script desde Terminal)
   - **Python** (si está disponible en la lista)

#### 2. Si no ves Terminal o Python en la lista

Ejecuta este comando en Terminal para asegurarte de que se solicite el permiso:

```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Cámara abierta:', cap.isOpened()); cap.release()"
```

Esto debería provocar que aparezca en la lista de permisos.

#### 3. Reiniciar después de cambiar permisos

Después de otorgar permisos, reinicia:
- La Terminal
- El IDE que estés usando

#### 4. Verificar permisos

Puedes verificar los permisos otorgados ejecutando:

```bash
sqlite3 ~/Library/Application\ Support/com.apple.TCC/TCC.db 'select * from access where service = "kTCCServiceCamera"'
```

## Scripts principales

### stereo_calibration.py
Calibración estéreo con dos cámaras:

```bash
# Calibración básica
python stereo_calibration.py

# Calibración con parámetros específicos
python stereo_calibration.py --left 0 --right 1 --rows 6 --cols 9 --square-size 25.0
python stereo_calibration.py --left 1 --right 0 --rows 6 --cols 9 --square-size 6.70 --images 15 --output stereo_calibration.txt --template calibJMS1006207.txt --dev-id JMS1006207

# Usar patrón de círculos
python stereo_calibration.py --pattern-type circles
python stereo_calibration.py --output stereo_calibration.txt --template calibJMS1006207.txt --dev-id JMS1006207 --pattern-type circles --rows 11 --cols 4

# Usar patrón ChArUco
python stereo_calibration.py --pattern-type charuco

# Ajustar brillo de cámara UV
python stereo_calibration.py --uv-brightness 0.5 --uv-contrast 0.5
```

### Métodos de calibración

1. **Tablero de ajedrez (chessboard)**:
   - Patrón tradicional de calibración
   - Detección robusta en diversas condiciones de iluminación
   - Requiere patrón plano completamente visible
   - Teclas: +/- para ajustar brillo/contraste de cámara UV

2. **Círculos asimétricos (circles)**:
   - Patrón de círculos dispuestos en cuadrícula asimétrica
   - Menos sensible a las distorsiones de lente
   - Permite detección parcial del patrón
   - Teclas: +/- para ajustar brillo/contraste de cámara UV

3. **ChArUco (charuco)**:
   - Combinación de marcadores ArUco y tablero de ajedrez
   - Mayor precisión en la detección de esquinas
   - Permite detección con oclusiones parciales
   - Teclas: +/- para ajustar brillo/contraste de cámara UV

### Controles de brillo y contraste

Durante la calibración, puedes ajustar en tiempo real el brillo y contraste de la cámara UV (cámara derecha/índice 1):

- **Tecla +**: Aumentar brillo
- **Tecla -**: Disminuir brillo

Para ajustar el contraste, usa los argumentos de línea de comandos:
```bash
python stereo_calibration.py --uv-contrast 0.5
```

Características:
- Detección automática de patrones
- Soporte para patrones de tablero, círculos asimétricos y ChArUco
- Ajuste de brillo/contraste en tiempo real con teclas +/- 
- Generación de archivos de calibración en formato SEAL
- Resolución fija de 1280x720


### test_camera_a-b.py
Pruebas individuales de cámaras A y B.

## Formato de archivo de calibración

Los archivos generados siguen el formato SEAL con encabezado:
```
***DevID:JMS1006207***CalibrateDate:YYYY-MM-DD_HH-MM-SS***Type:Sychev-calibration***SoftVersion:3.0.0.1116
```

## Código actualizado

El script principal ya maneja mejor los errores de permisos y proporciona mensajes más informativos.


## Instalación

`$ uv venv`
`$ source .venv/bin/activate`
`$ pip install -r requirements.txt`

### Docs

* https://developer.mamezou-tech.com/en/robotics/vision/calibration-pattern/