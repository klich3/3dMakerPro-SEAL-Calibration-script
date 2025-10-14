# 3DMakerpro Seal Lite re-calibration

> Tools for **stereo camera calibration** and **3D reconstruction** with laser and UV, developed independently for the **3DMakerPro SEAL Lite** scanner.

- 🔬 **Compatibility**: tested **exclusively with 3DMakerPro SEAL Lite**.
- 🙅‍♂️ Project **not affiliated** with 3DMakerPro or distributors.

---

## Context / Background

After normal and careful use of the **3DMakerPro SEAL Lite**, the scanner began to produce **horizontal lines** when attempting to capture images.  
The following attempts were made, without success:

- Software updates on **macOS** and **Windows**  
- Support from **official technical support**
- Contacting the **administrator** of the Facebook group

Frustrated by the lack of a solution (even with a warranty), I developed this **proprietary calibration tool** based on **OpenCV** to restore functionality and better understand the device.

---

## ⚠️ Notice and Disclaimer

This software is provided **as is**, **without warranty** of any kind.

- Misuse may **void the warranty** of the device.
- The author **is not responsible** for any damage, loss, or failure resulting from use or misuse.
- Each user is **fully responsible** for its execution and consequences.

**Use it at your own risk and discretion.**

---

## Technical description

Stereo calibration and 3D reconstruction tools with pattern support:

- **Chessboard**
- **Asymmetric circles**
- **ChArUco (ArUco + Chessboard)**

**Cameras (SEAL Lite):**
- Camera **A**: **Laser** (front) → **index 0**
- Camera **B**: **UV** (tilted) → **index 1**

---

## Camera permissions in macOS from terminal

If you see:
```
OpenCV: not authorized to capture video (status 0)
```
Grant permissions to **Terminal/Python** in  
**System Preferences → Security & Privacy → Privacy → Camera**

Force the prompt:
```bash
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('Cam abierta:', cap.isOpened()); cap.release()"
```

If not, in my profile there is a repository for a tool that can grant authorization to programs to access PC devices. ([Repo See here](https://github.com/klich3/sonoma-workaround-allow-services))

---

## Installation

```bash
uv venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

### Basic calibration (dashboard)
```bash
python stereo_calibration.py --left 0 --right 1 --rows 6 --cols 9 --square-size 25.0
```

### Asymmetric circles + UV adjustment
```bash
python stereo_calibration.py --pattern-type circles --rows 11 --cols 4 --square-size 6.70 --uv-brightness 0.6 --uv-contrast 0.5
```

### ChArUco (with automatic fallback)
```bash
python stereo_calibration.py --pattern-type charuco --rows 7 --cols 5 --square-size 20.0
```

---

## Interactive session (detection, drawing and auto-capture)

During calibration:

1. The stream is **combined**: **Left (A/Laser)** | **Right (B/UV)** in a 2560×720 window.  
2. The chosen pattern is **detected** on each side.  
3. Corners/points are **drawn** (with horizontal **offset** applied to the right image).  
4. A pair is **auto-captured** when **both** sides detect a pattern (with anti-bounce delay ≈ **1.0 s**).  
5. Temporary images are **saved** and 2D/3D points are **accumulated** for calibration.

**On-screen HUD:**
- `Pairs captured: X/N`
- `Right (B-UV) FPS | Left (A-Laser) FPS`
- `Right/Left pattern: YES/NO`
- `Type: chessboard | circles | charuco`

**Shortcuts:**
- `q` → exit
- `+ / -` → **UV** brightness
- `c / x` → **UV** contrast

---

## Calibration template (download and attributes)



**Generate results** and “SEAL compatible” file:
```bash
python stereo_calibration.py   --left 0 --right 1   --rows 6 --cols 9 --square-size 25.0   --images 15   --output stereo_calibration.txt   --template calib_SEALLITE_template.txt   --dev-id JMS1006207
```

**Attributes** that the script inserts/updates when generating the final file (based on your calibration):
- **Intrinsic** (**A/Laser** camera): `fx, fy, cx, cy`
- **Distortion** (5 coef.): `k1, k2, p1, p2, k3`
- **Forced resolution**: `1280×720`  
- **Metadata**: `CalibrateDate`, `Type: Sychev-calibration`, `SoftVersion: 3.0.0.1116`, `DevID` (if `--dev-id` is passed)

> You can also provide a **base file** (not a template) with reasonable values for SEAL Lite, so that the user can adjust it after their own calibration.

---

## Implementation details (code summary)

- **Capture and low latency**: `CameraStream` uses `CAP_PROP_BUFFERSIZE=1`, queue `maxsize=3`, real FPS calculation, and cascading backends (`CAP_ANY → CAP_AVFOUNDATION → CAP_V4L2`).  
- **Cached detection**: `PatternDetector` limits detections to 0.1s intervals to avoid CPU overload.
- **Patterns**:
- *Chessboard*: `findChessboardCorners` + `cornerSubPix`  
  - *Asymmetric circles*: `SimpleBlobDetector` + `findCirclesGrid(ASYMMETRIC_GRID)` with direct fallback
- *ChArUco*: `CharucoDetector` (DICT_6X6_250); during capture, attempts `matchImagePoints` for matches; if it fails, standard fallback  
- **Auto capture**: when both sides detect a pattern, save `left_XX.jpg` / `right_XX.jpg` (tmp) and accumulate `objpoints/imgpoints`.
- **Calibration**:
- Individual A/B: `cv2.calibrateCamera`  
  - Stereo: `cv2.stereoCalibrate` (criterion `EPS|MAX_ITER`)
- **Output**:
- **Technical** file (`--output`) with `K_right`, `dist_right`, `K_left`, `dist_left`, `R`, `T`  
  - **SEAL compatible** file via `seal_calib_writer.write_new_calibration(...)` and metadata update

---

## Quick diagnosis

Test cameras individually:
```bash
python -c "import stereo_calibration as s; s.test_single_camera(0, 'Laser (A)', 10)"
python -c "import stereo_calibration as s; s.test_single_camera(1, 'UV (B)', 10)"
```
- Shows **backend**, **FOURCC**, **theoretical/actual FPS** and confirms OS permissions.

---

## Notes / Best practices

- Start **A (Laser)** first, wait ~2 s, and then **B (UV)**.
- Fill most of the frame with the pattern and vary angles/distances between captures.
- For **circles**, avoid saturation and pay attention to contrast (important for the blob detector).  
- If **ChArUco** does not detect, there is a **fallback** to *chessboard* (notified by logs).  
- The working resolution is **forced to 1280×720** for consistency with the device software.

---


## **Types of UV brightness/contrast (optional):** if your backend supports floating point values, use `float` instead of `int` in `argparse`:
   ```python
   parser.add_argument("--uv-brightness", type=float, default=-1.0, ...)
   parser.add_argument("--uv-contrast",  type=float, default=-1.0, ...)
   ```

---

## Detected USB Cameras

```text
KYT Camera A:

  N.º de modelo:	UVC Camera VendorID_3141 ProductID_25450
  Unique identifier:	0x145110000c45636a

KYT Camera B:

  N.º de modelo:	UVC Camera VendorID_3141 ProductID_25451
  Unique identifier:	0x145120000c45636b
```

## Cmera configuration

- Laser camera (front): Index 0
- UV camera (inclined): Index 1

## Main scripts

### stereo_calibration.py
Stereo calibration with two cameras:

```bash
# Calibración básica
python stereo_calibration.py

# Calibración con parámetros específicos
python stereo_calibration.py --left 0 --right 1 --rows 6 --cols 9 --square-size 25.0
python stereo_calibration.py --left 1 --right 0 --rows 6 --cols 9 --square-size 6.60 --images 15 --output stereo_calibration.txt --template calibJMS1006207.txt --dev-id JMS1006207
python stereo_calibration.py --left 1 --right 0 --rows 6 --cols 9 --square-size 3 --images 15 --output stereo_calibration.txt --template calibJMS1006207.txt --dev-id JMS1006207 --no-auto-capture

# Usar patrón de círculos
python stereo_calibration.py --pattern-type circles
python stereo_calibration.py --output stereo_calibration.txt --template calibJMS1006207.txt --dev-id JMS1006207 --pattern-type circles --rows 11 --cols 4

# Usar patrón ChArUco
python stereo_calibration.py --pattern-type charuco
python stereo_calibration.py --left 1 --right 0 --images 15 --output stereo_calibration.txt --template calibJMS1006207.txt --dev-id JMS1006207 --pattern-type charuco  --rows 7 --cols 5 --square-size 8.57

# Ajustar brillo de cámara UV
python stereo_calibration.py --uv-brightness 0.5 --uv-contrast 0.5
```

### Calibration methods

1. **Chessboard**:
   - Traditional calibration pattern
   - Robust detection in various lighting conditions
   - Requires a completely visible flat pattern

2. **Asymmetric circles:**
   - Pattern of circles arranged in an asymmetric grid
   - Less sensitive to lens distortions
   - Allows partial detection of the pattern

3. **ChArUco (charuco)**:
   - Combination of ArUco markers and chessboard
   - Greater precision in corner detection
   - Allows detection with partial occlusions

---

# SEAL_CALIB_Builder

Herramienta de calibración estéreo para cámaras duales.

## Características

- Calibración estéreo de dos cámaras simultáneamente
- Detección de patrones: tablero de ajedrez, círculos asimétricos y ChArUco
- Captura automática o manual de imágenes
- Soporte para diferentes diccionarios ArUco
- Configuración de brillo y contraste para la cámara UV
- Control de FPS

## Requisitos

- Python 3.7+
- OpenCV 4.5+
- NumPy

## Instalación

```bash
pip install -r requirement.txt
```

## Uso

### Calibración automática (por defecto)

```bash
python stereo_calibration.py
```

### Calibración manual con barra espaciadora

```bash
python stereo_calibration.py --no-auto-capture
```

### Opciones disponibles

```
--left INDICE          Índice cámara izquierda (A - laser, por defecto 0)
--right INDICE         Índice cámara derecha (B - UV, por defecto 1)
--rows FILAS           Filas tablero (por defecto 6)
--cols COLUMNAS        Columnas tablero (por defecto 9)
--square-size TAMANO   Tamaño cuadrado en mm (por defecto 25.0)
--images NUMERO        Número de pares a capturar (por defecto 15)
--pattern-type TIPO    Tipo de patrón: chessboard, circles, charuco
--no-auto-capture      Deshabilitar captura automática y usar barra espaciadora
--fps FPS              FPS objetivo para las cámaras
--aruco-dict DICCIONARIO  Diccionario ArUco para ChArUco (usar 'auto' para detección automática)
--uv-brightness VALOR  Brillo para la cámara UV
--uv-contrast VALOR    Contraste para la cámara UV
```

### Controles durante la calibración

- **Barra espaciadora**: Capturar imagen (en modo manual)
- **q**: Salir
- **+**: Aumentar brillo de la cámara UV
- **-**: Disminuir brillo de la cámara UV
- **c**: Aumentar contraste de la cámara UV
- **x**: Disminuir contraste de la cámara UV

## Salida

El programa genera dos archivos de calibración:
- `stereo_calibration.txt`: Resultados técnicos de la calibración
- `stereo_calibration_seal.txt`: Archivo de calibración compatible con SEAL

## Notas

- Asegúrate de tener permisos de acceso a la cámara en tu sistema
- Para macOS, otorga permisos de cámara a Terminal/Python en Preferencias del Sistema

---

### Docs

* https://developer.mamezou-tech.com/en/robotics/vision/calibration-pattern/
* https://github.com/chandravaran/Stereo_camera_3D_map_generation