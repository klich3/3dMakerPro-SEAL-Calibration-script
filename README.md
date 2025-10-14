# 3DMakerpro Seal Lite re-calibration

> Herramientas para la **calibración estéreo de cámaras** y **reconstrucción 3D** con láser y UV, desarrolladas de forma independiente para el escáner **3DMakerPro SEAL Lite**.

- 🔬 **Compatibilidad**: probado **exclusivamente con 3DMakerPro SEAL Lite**.  
- 🙅‍♂️ Proyecto **no afiliado** con 3DMakerPro ni distribuidores.

---

## Contexto / Prehistoria

Tras un uso normal y cuidadoso del **3DMakerPro SEAL Lite**, el escáner empezó a producir **líneas horizontales** al intentar capturar.  
Se intentaron, sin éxito:

- Actualizaciones de software en **macOS** y **Windows**  
- Soporte del **servicio técnico oficial**  
- Contacto con el **administrador** del grupo de Facebook

Frustrado por la falta de solución (aun con garantía), desarrollé esta **herramienta propia de calibración** basada en **OpenCV** para recuperar la funcionalidad y entender mejor el dispositivo.

---

## ⚠️ Aviso y Descargo de Responsabilidad

Este software se ofrece **tal cual**, **sin garantía** de ningún tipo.

- Su mal uso puede **anular la garantía** del dispositivo.  
- El autor **no se responsabiliza** de daños, pérdidas o fallos derivados del uso o mal uso.  
- Cada usuario es **plenamente responsable** de su ejecución y consecuencias.

**Úsalo bajo tu propia responsabilidad y conciencia.**

---

## 🧩 Descripción técnica

Herramientas de calibración estéreo y reconstrucción 3D con soporte para patrones:

- **Tablero de ajedrez (chessboard)**
- **Círculos asimétricos**
- **ChArUco (ArUco + Chessboard)**

**Cámaras (SEAL Lite):**
- Cámara **A**: **Laser** (frontal) → **índice 0**
- Cámara **B**: **UV** (inclinada) → **índice 1**

---

## Permisos de cámara en macOS desde terminal

Si ves:
```
OpenCV: not authorized to capture video (status 0)
```
Otorga permisos a **Terminal/Python** en  
**Preferencias del Sistema → Seguridad y privacidad → Privacidad → Cámara**

Forzar el prompt:
```bash
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('Cam abierta:', cap.isOpened()); cap.release()"
```

Si no en mi perfil hay un repositorio para una herramienta de poder ceder autorización a los programas para poder acceder a dispositivos del pc.

---

## 📦 Instalación

```bash
uv venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Uso rápido

### Calibración básica (tablero)
```bash
python stereo_calibration.py --left 0 --right 1 --rows 6 --cols 9 --square-size 25.0
```

### Círculos asimétricos + ajuste UV
```bash
python stereo_calibration.py --pattern-type circles --rows 11 --cols 4 --square-size 6.70 --uv-brightness 0.6 --uv-contrast 0.5
```

### ChArUco (con fallback automático)
```bash
python stereo_calibration.py --pattern-type charuco --rows 7 --cols 5 --square-size 20.0
```

---

## Sesión interactiva (detección, dibujo y autocaptura)

Durante la calibración:

1. Se **combina** el stream: **Izq (A/Laser)** | **Der (B/UV)** en una ventana 2560×720.  
2. Se **detecta** el patrón elegido en cada lado.  
3. Se **dibujan** esquinas/puntos (con **offset** horizontal aplicado en la imagen derecha).  
4. Se **autocaptura** un par cuando **ambos** lados detectan patrón (con delay ≈ **1.0 s** anti-rebote).  
5. Se **guardan** imágenes temporales y se **acumulan** puntos 2D/3D para calibrar.

**HUD on-screen:**
- `Pares capturados: X/N`
- `Derecha (B-UV) FPS | Izquierda (A-Laser) FPS`
- `Patrón Derecha/Izquierda: SI/NO`
- `Tipo: chessboard | circles | charuco`

**Atajos:**
- `q` → salir  
- `+ / -` → brillo **UV**  
- `c / x` → contraste **UV**

---

## Plantilla de calibración (descarga y atributos)



**Generar resultados** y archivo “SEAL compatible”:
```bash
python stereo_calibration.py   --left 0 --right 1   --rows 6 --cols 9 --square-size 25.0   --images 15   --output stereo_calibration.txt   --template calib_SEALLITE_template.txt   --dev-id JMS1006207
```

**Atributos** que el script inserta/actualiza al generar el archivo final (a partir de tu calibración):
- **Intrínsecos** (cámara **A/Laser**): `fx, fy, cx, cy`  
- **Distorsión** (5 coef.): `k1, k2, p1, p2, k3`  
- **Resolución forzada**: `1280×720`  
- **Metadatos**: `CalibrateDate`, `Type: Sychev-calibration`, `SoftVersion: 3.0.0.1116`, `DevID` (si se pasa `--dev-id`)

> Puedes ofrecer además un **archivo base** (no plantilla) con valores razonables para SEAL Lite, para que el usuario lo ajuste tras su propia calibración.

---

## Detalles de implementación (resumen del código)

- **Captura y latencia baja**: `CameraStream` usa `CAP_PROP_BUFFERSIZE=1`, cola `maxsize=3`, cálculo de FPS real y backends en cascada (`CAP_ANY → CAP_AVFOUNDATION → CAP_V4L2`).  
- **Detección con caché**: `PatternDetector` limita detecciones a intervalos de 0.1s para no sobrecargar CPU.  
- **Patrones**:
  - *Chessboard*: `findChessboardCorners` + `cornerSubPix`  
  - *Circles asimétricos*: `SimpleBlobDetector` + `findCirclesGrid(ASYMMETRIC_GRID)` con fallback directo  
  - *ChArUco*: `CharucoDetector` (DICT_6X6_250); en captura, intenta `matchImagePoints` para correspondencias; si falla, fallback estándar  
- **Autocaptura**: cuando ambos lados detectan patrón, guarda `left_XX.jpg` / `right_XX.jpg` (tmp) y acumula `objpoints/imgpoints`.  
- **Calibración**:  
  - Individual A/B: `cv2.calibrateCamera`  
  - Estéreo: `cv2.stereoCalibrate` (criterio `EPS|MAX_ITER`)  
- **Salida**:  
  - Archivo **técnico** (`--output`) con `K_right`, `dist_right`, `K_left`, `dist_left`, `R`, `T`  
  - Archivo **SEAL compatible** vía `seal_calib_writer.write_new_calibration(...)` y actualización de metadatos

---

## 🔧 Diagnóstico rápido

Probar cámaras individualmente:
```bash
python -c "import stereo_calibration as s; s.test_single_camera(0, 'Laser (A)', 10)"
python -c "import stereo_calibration as s; s.test_single_camera(1, 'UV (B)', 10)"
```
- Muestra **backend**, **FOURCC**, **FPS teórico/real** y confirma permisos del SO.

---

## 🧷 Notas / Buenas prácticas

- Inicia primero **A (Laser)**, espera ~2 s y luego **B (UV)**.  
- Ocupa gran parte del encuadre con el patrón y varía ángulos/distancias entre capturas.  
- Para **circles**, evita saturación y cuida el contraste (importante para el blob detector).  
- Si **ChArUco** no detecta, hay **fallback** a *chessboard* (se avisa por logs).  
- La resolución de trabajo se **fuerza a 1280×720** para consistencia con el software del dispositivo.

---


## **Tipos de brillo/contraste UV (opcional):** si tu backend soporta valores flotantes, en `argparse` usa `float` en lugar de `int`:
   ```python
   parser.add_argument("--uv-brightness", type=float, default=-1.0, ...)
   parser.add_argument("--uv-contrast",  type=float, default=-1.0, ...)
   ```

---

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

---

### Docs

* https://developer.mamezou-tech.com/en/robotics/vision/calibration-pattern/