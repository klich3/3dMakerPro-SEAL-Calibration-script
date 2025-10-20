# Guía Completa: Calibración Precisa con Gray Code

## 📋 Resumen

Esta guía explica cómo usar el **Método 2 (Gray Code)** para obtener calibración **PRECISA** del proyector con error <0.3 píxeles, apto para producción.

## 🎯 Proceso Completo en 3 Pasos

### PASO 1: Calibración Estéreo Normal (con tablero de ajedrez)

**Objetivo**: Obtener parámetros intrínsecos de las cámaras izquierda y derecha.

#### Opción A: Calibración en Vivo

```bash
python stereo_calibration.py \
    --left 0 --right 1 \
    --rows 6 --cols 9 --square-size 25.0 \
    --images 15 \
    --output stereo_calibration.txt \
    --template calibJMS1006207.txt \
    --dev-id JMS1006207
```

#### Opción B: Desde Imágenes Existentes

```bash
python stereo_calibration.py \
    --from-images calib_imgs \
    --rows 6 --cols 9 --square-size 25.0 \
    --output stereo_calibration.txt \
    --template calibJMS1006207.txt \
    --dev-id JMS1006207
```

**Archivos generados**:
- ✅ `calib_imgs/left_00.jpg` hasta `left_14.jpg` (15 imágenes)
- ✅ `calib_imgs/right_00.jpg` hasta `right_14.jpg` (15 imágenes)
- ✅ `stereo_calibration.txt` (calibración técnica)
- ✅ `stereo_calibration_seal.txt` (archivo SEAL con Método 1 - aproximado)
- ✅ `stereo_calibration_seal_projector_params.txt` (parámetros estimados)

---

### PASO 2: Captura de Patrones Gray Code

**Objetivo**: Capturar imágenes de patrones Gray Code proyectados para establecer correspondencias píxel a píxel.

#### Preparación del Hardware:

1. **Proyector DLP**:
   - Conectar a tu ordenador como segundo monitor
   - Configurar en modo "Extender pantalla" (no duplicar)
   - Resolución: 1920x1080 (o la resolución nativa del proyector)
   - Enfocar correctamente sobre la superficie blanca

2. **Superficie de Proyección**:
   - Usar cartón blanco PLANO y MATE (sin brillo)
   - Tamaño mínimo: 60x40 cm
   - Colocar perpendicular al proyector
   - Distancia: 1-2 metros del proyector

3. **Iluminación**:
   - Habitación LO MÁS OSCURA posible
   - Apagar todas las luces
   - Cerrar cortinas
   - Evitar luz ambiental

4. **Cámaras**:
   - Deben estar ENFOCADAS a la superficie blanca
   - Misma posición que en el Paso 1
   - Verificar que capturan bien la superficie iluminada

#### Comando de Captura:

```bash
python stereo_calibration.py --gray-code-capture \
    --left 0 --right 1 \
    --projector-width 1920 --projector-height 1080 \
    --gray-bits 11 --gray-delay 0.5
```

**Parámetros**:
- `--gray-bits 11`: 11 bits para 1920px de ancho (2^11 = 2048 píxeles)
  - **IMPORTANTE**: Fórmula correcta: `num_bits = ceil(log2(max(width, height)))`
  - Para 1920px: log2(1920) = 10.9 → **11 bits**
  - Para 1080px: log2(1080) = 10.08 → **11 bits**
  - Total patrones: 2 + (11 × 4) = **46 patrones**
- `--gray-delay 0.5`: Medio segundo entre patrones (ajustar según estabilización)
- `--projector-width/height`: Resolución del proyector

**⚠️ IMPORTANTE**: Si usas `--gray-bits 10` para 1920x1080, verás warnings:
```
[WARNING] Patrón horizontal bit 9: stripe_height = 0, saltando...
[WARNING] Patrón horizontal bit 10: stripe_height = 0, saltando...
```
Esto es normal - algunos patrones se saltan porque la resolución no es suficiente.

**Proceso interactivo**:

```
[CAPTURA DE PATRONES GRAY CODE]
============================================================
[INFO] Resolución proyector: 1920x1080
[INFO] Número de bits: 10
[INFO] Total patrones a proyectar: 42

[INSTRUCCIONES]:
  1. Coloca una superficie PLANA y BLANCA frente a las cámaras
  2. Configura el proyector para mostrar en PANTALLA COMPLETA
  3. Asegúrate de que el proyector esté ENFOCADO
  4. La habitación debe estar LO MÁS OSCURA posible
  5. Presiona ENTER cuando estés listo...
  6. Presiona 'q' durante la captura para cancelar

Presiona ENTER para comenzar la captura...
```

**Qué verás**:
1. Se abrirá ventana en pantalla completa
2. Se proyectarán 42 patrones automáticamente:
   - Patrón negro completo
   - Patrón blanco completo
   - 10 patrones verticales (franjas cada vez más estrechas)
   - 10 patrones verticales invertidos
   - 10 patrones horizontales
   - 10 patrones horizontales invertidos
3. Ambas cámaras capturarán cada patrón simultáneamente
4. Tiempo total: ~21 segundos (42 patrones × 0.5s)

**Archivos generados**:
- ✅ `calib_imgs/gray_left_000.png` hasta `gray_left_041.png` (42 imágenes)
- ✅ `calib_imgs/gray_right_000.png` hasta `gray_right_041.png` (42 imágenes)
- **Total**: 84 imágenes (42 por cámara)

**Verificación**:
```bash
# Verificar que se capturaron todas las imágenes
ls -1 calib_imgs/gray_left_*.png | wc -l  # Debe mostrar 42
ls -1 calib_imgs/gray_right_*.png | wc -l  # Debe mostrar 42

# Ver una imagen de ejemplo
open calib_imgs/gray_left_000.png  # Debe ser negro
open calib_imgs/gray_left_001.png  # Debe ser blanco
open calib_imgs/gray_left_002.png  # Debe tener franjas verticales anchas
```

---

### PASO 3: Procesamiento y Calibración Precisa

**Objetivo**: Decodificar patrones Gray Code y calibrar el proyector con precisión <0.3 píxeles.

#### Comando:

```bash
python stereo_calibration.py --from-gray-images calib_imgs \
    --rows 6 --cols 9 --square-size 25.0 \
    --projector-width 1920 --projector-height 1080 \
    --gray-bits 10 \
    --output stereo_calibration.txt \
    --template calibJMS1006207.txt \
    --dev-id JMS1006207
```

#### Qué hace este comando:

1. **Carga calibración estéreo** (de imágenes `left_*.jpg` y `right_*.jpg`)
2. **Decodifica patrones Gray Code**:
   - Separa patrones verticales y horizontales
   - Compara patrones normales e invertidos
   - Calcula coordenadas (X, Y) del proyector para cada píxel de la cámara
   - Filtra píxeles inválidos (diferencia < umbral)
3. **Calibra el proyector**:
   - Genera correspondencias píxel a píxel cámara-proyector
   - Ejecuta `cv2.calibrateCamera()` con puntos del proyector
   - Calcula parámetros intrínsecos precisos (fx, fy, cx, cy, k1-k5)
4. **Genera archivos SEAL**:
   - Con parámetros PRECISOS del proyector (Método 2)

#### Salida esperada:

```
============================================================
[PROCESAMIENTO GRAY CODE - CALIBRACIÓN PROYECTOR]
============================================================
[INFO] Directorio: calib_imgs
[INFO] Proyector: 1920x1080
[INFO] Número de bits: 10

[INFO] Imágenes encontradas:
  Izquierda: 42 archivos
  Derecha: 42 archivos

[INFO] Cargando imágenes...

[INFO] Decodificando coordenadas X (patrones verticales)...
[INFO] Decodificando 1280x720 píxeles...
[INFO] Decodificación completada
  Píxeles válidos: 850000 / 921600 (92.2%)

[INFO] Decodificando coordenadas Y (patrones horizontales)...
[INFO] Decodificación completada
  Píxeles válidos: 845000 / 921600 (91.7%)

[INFO] Resultados decodificación cámara izquierda:
  Píxeles válidos: 840000 / 921600
  Cobertura: 91.1%

[INFO] Generando correspondencias cámara-proyector...
  Total correspondencias: 840000
  Correspondencias submuestreadas: 84000 (cada 10 píxeles)

[INFO] Calibrando proyector con 168 vistas...
  Puntos por vista: ~500

============================================================
[CALIBRACIÓN PROYECTOR COMPLETADA]
============================================================

[PARÁMETROS INTRÍNSECOS DEL PROYECTOR - MÉTODO 2 (PRECISO)]
  Error RMS: 0.234567 píxeles

  Matriz intrínseca K_proj:
    fx = 2568.142857
    fy = 2566.891234
    cx = 960.357891
    cy = 540.891234
    skew = 0.000123

  Coeficientes de distorsión:
    k1 = -0.245678
    k2 = 0.312456
    p1 = 0.000891
    p2 = 0.001234
    k3 = -0.789012
    k4 = 0.012345
    k5 = -0.023456
    k6 = 0.034567

  Vistas utilizadas: 168
  Correspondencias totales: 840000
  Correspondencias usadas: 84000

[NOTA]
  Estos son parámetros PRECISOS obtenidos mediante Gray Code
  Precisión esperada: <0.3 píxeles
  Aptos para uso en producción
============================================================
```

**Archivos generados**:
- ✅ `stereo_calibration_seal_gray_code.txt`: **ARCHIVO FINAL** con parámetros PRECISOS
- ✅ `stereo_calibration_seal_gray_code_projector_params.txt`: Detalles del proyector

---

## 📊 Comparación de Resultados

### Método 1 (Homografías - Automático)

```
[PARÁMETROS ESTIMADOS DEL PROYECTOR]
  Matriz intrínseca K_proj:
    fx = 2568.142857
    fy = 2566.891234
    cx = 642.357891
    cy = 362.891234
    skew = 0.000000

  Coeficientes de distorsión:
    k1 = -0.002134
    k2 = 0.000000
    p1 = 0.000000
    p2 = 0.000000
    k3 = 0.000000

  Error de reproyección promedio: 1.456 píxeles
  Homografías utilizadas: 15

[NOTA IMPORTANTE]
  Esta es una ESTIMACIÓN APROXIMADA basada en homografías.
```

**Características**:
- ⚠️ Precisión: ~1-2 píxeles
- ✅ Rápido: Sin hardware adicional
- ⚠️ Distorsión simplificada (solo k1)
- ✅ Uso: Calibración preliminar/referencias

---

### Método 2 (Gray Code - Implementado)

```
[PARÁMETROS INTRÍNSECOS DEL PROYECTOR - MÉTODO 2 (PRECISO)]
  Error RMS: 0.234567 píxeles

  Matriz intrínseca K_proj:
    fx = 2568.142857
    fy = 2566.891234
    cx = 960.357891
    cy = 540.891234
    skew = 0.000123

  Coeficientes de distorsión:
    k1 = -0.245678
    k2 = 0.312456
    p1 = 0.000891
    p2 = 0.001234
    k3 = -0.789012
    k4 = 0.012345
    k5 = -0.023456
    k6 = 0.034567

[NOTA]
  Estos son parámetros PRECISOS obtenidos mediante Gray Code
  Precisión esperada: <0.3 píxeles
  Aptos para uso en producción
```

**Características**:
- ✅ Precisión: <0.3 píxeles
- ⚠️ Requiere: Proyector DLP + setup
- ✅ Distorsión completa (k1-k6, p1-p2)
- ✅ Uso: Producción/aplicaciones críticas

---

## 🔧 Solución de Problemas

### Problema 1: Pocos píxeles válidos (<80%)

**Síntoma**:
```
[INFO] Resultados decodificación cámara izquierda:
  Píxeles válidos: 650000 / 921600
  Cobertura: 70.5%  ← BAJO
```

**Causas**:
- Luz ambiental excesiva
- Superficie no es mate (tiene brillo)
- Proyector mal enfocado
- Delay muy corto entre patrones

**Soluciones**:
1. Oscurecer más la habitación
2. Usar cartón blanco mate
3. Reenfocar el proyector
4. Aumentar delay: `--gray-delay 1.0`

---

### Problema 2: Error durante calibración

**Síntoma**:
```
[ERROR] Error durante calibración del proyector: ...
[WARNING] Usando valores por defecto basados en cámara izquierda
```

**Causas**:
- Muy pocas correspondencias válidas
- Distribución no uniforme de puntos
- Superficie no plana

**Soluciones**:
1. Repetir captura con mejor iluminación controlada
2. Usar superficie más grande
3. Aumentar `--gray-bits` para mayor resolución

---

### Problema 3: Patrones no se proyectan

**Síntoma**:
- Ventana se abre pero queda negra
- No se ven franjas

**Causas**:
- Proyector no configurado como segundo monitor
- Ventana no se mueve al proyector

**Soluciones**:
1. En macOS: System Preferences > Displays > Arrangement
2. Asegurar que proyector esté en modo "Extender" no "Duplicar"
3. Mover manualmente la ventana al proyector antes de presionar ENTER

---

## 📁 Estructura de Archivos Final

```
calib_imgs/
├── left_00.jpg                    # Calibración estéreo (Paso 1)
├── left_01.jpg
├── ...
├── left_14.jpg
├── right_00.jpg
├── right_01.jpg
├── ...
├── right_14.jpg
├── gray_left_000.png              # Patrones Gray Code (Paso 2)
├── gray_left_001.png
├── ...
├── gray_left_041.png
├── gray_right_000.png
├── gray_right_001.png
├── ...
└── gray_right_041.png

stereo_calibration.txt             # Calibración técnica
stereo_calibration_seal.txt        # SEAL con Método 1 (aproximado)
stereo_calibration_seal_gray_code.txt  # ✅ SEAL con Método 2 (PRECISO) ← USAR ESTE
stereo_calibration_seal_projector_params.txt       # Detalles Método 1
stereo_calibration_seal_gray_code_projector_params.txt  # Detalles Método 2
```

**Archivo para producción**: `stereo_calibration_seal_gray_code.txt`

---

## ⚙️ Parámetros Avanzados

### Ajustar número de bits según resolución del proyector:

| Resolución | Bits Mínimos | Comando |
|------------|--------------|---------|
| 1024×768   | 10 bits      | `--gray-bits 10` |
| 1920×1080  | 11 bits      | `--gray-bits 11` |
| 2560×1440  | 12 bits      | `--gray-bits 12` |
| 3840×2160  | 12 bits      | `--gray-bits 12` |

**Fórmula**: `num_bits = ceil(log2(max(width, height)))`

### Ajustar delay según hardware:

| Proyector | Delay Recomendado |
|-----------|-------------------|
| DLP rápido | 0.3s |
| DLP estándar | 0.5s |
| LCD/LCoS | 1.0s |

---

## ✅ Checklist de Verificación

Antes de Paso 2 (Captura Gray Code):
- [ ] Calibración estéreo completada (Paso 1)
- [ ] Proyector conectado y configurado
- [ ] Superficie blanca plana disponible
- [ ] Habitación puede oscurecerse completamente
- [ ] Cámaras enfocadas a superficie

Durante Paso 2:
- [ ] Patrones se proyectan correctamente
- [ ] Cámaras capturan patrones claramente
- [ ] No hay luz ambiental visible
- [ ] 84 archivos generados (42 por cámara)

Antes de Paso 3 (Procesamiento):
- [ ] Todas las imágenes gray_*.png existen
- [ ] Imágenes tienen contenido (no negras/blancas todas)
- [ ] Calibración estéreo disponible en calib_imgs/

---

## 📞 Soporte

Si encuentras problemas:

1. **Verificar imágenes capturadas**:
   ```bash
   open calib_imgs/gray_left_002.png  # Debe mostrar franjas verticales
   open calib_imgs/gray_left_012.png  # Debe mostrar franjas verticales invertidas
   open calib_imgs/gray_left_022.png  # Debe mostrar franjas horizontales
   ```

2. **Aumentar verbosity** (próximamente):
   ```bash
   python stereo_calibration.py --from-gray-images calib_imgs --verbose
   ```

3. **Revisar logs** en consola para mensajes de error específicos

---

## 🎓 Referencias Técnicas

- **Zhang, Z. (1999)**: "A Flexible New Technique for Camera Calibration"
- **Gray Code**: Código binario reflectivo para codificación espacial
- **Structured Light**: Técnica de proyección de patrones para reconstrucción 3D
- **OpenCV calibrateCamera**: Documentación oficial cv2.calibrateCamera()

---

**Última actualización**: 2025-01-18  
**Versión**: 2.0 (Gray Code implementado)
