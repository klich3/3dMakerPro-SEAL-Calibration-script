# 📊 Tabla de Correspondencias Gray Code

## ¿Qué es la Tabla Gray Code?

La **tabla de correspondencias Gray Code** es una estructura de datos que representa las **coordenadas del proyector decodificadas** a partir de los patrones Gray Code capturados.

Esta tabla se inserta en el **archivo de calibración SEAL** (líneas 2-156) y es esencial para la reconstrucción 3D precisa.

## Formato de la Tabla

Cada línea de la tabla tiene el formato:

```
número valor_x valor_y
```

Donde:
- `número`: Índice de la muestra (empieza en 2)
- `valor_x`: Coordenada X normalizada del proyector (rango: [-0.01, 0.01])
- `valor_y`: Coordenada Y del proyector en píxeles (rango: [0, 1080])

### Ejemplo del archivo SEAL:

```
1 1280 720
2 -0.007498 24.452629
3 -0.009220 29.683603
4 -0.009219 34.082027
5 -0.009201 38.469620
...
156 0.005692 685.000189
```

## ¿Cómo se Genera?

### 1. Decodificación de Patrones

Durante el **PASO 3** (`--from-gray-images`), el sistema:

1. **Carga las imágenes Gray Code** capturadas (42 patrones por cámara)
2. **Separa por tipo**:
   - Negro/Blanco (calibración)
   - Verticales (coordenadas X del proyector)
   - Verticales invertidos
   - Horizontales (coordenadas Y del proyector)
   - Horizontales invertidos

3. **Decodifica píxel a píxel**:
   ```python
   # Para cada píxel de la cámara:
   for bit in range(num_bits):
       gray_img = patrones_normales[bit]
       inv_img = patrones_invertidos[bit]
       
       # Eliminar luz ambiental comparando
       diff = gray_img - inv_img
       
       # Si diferencia > umbral, bit = 1, sino bit = 0
       if abs(diff) > 20:
           bit_value = 1 if diff > 0 else 0
       
       # Construir código Gray
       decoded += bit_value * (2^bit)
   
   # Convertir Gray Code a binario
   binary = gray_to_binary(decoded)
   
   # binary = coordenada del proyector (0-1920 para X, 0-1080 para Y)
   ```

4. **Genera mapas de correspondencias**:
   - `x_map`: Matriz 1280x720 con coordenadas X del proyector
   - `y_map`: Matriz 1280x720 con coordenadas Y del proyector
   - `valid_mask`: Máscara de píxeles válidos (donde decodificación fue exitosa)

### 2. Muestreo y Normalización

La función `generate_gray_code_table()`:

```python
# 1. Obtener píxeles válidos
valid_y, valid_x = np.where(valid_mask)
# Ejemplo: 820,000 píxeles válidos de 921,600 totales (89% cobertura)

# 2. Submuestrear a ~155 muestras (distribuidas uniformemente)
step = len(valid_x) // 155
indices = np.arange(0, len(valid_x), step)[:155]

# 3. Para cada muestra:
for i, idx in enumerate(indices, start=2):
    pixel_x = valid_x[idx]  # Coordenada X en la cámara
    pixel_y = valid_y[idx]  # Coordenada Y en la cámara
    
    # Coordenadas decodificadas del proyector
    proj_x = x_map[pixel_y, pixel_x]  # Ej: 1245 (píxeles)
    proj_y = y_map[pixel_y, pixel_x]  # Ej: 567 (píxeles)
    
    # Normalizar X al rango [-0.01, 0.01]
    norm_x = (proj_x / 1920.0) * 0.02 - 0.01
    
    # Y se mantiene en píxeles
    norm_y = proj_y
    
    # Generar línea de tabla
    table.append(f"{i} {norm_x:.6f} {norm_y:.6f}")
```

### 3. Inserción en el Archivo SEAL

La tabla se inserta automáticamente después de la línea 1 (resolución):

```python
# Archivo SEAL antes:
"""
1 1280 720
157 1250.71 ... (línea de parámetros intrínsecos)
"""

# Archivo SEAL después:
"""
1 1280 720
2 -0.007498 24.452629
3 -0.009220 29.683603
...
156 0.005692 685.000189
157 1250.71 ... (línea de parámetros intrínsecos)
"""
```

## ¿Para Qué Sirve?

### 1. Reconstrucción 3D Precisa

El sistema SEAL usa esta tabla para:

1. **Mapeo píxel a píxel**: Cada píxel de la cámara sabe exactamente qué píxel del proyector lo iluminó
2. **Triangulación estéreo**: Con las coordenadas del proyector, se puede triangular posición 3D
3. **Corrección de distorsión**: Compensar aberraciones del proyector

### 2. Ventajas vs. Estimación por Homografías

| Aspecto | Método 1 (Homografías) | Método 2 (Gray Code) |
|---------|------------------------|----------------------|
| **Precisión** | ~1-2 píxeles | <0.3 píxeles |
| **Cobertura** | Solo puntos del patrón | 89% del campo visual |
| **Correspondencias** | ~54 puntos | ~820,000 puntos |
| **Tabla Gray Code** | ❌ No incluida | ✅ 155 entradas |
| **Uso recomendado** | Desarrollo/pruebas | Producción |

## Verificación de la Tabla

Para verificar que la tabla se generó correctamente:

```bash
# Contar líneas de la tabla (debe ser ~155)
sed -n '2,156p' stereo_calibration_seal_gray_code.txt | wc -l

# Ver primeras 5 líneas
sed -n '2,6p' stereo_calibration_seal_gray_code.txt

# Verificar formato (debe tener 3 columnas: número, X, Y)
sed -n '2,10p' stereo_calibration_seal_gray_code.txt | awk '{print NF}' | sort -u
# Debe mostrar: 3
```

## Ejemplo de Salida

Al ejecutar `--from-gray-images`, verás:

```
[INFO] Generando tabla de correspondencias Gray Code...
  Píxeles válidos totales: 820000
  Muestras seleccionadas: 155
  Tabla generada con 155 entradas
  Rango X: [-0.009876, 0.008765]
  Rango Y: [24.123456, 685.234567]

[INFO] Insertando tabla de correspondencias Gray Code (155 entradas)...
  Tabla insertada después de línea 2
```

## Parámetros Ajustables

En el código (`stereo_calibration.py`), puedes modificar:

```python
def generate_gray_code_table(x_map, y_map, valid_mask, max_samples=200):
    #                                                 ^^^^^^^^^^^
    # Número máximo de muestras en la tabla (default: 200)
    # El archivo SEAL original tiene 155 (líneas 2-156)
```

Para generar más muestras:

```python
gray_code_table = generate_gray_code_table(
    x_map_left, y_map_left, valid_mask_left,
    max_samples=300  # Generar hasta 300 muestras
)
```

## Preguntas Frecuentes

### ¿Por qué normalizar X pero no Y?

El formato SEAL original muestra:
- **X normalizado**: Rango [-0.01, 0.01] → Independiente de resolución
- **Y en píxeles**: Rango [0-1080] → Valor absoluto

Esto permite escalar horizontalmente manteniendo proporciones verticales.

### ¿Por qué ~155 muestras?

El archivo SEAL de referencia tiene 155 líneas (2-156). Este número:
- Cubre adecuadamente el campo visual
- Mantiene tamaño de archivo manejable
- Es suficiente para interpolación precisa

### ¿Qué pasa si tengo menos píxeles válidos?

Si `valid_mask` tiene pocos píxeles (<200):
```python
if total_valid < max_samples:
    # Usa TODOS los píxeles válidos
    indices = np.arange(total_valid)
```

### ¿Cómo afecta la calidad de captura?

| Calidad | Píxeles válidos | Muestras | Precisión |
|---------|----------------|----------|-----------|
| **Excelente** | >90% (>830k) | 155 | <0.2 px |
| **Buena** | 80-90% (740-830k) | 155 | <0.3 px |
| **Aceptable** | 70-80% (650-740k) | 155 | <0.5 px |
| **Pobre** | <70% (<650k) | <155 | >0.5 px ⚠️ |

**Factores que afectan cobertura**:
- Iluminación ambiental (debe ser oscura)
- Superficie no plana o reflectante
- Proyector desenfocado
- Movimiento durante captura

## Resolución de Problemas

### Error: "No hay píxeles válidos para generar tabla"

```
[WARNING] No hay píxeles válidos para generar tabla
```

**Causa**: Todos los píxeles fueron rechazados durante decodificación.

**Solución**:
1. Verificar que las imágenes Gray Code se capturaron correctamente
2. Revisar iluminación (debe ser oscura)
3. Verificar superficie de proyección (blanca, mate, plana)
4. Aumentar `--gray-delay` a 1.0 segundos

### Advertencia: "Tabla insertada después de línea X" (X ≠ 2)

```
[WARNING] No se encontró línea de resolución, tabla NO insertada
```

**Causa**: El archivo SEAL no tiene formato esperado.

**Solución**:
1. Verificar que el template tiene línea `1 1280 720`
2. Revisar formato del archivo de salida

### Rangos fuera de lo esperado

```
Rango X: [-0.050000, 0.050000]  ← ⚠️ Muy grande
Rango Y: [0.000000, 2000.000000]  ← ⚠️ Fuera de rango
```

**Causa**: Error en normalización o decodificación.

**Solución**:
1. Verificar resolución del proyector (`--projector-width/height`)
2. Revisar número de bits (`--gray-bits`)
3. Validar que las imágenes corresponden a la resolución especificada

## Referencias

- [GRAY_CODE_GUIDE.md](GRAY_CODE_GUIDE.md): Guía completa del proceso
- [PROJECTOR_CALIBRATION.md](PROJECTOR_CALIBRATION.md): Teoría de calibración
- Código fuente: `stereo_calibration.py` líneas 1222-1282 (función `generate_gray_code_table`)
