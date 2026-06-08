# Modelo Oculto de Markov (HMM) — Predicción Climática

Simulación de condiciones climáticas a partir de observaciones de humedad ambiental.  
Asignatura: Simulación — Módulo 2

---

## Descripción

Este proyecto implementa un Modelo Oculto de Markov (HMM) de 3 estados para simular
la secuencia de condiciones climáticas diarias (Soleado, Nublado, Lluvioso) a partir
de niveles de humedad observables (Baja, Media, Alta).

El modelo parte de la idea de que el clima real no es directamente observable,
pero sí podemos medir la humedad ambiental, que está estadísticamente relacionada
con el estado climático subyacente.

---

## Requisitos

- Python 3.8 o superior
- NumPy
- Matplotlib

Instalar dependencias:

```
pip install numpy matplotlib
```

---

## Cómo ejecutar

```
python hmm_clima.py
```

El programa imprime en consola las primeras 10 observaciones, las estadísticas
completas de los 30 días y guarda 2 gráficas en la misma carpeta.

---

## Parámetros del modelo

### Estados ocultos (clima real)

| Estado    | Descripción                        |
|-----------|------------------------------------|
| Soleado   | Día despejado, baja probabilidad de lluvia |
| Nublado   | Cielo cubierto, estado intermedio  |
| Lluvioso  | Precipitación activa               |

### Observaciones (humedad medible)

| Observación   | Descripción                  |
|---------------|------------------------------|
| Humedad Baja  | Ambiente seco                |
| Humedad Media | Condición intermedia         |
| Humedad Alta  | Ambiente húmedo              |

### Distribución inicial π

| Estado    | Probabilidad |
|-----------|--------------|
| Soleado   | 0.60         |
| Nublado   | 0.30         |
| Lluvioso  | 0.10         |

### Matriz de transición A

|            | → Soleado | → Nublado | → Lluvioso |
|------------|-----------|-----------|------------|
| Soleado    | 0.70      | 0.20      | 0.10       |
| Nublado    | 0.30      | 0.40      | 0.30       |
| Lluvioso   | 0.20      | 0.30      | 0.50       |

### Matriz de emisión B

|            | Baja | Media | Alta |
|------------|------|-------|------|
| Soleado    | 0.70 | 0.20  | 0.10 |
| Nublado    | 0.20 | 0.50  | 0.30 |
| Lluvioso   | 0.10 | 0.30  | 0.60 |

---

## Estructura del código

```
hmm_clima.py
│
├── class HMM
│   ├── __init__()                     → Inicializa parámetros π, A, B
│   ├── simular(n_pasos)               → Genera secuencia de estados y observaciones
│   ├── calcular_estadisticas()        → Días por estado, % humedad, matriz de confusión
│   └── imprimir_primeras_observaciones() → Muestra tabla en consola
│
├── CONFIGURACIÓN                      → Definición de π, A, B y nombres
├── EJECUCIÓN                          → Simulación de 30 días
├── ESTADÍSTICAS EN CONSOLA            → Impresión de resultados
└── VISUALIZACIÓN
    ├── Figura 1: secuencia temporal   → simulacion_hmm.png
    └── Figura 2: estadísticas         → estadisticas_hmm.png
```

---

## Salidas generadas

| Archivo               | Contenido                                                        |
|-----------------------|------------------------------------------------------------------|
| `simulacion_hmm.png`  | Secuencia diaria de estados climáticos y niveles de humedad (barras de color por día) |
| `estadisticas_hmm.png`| Gráfico de torta de estados, barras de humedad y mapa de calor de la matriz de confusión |

---

## Salida en consola (ejemplo)

```
====================================================
  PRIMERAS 10 OBSERVACIONES DE LA SIMULACIÓN
====================================================
  Día  1: Soleado     →  Humedad Baja
  Día  2: Soleado     →  Humedad Baja
  ...
====================================================

====================================================
  ESTADÍSTICAS DE LA SIMULACIÓN (30 días)
====================================================

  Días en cada estado climático:
    Soleado   : 17 días ( 56.7%)  █████████████████
    Nublado   :  8 días ( 26.7%)  ████████
    Lluvioso  :  5 días ( 16.7%)  █████

  Porcentaje de humedad observada:
    Humedad Baja   :  50.0%
    Humedad Media  :  30.0%
    Humedad Alta   :  20.0%
```

---

## Conceptos clave

**¿Qué es un HMM?**  
Un Modelo Oculto de Markov es un sistema estocástico donde la variable de interés
(el estado) no es directamente observable. Solo podemos ver las emisiones (observaciones)
que genera ese estado. El modelo aprende la relación entre ambos.

**Proceso generativo:**
1. Se elige el estado inicial según π.
2. En cada paso se emite una observación según B[estado actual].
3. El sistema transita al siguiente estado según A[estado actual].

**Reproducibilidad:**  
El código usa `np.random.seed(42)` para que los resultados sean idénticos en cada ejecución.

---

## Autor

Estudiante de Ingeniería de Software y Datos  
Asignatura: Simulación — Módulo 2
