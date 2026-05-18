# 🌤️ Predicción Climática con Modelo Oculto de Markov (HMM)

Simulación de un Hidden Markov Model que predice el estado del clima (Soleado, Nublado, Lluvioso) a partir de observaciones de humedad atmosférica. Implementado en Python puro con `numpy` y visualizado con `matplotlib`.

---

## 📋 Descripción del Modelo

Un **Modelo Oculto de Markov** asume que existe una secuencia de estados ocultos (el clima real) que no podemos observar directamente, y una secuencia de observaciones (la humedad) que sí podemos medir. El modelo aprende las relaciones probabilísticas entre ambas.

```
Estado oculto:    Soleado → Soleado → Nublado → Lluvioso → ...
                     ↓          ↓         ↓          ↓
Observación:      H. Baja  → H. Baja → H. Media → H. Alta → ...
```

### Estados ocultos
| Estado | Color en gráfico |
|--------|-----------------|
| ☀️ Soleado | Amarillo `#FFD700` |
| ☁️ Nublado | Azul claro `#87CEEB` |
| 🌧️ Lluvioso | Azul oscuro `#1E4D8C` |

### Observaciones
| Observación | Descripción |
|------------|-------------|
| 💧 Humedad Baja | Ambiente seco |
| 💧💧 Humedad Media | Ambiente moderado |
| 💧💧💧 Humedad Alta | Ambiente húmedo |

---

## 🔢 Parámetros del Modelo

### Matriz de Transición `A`
Probabilidad de pasar de un estado climático a otro al día siguiente.

|             | Soleado | Nublado | Lluvioso |
|-------------|:-------:|:-------:|:--------:|
| **Soleado** | 0.7     | 0.2     | 0.1      |
| **Nublado** | 0.3     | 0.4     | 0.3      |
| **Lluvioso**| 0.2     | 0.3     | 0.5      |

### Matriz de Emisión `B`
Probabilidad de observar cierto nivel de humedad dado el estado climático.

|             | H. Baja | H. Media | H. Alta |
|-------------|:-------:|:--------:|:-------:|
| **Soleado** | 0.7     | 0.2      | 0.1     |
| **Nublado** | 0.2     | 0.5      | 0.3     |
| **Lluvioso**| 0.1     | 0.3      | 0.6     |

### Distribución Inicial `π`
| Soleado | Nublado | Lluvioso |
|:-------:|:-------:|:--------:|
| 0.6     | 0.3     | 0.1      |

---

## 📊 Salida del Programa

El script genera:

1. **Consola** — Primeros 10 días con formato `Día X: Estado → Observación`
2. **Estadísticas** — Conteo de días por estado y distribución de humedad
3. **Matriz de confusión** — Cruce entre estados ocultos y observaciones
4. **Gráfico** (`hmm_clima.png`) — Dos gráficos de barras alineados temporalmente

### Ejemplo de salida en consola
```
Día  1: Soleado    →  Humedad Alta
Día  2: Nublado    →  Humedad Media
Día  3: Soleado    →  Humedad Baja
...
```

---

## 🚀 Instalación y Uso

### Requisitos
```
python >= 3.8
numpy
matplotlib
```

### Instalación de dependencias
```bash
pip install numpy matplotlib
```

### Ejecución
```bash
python hmm_clima.py
```

### Google Colab / Jupyter Notebook
Copia y pega el contenido de `hmm_clima.py` directamente en una celda — `numpy` y `matplotlib` ya vienen preinstalados.

---

## 🗂️ Estructura del Proyecto

```
📦 hmm-clima/
 ┣ 📄 hmm_clima.py      # Código principal
 ┣ 📄 README.md         # Este archivo
 ┗ 📊 hmm_clima.png     # Gráfico generado (se crea al ejecutar)
```

---

## 🧠 Cómo Funciona el Código

```python
# 1. Definir el modelo
hmm = HMM(states, observations, trans_matrix, emis_matrix, init_dist)

# 2. Simular T días
hidden_seq, obs_seq = hmm.simulate(T=30)

# 3. El método simulate() en cada paso t:
#    a) Transiciona el estado:   estado_t ~ A[estado_{t-1}]
#    b) Emite una observación:   obs_t    ~ B[estado_t]
```

### Clase `HMM`
| Método | Descripción |
|--------|-------------|
| `__init__()` | Inicializa matrices A, B y distribución π |
| `sample_state(probs)` | Muestrea un índice dado un vector de probabilidades |
| `simulate(T)` | Genera secuencias de longitud T |

---

## 📈 Ejemplo de Resultados (semilla 42, 30 días)

| Estado | Días | Porcentaje |
|--------|:----:|:----------:|
| ☀️ Soleado | 22 | 73.3 % |
| ☁️ Nublado | 4 | 13.3 % |
| 🌧️ Lluvioso | 4 | 13.3 % |

**Matriz de confusión:**
```
               H. Baja   H. Media   H. Alta
Soleado           15         3         4
Nublado            1         3         0
Lluvioso           0         0         4
```

---

## 📚 Referencias

- Rabiner, L. R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition.* Proceedings of the IEEE.
- Jurafsky, D. & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). [web.stanford.edu/~jurafsky/slp3](https://web.stanford.edu/~jurafsky/slp3/)

---

## 📝 Licencia

Este proyecto está bajo la licencia MIT. Libre para usar, modificar y distribuir.
