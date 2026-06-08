# ============================================================
# Modelo Oculto de Markov (HMM) - Predicción Climática
# Basado en observaciones de humedad ambiental
#
# Autor: Estudiante de Ingeniería de Software y Datos
# Asignatura: Simulación - Módulo 2
# Descripción: Simula la secuencia de condiciones climáticas
#              (estados ocultos) a partir de niveles de humedad
#              (observaciones), usando un HMM de 3 estados.
# ============================================================

# --- Importación de librerías ---
import numpy as np                          # Operaciones numéricas y generación aleatoria
import matplotlib.pyplot as plt             # Visualización de gráficas
import matplotlib.patches as mpatches      # Parches para leyendas personalizadas
from collections import Counter            # Conteo de frecuencias
import warnings
warnings.filterwarnings('ignore')

# --- Semilla para reproducibilidad de resultados ---
np.random.seed(42)

# ============================================================
# CLASE HMM: Modelo Oculto de Markov
# ============================================================

class HMM:
    """
    Implementación de un Modelo Oculto de Markov (HMM).

    En un HMM existen dos tipos de variables:
      - Estados ocultos: el sistema real que no podemos observar directamente
        (en este caso, la condición climática: Soleado, Nublado, Lluvioso).
      - Observaciones: señales que sí podemos medir y que están relacionadas
        con los estados ocultos (en este caso, el nivel de humedad: Baja, Media, Alta).

    El modelo se define por tres parámetros:
      - π (prob_inicial): distribución de probabilidad sobre el estado inicial.
      - A (matriz_transicion): probabilidades de pasar de un estado a otro.
      - B (matriz_emision): probabilidades de emitir cada observación dado un estado.
    """

    def __init__(self, estados, observaciones, prob_inicial,
                 matriz_transicion, matriz_emision):
        """
        Constructor: inicializa el HMM con sus parámetros fundamentales.

        Args:
            estados (list[str])         : Nombres de los estados ocultos.
            observaciones (list[str])   : Nombres de las posibles observaciones.
            prob_inicial (list[float])  : Distribución inicial π sobre los estados.
            matriz_transicion (list)    : Matriz A (n_estados × n_estados).
            matriz_emision (list)       : Matriz B (n_estados × n_observaciones).
        """
        # Almacenar nombres de estados y observaciones
        self.estados = estados
        self.observaciones = observaciones

        # Convertir listas a arrays numpy para operar eficientemente
        self.prob_inicial = np.array(prob_inicial)
        self.matriz_transicion = np.array(matriz_transicion)
        self.matriz_emision = np.array(matriz_emision)

        # Dimensiones del modelo
        self.n_estados = len(estados)
        self.n_observaciones = len(observaciones)

    # ----------------------------------------------------------
    # MÉTODO: Simular una secuencia de estados y observaciones
    # ----------------------------------------------------------
    def simular(self, n_pasos):
        """
        Genera una secuencia de estados ocultos y observaciones siguiendo
        el proceso generativo del HMM:
          1. Seleccionar el estado inicial t=0 según la distribución π.
          2. En cada paso t:
             a. Emitir una observación según B[estado_actual].
             b. Transitar al siguiente estado según A[estado_actual].

        Args:
            n_pasos (int): Número de pasos temporales (días a simular).

        Returns:
            estados_seq (list[int]): Índices de estados ocultos generados.
            obs_seq (list[int])    : Índices de observaciones generadas.
        """
        estados_seq = []    # Almacena los estados ocultos simulados
        obs_seq = []        # Almacena las observaciones simuladas

        # --- Paso 1: Selección del estado inicial según π ---
        estado_actual = np.random.choice(self.n_estados, p=self.prob_inicial)

        for _ in range(n_pasos):
            # --- Paso 2a: Registrar el estado oculto actual ---
            estados_seq.append(estado_actual)

            # --- Paso 2b: Emitir observación según B[estado_actual] ---
            obs = np.random.choice(
                self.n_observaciones,
                p=self.matriz_emision[estado_actual]
            )
            obs_seq.append(obs)

            # --- Paso 2c: Transitar al siguiente estado según A[estado_actual] ---
            estado_actual = np.random.choice(
                self.n_estados,
                p=self.matriz_transicion[estado_actual]
            )

        return estados_seq, obs_seq

    # ----------------------------------------------------------
    # MÉTODO: Calcular estadísticas de la simulación
    # ----------------------------------------------------------
    def calcular_estadisticas(self, estados_seq, obs_seq):
        """
        Calcula métricas descriptivas sobre la secuencia simulada.

        Args:
            estados_seq (list[int]): Secuencia de índices de estados ocultos.
            obs_seq (list[int])    : Secuencia de índices de observaciones.

        Returns:
            dict: Diccionario con las siguientes claves:
                  - 'dias_por_estado'  : días totales en cada estado climático.
                  - 'pct_humedad'      : porcentaje de cada nivel de humedad.
                  - 'matriz_confusion' : tabla cruzada estado × observación.
        """
        n = len(estados_seq)

        # --- Contar días en cada estado climático ---
        conteo_estados = Counter(estados_seq)
        dias_por_estado = {
            self.estados[i]: conteo_estados.get(i, 0)
            for i in range(self.n_estados)
        }

        # --- Calcular porcentaje de cada tipo de humedad ---
        conteo_obs = Counter(obs_seq)
        pct_humedad = {
            self.observaciones[i]: (conteo_obs.get(i, 0) / n) * 100
            for i in range(self.n_observaciones)
        }

        # --- Construir la matriz de confusión (estado × observación) ---
        # Cada celda [i][j] cuenta cuántas veces el estado i emitió la observación j
        matriz_confusion = np.zeros(
            (self.n_estados, self.n_observaciones), dtype=int
        )
        for estado, obs in zip(estados_seq, obs_seq):
            matriz_confusion[estado][obs] += 1

        return {
            'dias_por_estado': dias_por_estado,
            'pct_humedad': pct_humedad,
            'matriz_confusion': matriz_confusion
        }

    # ----------------------------------------------------------
    # MÉTODO: Imprimir primeras N observaciones formateadas
    # ----------------------------------------------------------
    def imprimir_primeras_observaciones(self, estados_seq, obs_seq, n=10):
        """
        Muestra en consola las primeras N observaciones en el formato:
        'Día X: Estado → Observación'.

        Args:
            estados_seq (list[int]): Secuencia de estados.
            obs_seq (list[int])    : Secuencia de observaciones.
            n (int)                : Número de filas a imprimir (default=10).
        """
        print("\n" + "=" * 52)
        print(f"  PRIMERAS {n} OBSERVACIONES DE LA SIMULACIÓN")
        print("=" * 52)
        for i in range(min(n, len(estados_seq))):
            estado_nombre = self.estados[estados_seq[i]]
            obs_nombre = self.observaciones[obs_seq[i]]
            print(f"  Día {i + 1:2d}: {estado_nombre:10s}  →  {obs_nombre}")
        print("=" * 52)


# ============================================================
# CONFIGURACIÓN DE LOS PARÁMETROS DEL MODELO
# ============================================================

# --- Estados ocultos: condiciones climáticas no observables directamente ---
estados = ['Soleado', 'Nublado', 'Lluvioso']

# --- Observaciones: niveles de humedad medibles ---
observaciones = ['Humedad Baja', 'Humedad Media', 'Humedad Alta']

# --- Distribución inicial π ---
# Probabilidad de que el primer día sea Soleado (0.6), Nublado (0.3) o Lluvioso (0.1)
prob_inicial = [0.6, 0.3, 0.1]

# --- Matriz de transición A (3×3) ---
# A[i][j] = P(estado_{t+1} = j | estado_t = i)
# Cada fila debe sumar 1.0
matriz_transicion = [
    [0.7, 0.2, 0.1],   # Desde Soleado  → Soleado:0.7, Nublado:0.2, Lluvioso:0.1
    [0.3, 0.4, 0.3],   # Desde Nublado  → Soleado:0.3, Nublado:0.4, Lluvioso:0.3
    [0.2, 0.3, 0.5],   # Desde Lluvioso → Soleado:0.2, Nublado:0.3, Lluvioso:0.5
]

# --- Matriz de emisión B (3×3) ---
# B[i][k] = P(observacion = k | estado = i)
# Cada fila debe sumar 1.0
matriz_emision = [
    [0.7, 0.2, 0.1],   # Soleado  → Humedad Baja:0.7, Media:0.2, Alta:0.1
    [0.2, 0.5, 0.3],   # Nublado  → Humedad Baja:0.2, Media:0.5, Alta:0.3
    [0.1, 0.3, 0.6],   # Lluvioso → Humedad Baja:0.1, Media:0.3, Alta:0.6
]

# ============================================================
# INSTANCIACIÓN Y EJECUCIÓN DEL MODELO
# ============================================================

# --- Crear instancia del HMM con los parámetros definidos ---
hmm = HMM(estados, observaciones, prob_inicial, matriz_transicion, matriz_emision)

# --- Ejecutar simulación de 30 días ---
N_DIAS = 30
estados_seq, obs_seq = hmm.simular(N_DIAS)

# --- Mostrar las primeras 10 observaciones ---
hmm.imprimir_primeras_observaciones(estados_seq, obs_seq, n=10)

# --- Obtener estadísticas de la simulación completa ---
stats = hmm.calcular_estadisticas(estados_seq, obs_seq)

# ============================================================
# IMPRESIÓN DE ESTADÍSTICAS EN CONSOLA
# ============================================================

print("\n" + "=" * 52)
print(f"  ESTADÍSTICAS DE LA SIMULACIÓN ({N_DIAS} días)")
print("=" * 52)

# --- Días en cada estado climático ---
print("\n  Días en cada estado climático:")
for estado, dias in stats['dias_por_estado'].items():
    barra = "█" * dias
    print(f"    {estado:10s}: {dias:2d} días ({dias / N_DIAS * 100:5.1f}%)  {barra}")

# --- Porcentaje de humedad observada ---
print("\n  Porcentaje de humedad observada:")
for obs, pct in stats['pct_humedad'].items():
    print(f"    {obs:15s}: {pct:5.1f}%")

# --- Matriz de confusión ---
print("\n  Matriz de confusión (Estado × Observación):")
print(f"    {'':12s}", end="")
for obs in observaciones:
    etq = obs.replace("Humedad ", "")
    print(f"  {etq:>8s}", end="")
print()
print(f"    {'':12s}" + "-" * 33)
for i, estado in enumerate(estados):
    print(f"    {estado:12s}", end="")
    for j in range(len(observaciones)):
        print(f"  {stats['matriz_confusion'][i][j]:>8d}", end="")
    print()
print("=" * 52)

# ============================================================
# VISUALIZACIÓN - GRÁFICOS CON MATPLOTLIB
# ============================================================

# --- Paleta de colores distintivos según especificación ---
COLORES_ESTADOS = {
    'Soleado':  '#FFD700',    # Amarillo dorado
    'Nublado':  '#87CEEB',    # Azul cielo (azul claro)
    'Lluvioso': '#1E3A5F',    # Azul marino (azul oscuro)
}

COLORES_HUMEDAD = {
    'Humedad Baja':  '#FFA040',   # Naranja cálido (baja humedad)
    'Humedad Media': '#6495ED',   # Azul aciano (humedad media)
    'Humedad Alta':  '#00008B',   # Azul oscuro (alta humedad)
}

# --- Preparar eje X y colores para cada barra ---
dias = list(range(1, N_DIAS + 1))
colores_estados_barras = [COLORES_ESTADOS[estados[e]] for e in estados_seq]
colores_obs_barras     = [COLORES_HUMEDAD[observaciones[o]] for o in obs_seq]

# ============================================================
# FIGURA 1: Secuencia temporal de estados y observaciones
# Ambos gráficos de barras alineados temporalmente (eje X compartido)
# ============================================================

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(18, 9), sharex=True,
    gridspec_kw={'hspace': 0.08}
)
fig.suptitle(
    'Simulación HMM — Predicción Climática basada en Humedad (30 días)',
    fontsize=15, fontweight='bold', y=0.99
)

# ---------- Subgráfico 1: Estados ocultos (condición climática) ----------
ax1.bar(dias, [1] * N_DIAS, color=colores_estados_barras,
        edgecolor='white', linewidth=0.6, width=0.9)
ax1.set_ylabel('Estado Climático', fontsize=12)
ax1.set_title('Estados Ocultos: Condición Climática por Día', fontsize=12, pad=6)
ax1.set_yticks([])
ax1.set_ylim(0, 1.4)
ax1.grid(axis='x', linestyle='--', alpha=0.3)

# Etiquetas abreviadas dentro de cada barra (primeras 3 letras)
for dia, est_idx in zip(dias, estados_seq):
    ax1.text(
        dia, 0.5, estados[est_idx][:3],
        ha='center', va='center', fontsize=7, fontweight='bold',
        color='black', rotation=90
    )

# Leyenda de estados
leyenda_estados = [
    mpatches.Patch(color=COLORES_ESTADOS[e], label=e) for e in estados
]
ax1.legend(handles=leyenda_estados, loc='upper right', fontsize=9, framealpha=0.9)

# ---------- Subgráfico 2: Observaciones de humedad ----------
ax2.bar(dias, [1] * N_DIAS, color=colores_obs_barras,
        edgecolor='white', linewidth=0.6, width=0.9)
ax2.set_ylabel('Nivel de Humedad', fontsize=12)
ax2.set_title('Observaciones: Nivel de Humedad por Día', fontsize=12, pad=6)
ax2.set_xlabel('Día de simulación', fontsize=12)
ax2.set_yticks([])
ax2.set_ylim(0, 1.4)
ax2.set_xticks(dias)
ax2.set_xticklabels(dias, fontsize=7.5)
ax2.grid(axis='x', linestyle='--', alpha=0.3)

# Etiquetas abreviadas dentro de cada barra
etq_corta = {'Humedad Baja': 'Baja', 'Humedad Media': 'Med', 'Humedad Alta': 'Alta'}
for dia, obs_idx in zip(dias, obs_seq):
    color_texto = 'white' if observaciones[obs_idx] != 'Humedad Baja' else 'black'
    ax2.text(
        dia, 0.5, etq_corta[observaciones[obs_idx]],
        ha='center', va='center', fontsize=7, fontweight='bold',
        color=color_texto, rotation=90
    )

# Leyenda de observaciones
leyenda_obs = [
    mpatches.Patch(color=COLORES_HUMEDAD[o], label=o) for o in observaciones
]
ax2.legend(handles=leyenda_obs, loc='upper right', fontsize=9, framealpha=0.9)

# Guardar figura principal
plt.savefig('simulacion_hmm.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n  Grafico principal guardado como 'simulacion_hmm.png'")

# ============================================================
# FIGURA 2: Estadísticas de distribución (torta + barras)
# ============================================================

fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Estadísticas Globales de la Simulación HMM', fontsize=14, fontweight='bold')

# ---------- Gráfico 2a: Distribución de estados (torta) ----------
dias_vals  = [stats['dias_por_estado'][e] for e in estados]
col_pie    = [COLORES_ESTADOS[e] for e in estados]
etq_pie    = [f"{e}\n({d} días)" for e, d in zip(estados, dias_vals)]

axes[0].pie(
    dias_vals, labels=etq_pie, colors=col_pie,
    autopct='%1.1f%%', startangle=90,
    textprops={'fontsize': 10}, pctdistance=0.75
)
axes[0].set_title('Distribución de Estados Climáticos', fontsize=11)

# ---------- Gráfico 2b: Porcentaje de humedad (barras verticales) ----------
obs_etq  = [o.replace('Humedad ', '') for o in observaciones]
pct_vals = [stats['pct_humedad'][o] for o in observaciones]
col_bars = [COLORES_HUMEDAD[o] for o in observaciones]

bars = axes[1].bar(obs_etq, pct_vals, color=col_bars, edgecolor='black', linewidth=0.7)
axes[1].set_ylabel('Porcentaje (%)', fontsize=11)
axes[1].set_title('Distribución de Niveles de Humedad', fontsize=11)
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
for bar, pct in zip(bars, pct_vals):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold'
    )

# ---------- Gráfico 2c: Matriz de confusión (mapa de calor) ----------
mc = stats['matriz_confusion']
im = axes[2].imshow(mc, cmap='Blues', aspect='auto')
axes[2].set_xticks(range(len(observaciones)))
axes[2].set_yticks(range(len(estados)))
axes[2].set_xticklabels([o.replace('Humedad ', '') for o in observaciones], fontsize=10)
axes[2].set_yticklabels(estados, fontsize=10)
axes[2].set_title('Matriz de Confusión\n(Estado × Observación)', fontsize=11)
axes[2].set_xlabel('Observación (Humedad)', fontsize=10)
axes[2].set_ylabel('Estado Oculto (Clima)', fontsize=10)

# Anotar valores dentro de las celdas
for i in range(len(estados)):
    for j in range(len(observaciones)):
        color_txt = 'white' if mc[i, j] > mc.max() / 2 else 'black'
        axes[2].text(
            j, i, str(mc[i, j]),
            ha='center', va='center', fontsize=12,
            fontweight='bold', color=color_txt
        )
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('estadisticas_hmm.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Grafico de estadisticas guardado como 'estadisticas_hmm.png'")
print("\n  Simulacion completada exitosamente.")
