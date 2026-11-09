"""Red neuronal 165 -> 96 -> 4 entrenada con ejemplos de BFS."""
import numpy as np

from .laberinto_mundo import MAPAS, accion_experta, distancias_bfs, libres

LADO = 9
ENTRADAS = len(MAPAS) + 2 * LADO * LADO


def codificar(indice, posicion, meta):
    """Codifica mapa, posición y meta como vectores one-hot."""
    x = np.zeros(ENTRADAS, dtype=np.float64)
    x[indice] = 1
    mapas = len(MAPAS)
    x[mapas + posicion[0] * LADO + posicion[1]] = 1
    x[mapas + LADO * LADO + meta[0] * LADO + meta[1]] = 1
    return x


def generar_datos(semilla=42, fraccion_prueba=0.2):
    """Genera y separa ejemplos sin repetir pares posición/meta."""
    muestras = []
    for indice, mapa in enumerate(MAPAS):
        celdas = libres(mapa)
        for meta in celdas:
            distancias = distancias_bfs(mapa, meta)
            for posicion in celdas:
                if posicion == meta or posicion not in distancias:
                    continue
                muestras.append(
                    (indice, posicion, meta, accion_experta(mapa, posicion, distancias))
                )
    rng = np.random.default_rng(semilla)
    rng.shuffle(muestras)
    corte = int(len(muestras) * (1 - fraccion_prueba))

    def arrays(filas):
        return (
            np.stack([codificar(i, posicion, meta) for i, posicion, meta, _ in filas]),
            np.asarray([accion for _, _, _, accion in filas], dtype=np.int64),
        )

    return arrays(muestras[:corte]), arrays(muestras[corte:]), muestras[corte:]


class RedLaberinto:
    def __init__(self, semilla=42, ocultas=96, tasa=0.45):
        self.rng = np.random.default_rng(semilla)
        self.ocultas = ocultas
        self.tasa = tasa
        self.reiniciar()

    def reiniciar(self):
        self.W1 = self.rng.normal(
            0, np.sqrt(2 / ENTRADAS), (ENTRADAS, self.ocultas)
        )
        self.b1 = np.zeros((1, self.ocultas))
        self.W2 = self.rng.normal(
            0, np.sqrt(2 / self.ocultas), (self.ocultas, 4)
        )
        self.b2 = np.zeros((1, 4))
        self.epocas = 0
        self.historial = []

    def adelante(self, entradas):
        z1 = entradas @ self.W1 + self.b1
        activacion = np.maximum(0, z1)
        logits = activacion @ self.W2 + self.b2
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilidades = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return z1, activacion, probabilidades

    def entrenar_epoca(self, entradas, etiquetas, lote=256):
        indices = self.rng.permutation(len(entradas))
        total = 0.0
        for inicio in range(0, len(indices), lote):
            ids = indices[inicio : inicio + lote]
            x, y = entradas[ids], etiquetas[ids]
            z1, activacion, probabilidades = self.adelante(x)
            n = len(x)
            total -= np.log(
                np.clip(probabilidades[np.arange(n), y], 1e-10, 1)
            ).sum()
            delta2 = probabilidades.copy()
            delta2[np.arange(n), y] -= 1
            delta2 /= n
            dw2 = activacion.T @ delta2
            db2 = delta2.sum(axis=0, keepdims=True)
            delta1 = (delta2 @ self.W2.T) * (z1 > 0)
            dw1 = x.T @ delta1
            db1 = delta1.sum(axis=0, keepdims=True)
            self.W1 -= self.tasa * dw1
            self.b1 -= self.tasa * db1
            self.W2 -= self.tasa * dw2
            self.b2 -= self.tasa * db2
        self.epocas += 1
        perdida = total / len(entradas)
        self.historial.append(float(perdida))
        return perdida

    def probabilidades(self, indice, posicion, meta):
        return self.adelante(codificar(indice, posicion, meta)[None, :])[2][0]

    def exactitud(self, entradas, etiquetas):
        predicciones = np.argmax(self.adelante(entradas)[2], axis=1)
        return float(np.mean(predicciones == etiquetas))
