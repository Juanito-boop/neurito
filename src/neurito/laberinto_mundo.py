"""Mapas, búsqueda BFS experta y simulación del laberinto."""
from collections import deque
from dataclasses import dataclass
import random

DIRECCIONES = ((-1, 0), (1, 0), (0, -1), (0, 1))
NOMBRES = ("ARRIBA", "ABAJO", "IZQUIERDA", "DERECHA")
MAPAS = (
    (
        "#########",
        "#...#...#",
        "#.#.#.#.#",
        "#.#...#.#",
        "#.###.#.#",
        "#...#...#",
        "###.#.###",
        "#.......#",
        "#########",
    ),
    (
        "#########",
        "#...#...#",
        "#.#...#.#",
        "#...#...#",
        "###.#.###",
        "#...#...#",
        "#.#...#.#",
        "#...#...#",
        "#########",
    ),
    (
        "#########",
        "#.......#",
        "#.###.#.#",
        "#...#.#.#",
        "###.#...#",
        "#...###.#",
        "#.#.....#",
        "#.......#",
        "#########",
    ),
)


def libres(mapa):
    return [
        (fila, columna)
        for fila, linea in enumerate(mapa)
        for columna, celda in enumerate(linea)
        if celda == "."
    ]


def vecino(posicion, accion):
    df, dc = DIRECCIONES[accion]
    return posicion[0] + df, posicion[1] + dc


def valido(mapa, posicion):
    fila, columna = posicion
    return (
        0 <= fila < len(mapa)
        and 0 <= columna < len(mapa[0])
        and mapa[fila][columna] != "#"
    )


def distancias_bfs(mapa, meta):
    """Calcula distancias a la meta; se usa solo para generar etiquetas."""
    distancias = {meta: 0}
    cola = deque([meta])
    while cola:
        actual = cola.popleft()
        for accion in range(4):
            siguiente = vecino(actual, accion)
            if valido(mapa, siguiente) and siguiente not in distancias:
                distancias[siguiente] = distancias[actual] + 1
                cola.append(siguiente)
    return distancias


def accion_experta(mapa, origen, distancias):
    """Devuelve el movimiento BFS que reduce la distancia a la meta."""
    if origen not in distancias or distancias[origen] == 0:
        return None
    opciones = [
        (distancias.get(vecino(origen, accion), 10**9), accion)
        for accion in range(4)
    ]
    distancia, accion = min(opciones)
    assert distancia < distancias[origen]
    return accion


@dataclass
class Laberinto:
    indice: int = 0
    semilla: int = 99

    def __post_init__(self):
        self.azar = random.Random(self.semilla)
        self.exitos = 0
        self.pasos = 0
        self.choques = 0
        self.reiniciar(self.indice)

    def reiniciar(self, indice=None):
        if indice is not None:
            self.indice = indice % len(MAPAS)
        self.mapa = MAPAS[self.indice]
        celdas = libres(self.mapa)
        self.meta = self.azar.choice(celdas)
        distancias = distancias_bfs(self.mapa, self.meta)
        opciones = [
            posicion
            for posicion in celdas
            if posicion != self.meta and posicion in distancias
        ]
        self.posicion = self.azar.choice(opciones)
        self.visitadas = {self.posicion: 1}
        self.pasos = 0
        self.intentos = 0
        self.terminado = False
        self.ultimo = None

    def mover(self, accion):
        self.ultimo = accion
        self.intentos += 1
        nuevo = vecino(self.posicion, accion)
        if not valido(self.mapa, nuevo):
            self.choques += 1
            return "pared"
        self.posicion = nuevo
        self.pasos += 1
        self.visitadas[nuevo] = self.visitadas.get(nuevo, 0) + 1
        if nuevo == self.meta:
            self.exitos += 1
            self.terminado = True
            return "comida"
        return "avanza"


def elegir_accion(red, mundo, visitas, asistido):
    """Elige por argmax y, opcionalmente, evita paredes y visitas repetidas."""
    probabilidades = red.probabilidades(
        mundo.indice, mundo.posicion, mundo.meta
    )
    if not asistido:
        return int(probabilidades.argmax()), probabilidades
    opciones = [
        (
            float(probabilidades[accion])
            - 0.35 * visitas.get(vecino(mundo.posicion, accion), 0),
            -accion,
            accion,
        )
        for accion in range(4)
        if valido(mundo.mapa, vecino(mundo.posicion, accion))
    ]
    return max(opciones)[2], probabilidades
