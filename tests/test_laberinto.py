import unittest

import numpy as np

from neurito.laberinto_mundo import (
    MAPAS,
    Laberinto,
    accion_experta,
    distancias_bfs,
    elegir_accion,
    libres,
    valido,
    vecino,
)
from neurito.laberinto_red import RedLaberinto, generar_datos


class PruebaLaberinto(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ent, cls.prueba, cls.casos = generar_datos()
        cls.red = RedLaberinto()
        for _ in range(400):
            cls.red.entrenar_epoca(*cls.ent)

    def test_mapas_conectados_y_bfs_correcto(self):
        for mapa in MAPAS:
            celdas = libres(mapa)
            for meta in celdas:
                distancias = distancias_bfs(mapa, meta)
                self.assertEqual(len(distancias), len(celdas))
                for posicion in celdas:
                    if posicion == meta:
                        continue
                    accion = accion_experta(mapa, posicion, distancias)
                    siguiente = vecino(posicion, accion)
                    self.assertTrue(valido(mapa, siguiente))
                    self.assertEqual(
                        distancias[siguiente], distancias[posicion] - 1
                    )

    def test_datos_separados_y_red_aprende(self):
        entradas, etiquetas = self.ent
        prueba, etiquetas_prueba = self.prueba
        self.assertEqual(entradas.shape[1], 165)
        self.assertEqual(
            len(set(map(tuple, entradas)) & set(map(tuple, prueba))), 0
        )
        self.assertLess(self.red.historial[-1], self.red.historial[0])
        self.assertGreaterEqual(
            self.red.exactitud(prueba, etiquetas_prueba), 0.90
        )

    def test_probabilidades_validas(self):
        probabilidades = self.red.adelante(self.prueba[0][:20])[2]
        self.assertTrue(np.all(np.isfinite(probabilidades)))
        self.assertTrue(np.allclose(probabilidades.sum(axis=1), 1))

    def test_asistencia_no_cruza_paredes(self):
        for indice, inicio, meta, _ in self.casos[:80]:
            mundo = Laberinto(indice)
            mundo.posicion, mundo.meta = inicio, meta
            accion, _ = elegir_accion(self.red, mundo, {inicio: 1}, True)
            self.assertTrue(valido(mundo.mapa, vecino(inicio, accion)))


if __name__ == "__main__":
    unittest.main()
