# Resultados de los ejercicios del laberinto

Fecha de ejecución: 2026-11-09.

Los experimentos se ejecutaron con:

```bash
uv run python
```

Se usaron las semillas `42` del conjunto de datos y de la red. El conjunto
contiene 3 mapas, 2860 ejemplos de entrenamiento y 716 ejemplos de prueba.
Cada entrada tiene 165 características: identificador del mapa, posición y
meta codificados como one-hot.

## 1. Comportamiento sin entrenar

La red recién inicializada obtuvo **27.65 %** de exactitud en los ejemplos de
prueba. Esto no significa que haya aprendido una estrategia: sus pesos son
aleatorios, por lo que puede elegir paredes, retroceder o repetir posiciones.

## 2. Evolución del entrenamiento

Red de 96 neuronas ocultas, tasa `0.45`, mini-lotes de 256:

| Épocas | Pérdida de entrenamiento | Exactitud de prueba |
|---:|---:|---:|
| 1 | 1.36949 | 37.43 % |
| 10 | 0.68224 | 73.04 % |
| 50 | 0.27557 | 82.26 % |
| 100 | 0.13521 | 84.92 % |
| 200 | 0.03764 | 90.08 % |
| 400 | 0.01066 | 91.48 % |
| 800 | 0.00360 | 91.62 % |
| 1200 | 0.00201 | 91.90 % |

La pérdida baja rápidamente al principio y la mejora de generalización se
estabiliza después de aproximadamente 400 épocas.

## 3. Red pura frente a navegación asistida

Se evaluaron los 716 pares de posición-meta reservados para prueba. El límite
de cada reto fue `min(65, 2 * distancia óptima + 8)`.

| Modo | Retos completados | Porcentaje | Paredes | Pasos medios de los éxitos |
|---|---:|---:|---:|---:|
| Red pura | 595/716 | 83.10 % | 2 | 6.08 |
| Asistido | 715/716 | 99.86 % | 0 | 6.66 |

La red pura representa la decisión del modelo sin reglas externas. El modo
asistido descarta paredes y penaliza visitas repetidas; por eso mejora mucho
la tasa de finalización, aunque usa más pasos en promedio.

### Resultado por mapa

| Mapa | Red pura | Asistido |
|---:|---:|---:|
| 1 | 85.71 % | 100.00 % |
| 2 | 78.49 % | 98.80 % |
| 3 | 84.31 % | 100.00 % |

## 4. Tamaño de la capa oculta

Todas las redes se entrenaron durante 400 épocas con tasa `0.45`.

| Neuronas ocultas | Pérdida final | Exactitud de prueba |
|---:|---:|---:|
| 16 | 0.03424 | 87.15 % |
| 32 | 0.01608 | 89.25 % |
| 96 | 0.01066 | 91.48 % |
| 128 | 0.01028 | 90.78 % |

Pasar de 16 a 96 neuronas mejora la generalización. Aumentar a 128 no mejora
automáticamente el resultado: la capacidad adicional no garantiza una mayor
exactitud con la misma cantidad de épocas y semilla.

## 5. Tasa de aprendizaje

Todas las redes se entrenaron durante 400 épocas con 96 neuronas ocultas.

| Tasa | Pérdida final | Exactitud de prueba | ¿Valores finitos? |
|---:|---:|---:|---|
| 0.10 | 0.14905 | 87.15 % | Sí |
| 0.45 | 0.01066 | 91.48 % | Sí |
| 0.90 | 0.00327 | 93.30 % | Sí |

En esta ejecución, `0.90` fue la mejor de las tres. Este resultado depende de
la semilla y del número de épocas; una tasa alta puede ser más inestable si se
incrementa todavía más.

## 6. Cambio de mapa y generalización

Los pares de posición-meta de prueba no aparecen en entrenamiento, pero los
tres mapas sí forman parte del entrenamiento. Por tanto, el experimento mide
generalización dentro de los mismos mapas, no adaptación a un mapa
desconocido.

El mapa 2 fue el más difícil para la red pura, con 78.49 % de éxito. La
navegación asistida mantuvo al menos 98.80 % en los tres mapas.

Para generalizar a un mapa nuevo habría que incluir su estructura de paredes
en la entrada o entrenar con mapas separados de los usados en la prueba.

## 7. BFS, DFS y A*

### Comparación de rutas

Se calcularon rutas para los mismos 716 retos de prueba:

| Algoritmo | Longitud media | Nodos expandidos medios | Rutas óptimas |
|---|---:|---:|---:|
| BFS | 6.06 | 18.49 | 716/716 |
| DFS | 11.08 | 18.51 | 274/716 |
| A* | 6.06 | 10.18 | 716/716 |

BFS y A* encuentran rutas óptimas en estos grafos no ponderados. A* expande
menos nodos gracias a la heurística Manhattan. DFS puede encontrar una ruta
válida, pero no garantiza que sea corta.

### Entrenar con etiquetas de cada algoritmo

Se generó un conjunto equivalente, cambiando únicamente la etiqueta del
primer movimiento de cada buscador. Cada red se entrenó durante 400 épocas:

| Etiquetas usadas | Pérdida final | Exactitud de prueba |
|---|---:|---:|
| BFS | 0.01066 | 91.48 % |
| DFS | 0.01000 | 94.83 % |
| A* | 0.01072 | 92.60 % |

La exactitud de DFS fue mayor en esta partición, pero no implica que sus rutas
sean mejores: sus etiquetas representan decisiones de rutas más largas.
Cambiar BFS por DFS cambia el comportamiento que se imita, no convierte la
red en un planificador óptimo. A* conserva la calidad de las rutas de BFS y
reduce el trabajo de búsqueda durante la generación de datos.

## Conclusiones

1. La red aprende la política BFS y mejora claramente después de entrenar.
2. La precisión de clasificación no equivale por sí sola a completar el
   laberinto.
3. La asistencia es una capa de reglas externas y elimina los choques en la
   evaluación realizada.
4. A* es la alternativa más eficiente para generar etiquetas óptimas.
5. La evaluación actual no demuestra generalización a laberintos nuevos.
