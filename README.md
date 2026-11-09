## Neurito

Neurito incluye una red neuronal para elegir acciones y una fase de
integración con tres laberintos. La red aprende por imitación supervisada:
BFS genera las etiquetas durante el entrenamiento y no se consulta durante
la navegación.

### Ejecutar el laberinto

```bash
uv sync
uv run neurito-laberinto
```

También se puede iniciar con `uv run neurito`. La interfaz permite entrenar
la red, alternar entre red pura y navegación asistida, cambiar de mapa y
avanzar paso a paso.

### Pruebas

```bash
uv run python -m unittest discover -s tests -v
```
# Neurito

Además de la mascota original, el proyecto incluye un modo de salto con una
red neuronal entrenable:

```bash
uv run saltar.py
# o, después de instalar el proyecto:
uv run neurito-saltar
```

En el modo de salto, `1` entrena una época, `2` cien épocas, `3` activa el
entrenamiento continuo, `A` alterna entre control automático y manual,
`Espacio` salta manualmente, `C` reinicia la pista, `R` reinicia la red y
`Esc` sale.
