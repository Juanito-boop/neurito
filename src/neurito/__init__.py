"""Entradas públicas de Neurito."""

import runpy


__all__ = ["evaluate", "main", "make_dataset", "train_network"]


def main():
    """Inicia la experiencia integrada del laberinto."""
    from .laberinto import main as ejecutar_laberinto

    ejecutar_laberinto()
def main() -> None:
    """Ejecuta la interfaz original de la mascota."""
    runpy.run_module("neurito.main", run_name="__main__")
