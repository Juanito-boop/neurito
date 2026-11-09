"""Entradas públicas de Neurito."""

from .core import evaluate, make_dataset, train_network

__all__ = ["evaluate", "main", "make_dataset", "train_network"]


def main():
    """Inicia la experiencia integrada del laberinto."""
    from .laberinto import main as ejecutar_laberinto

    ejecutar_laberinto()
