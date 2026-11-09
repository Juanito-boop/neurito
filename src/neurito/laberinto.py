"""Interfaz Pygame del laberinto aprendido por imitación."""
from pathlib import Path

import pygame

from .laberinto_mundo import Laberinto, MAPAS, NOMBRES, elegir_accion
from .laberinto_red import RedLaberinto, generar_datos


FONDO = (250, 246, 252)
TEXTO = (58, 48, 76)
PANEL = (239, 232, 245)


def dibujar_texto(superficie, fuente, contenido, posicion, color=TEXTO):
    superficie.blit(fuente.render(str(contenido), True, color), posicion)


def dibujar_probabilidades(superficie, fuente, probabilidades, x, y):
    for indice, nombre in enumerate(NOMBRES):
        yy = y + indice * 34
        dibujar_texto(
            superficie,
            fuente,
            f"{nombre}: {probabilidades[indice]:.1%}",
            (x, yy),
        )
        pygame.draw.rect(
            superficie, (218, 207, 231), (x + 155, yy + 3, 190, 16), border_radius=6
        )
        pygame.draw.rect(
            superficie,
            (150, 104, 202),
            (x + 155, yy + 3, int(190 * probabilidades[indice]), 16),
            border_radius=6,
        )


def cargar_mascotas():
    carpeta = Path(__file__).resolve().parent / "assets"
    return [
        pygame.transform.smoothscale(
            pygame.image.load(str(carpeta / nombre)).convert_alpha(), (49, 49)
        )
        for nombre in ("neutral_1.png", "neutral_2.png", "neutral_3.png")
    ]


def main():
    pygame.init()
    pantalla = pygame.display.set_mode((1080, 690))
    pygame.display.set_caption("Neurito 3.0 - Aprende a encontrar comida")
    reloj = pygame.time.Clock()
    titulo = pygame.font.SysFont(None, 34)
    fuente = pygame.font.SysFont(None, 26)
    pequena = pygame.font.SysFont(None, 21)
    sprites = cargar_mascotas()
    red = RedLaberinto()
    (entrada, etiquetas), (entrada_prueba, etiquetas_prueba), _ = generar_datos()
    mundo = Laberinto()
    asistido = False
    automatico = True
    entrenando = False
    ultimo_tick = pygame.time.get_ticks()
    activo = True

    while activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT or (
                evento.type == pygame.KEYDOWN and evento.key == pygame.K_ESCAPE
            ):
                activo = False
            elif evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_1:
                    red.entrenar_epoca(entrada, etiquetas)
                elif evento.key == pygame.K_2:
                    for _ in range(100):
                        red.entrenar_epoca(entrada, etiquetas)
                elif evento.key == pygame.K_3:
                    entrenando = not entrenando
                elif evento.key == pygame.K_a:
                    asistido = not asistido
                    mundo.reiniciar()
                elif evento.key == pygame.K_p:
                    automatico = not automatico
                elif evento.key == pygame.K_n:
                    mundo.reiniciar(mundo.indice + 1)
                elif evento.key == pygame.K_g:
                    mundo.reiniciar()
                elif evento.key == pygame.K_r:
                    red.reiniciar()
                    entrenando = False
                    mundo.reiniciar()

        if entrenando:
            red.entrenar_epoca(entrada, etiquetas)
        ahora = pygame.time.get_ticks()
        if (automatico and ahora - ultimo_tick >= 210) or (
            not automatico and pygame.key.get_pressed()[pygame.K_SPACE]
        ):
            ultimo_tick = ahora
            accion, _ = elegir_accion(red, mundo, mundo.visitadas, asistido)
            resultado = mundo.mover(accion)
            if resultado == "comida" or mundo.intentos >= 45:
                mundo.reiniciar()

        pantalla.fill(FONDO)
        dibujar_texto(
            pantalla,
            titulo,
            "NEURITO 3.0 - LABERINTO INTELIGENTE",
            (30, 20),
        )
        dibujar_texto(
            pantalla,
            pequena,
            f"Mapa {mundo.indice + 1}/{len(MAPAS)} | Meta: alimento amarillo",
            (30, 55),
        )
        for fila, linea in enumerate(mundo.mapa):
            for columna, celda in enumerate(linea):
                color = (123, 94, 153) if celda == "#" else (234, 243, 245)
                pygame.draw.rect(
                    pantalla,
                    color,
                    (30 + columna * 55, 80 + fila * 55, 53, 53),
                    border_radius=6,
                )
        meta_fila, meta_columna = mundo.meta
        pygame.draw.circle(
            pantalla,
            (246, 191, 64),
            (30 + meta_columna * 55 + 27, 80 + meta_fila * 55 + 27),
            16,
        )
        fila, columna = mundo.posicion
        pantalla.blit(
            sprites[(ahora // 500) % len(sprites)],
            (32 + columna * 55, 82 + fila * 55),
        )

        panel_x = 565
        dibujar_texto(pantalla, fuente, f"Épocas: {red.epocas}", (panel_x, 82))
        perdida = red.historial[-1] if red.historial else None
        dibujar_texto(
            pantalla,
            fuente,
            f"Pérdida: {perdida:.4f}" if perdida is not None else "Pérdida: sin entrenar",
            (panel_x, 114),
        )
        dibujar_texto(
            pantalla,
            fuente,
            f"Exactitud prueba: {red.exactitud(entrada_prueba, etiquetas_prueba):.1%}",
            (panel_x, 146),
        )
        dibujar_texto(
            pantalla,
            fuente,
            f"Modo: {'ASISTIDO' if asistido else 'RED PURA'}",
            (panel_x, 178),
        )
        dibujar_texto(
            pantalla,
            fuente,
            f"Navegación: {'AUTOMÁTICA' if automatico else 'PASO A PASO'}",
            (panel_x, 210),
        )
        dibujar_texto(
            pantalla,
            fuente,
            f"Comidas: {mundo.exitos} | Pasos: {mundo.pasos} | Paredes: {mundo.choques}",
            (panel_x, 242),
        )
        accion, probabilidades = elegir_accion(
            red, mundo, mundo.visitadas, asistido
        )
        dibujar_texto(
            pantalla,
            pequena,
            f"Próximo movimiento: {NOMBRES[accion]}",
            (panel_x, 278),
        )
        dibujar_texto(
            pantalla, fuente, "SALIDAS SOFTMAX DE LA RED", (panel_x, 315)
        )
        dibujar_probabilidades(pantalla, pequena, probabilidades, panel_x, 350)
        pygame.draw.rect(pantalla, PANEL, (panel_x, 505, 465, 125), border_radius=10)
        dibujar_texto(pantalla, pequena, "CONTROLES", (panel_x + 15, 518))
        dibujar_texto(
            pantalla,
            pequena,
            "1: +1 época   2: +100 épocas   3: continuo",
            (panel_x + 15, 545),
        )
        dibujar_texto(
            pantalla,
            pequena,
            "A: asistido   P: automático/paso   ESPACIO: avanzar",
            (panel_x + 15, 570),
        )
        dibujar_texto(
            pantalla,
            pequena,
            "N: mapa   G: nuevo reto   R: reiniciar   ESC: salir",
            (panel_x + 15, 595),
        )
        pygame.display.flip()
        reloj.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
