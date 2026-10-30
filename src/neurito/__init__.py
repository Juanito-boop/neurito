from pathlib import Path

import numpy as np
import pygame

from .core import evaluate, make_dataset, train_network

LANGUAGE = "es"  # Change to "es" for Spanish interface text.x
TEXT = {
    "en": {
        "title": "NEURITO - MY ARTIFICIAL BRAIN",
        "hunger": "Hunger", "energy": "Energy", "boredom": "Boredom",
        "eat": "EAT", "sleep": "SLEEP", "play": "PLAY", "neutral": "CALM",
        "accuracy": "Test accuracy against synthetic policy",
        "epoch": "Epoch", "loss": "Loss",
        "controls1": "H/J: hunger +/-    E/D: energy +/-",
        "controls2": "A/S: boredom +/-    R: random",
        "controls3": "N: calm state    ESC: quit",
    },
    "es": {
        "title": "NEURITO - MI CEREBRO ARTIFICIAL",
        "hunger": "Hambre", "energy": "Energía", "boredom": "Aburrimiento",
        "eat": "COMER", "sleep": "DORMIR", "play": "JUGAR", "neutral": "NEUTRAL",
        "accuracy": "Precisión frente a la política sintética",
        "epoch": "Época", "loss": "Pérdida",
        "controls1": "H/J: hambre +/-    E/D: energía +/-",
        "controls2": "A/S: aburrimiento +/-    R: aleatorio",
        "controls3": "N: estado tranquilo    ESC: salir",
    },
}
labels = TEXT[LANGUAGE]
actions = ("eat", "sleep", "play")

X, targets = make_dataset(3000)
network, history = train_network(X, targets)
W1, b1, W2, b2 = (
    network.weights1,
    network.bias1,
    network.weights2,
    network.bias2,
)


def predict(data):
    return network.predict(data)


test, test_targets = make_dataset(1000, seed=43)
test_loss, accuracy = evaluate(network, test, test_targets)
for epoch in range(0, len(history), 200):
    print(f"{labels['epoch']} {epoch} | {labels['loss']}: {history[epoch]:.4f}")
print(f"{labels['loss']} final: {test_loss:.4f}")
print(f"{labels['accuracy']}: {accuracy:.2%}")

pygame.init()
screen = pygame.display.set_mode((900, 600))
pygame.display.set_caption("Neurito")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
small_font = pygame.font.SysFont(None, 23)
asset_dir = Path(__file__).resolve().parent / "assets"
pet_size = (330, 330)


def load_frames(names):
    return [
        pygame.transform.smoothscale(
            pygame.image.load(str(asset_dir / name)).convert_alpha(), pet_size
        )
        for name in names
    ]


frames = {
    "neutral": load_frames(["neutral_1.png", "neutral_2.png", "neutral_3.png"]),
    "eat": load_frames(["comer_1.png", "comer_2.png"]),
    "sleep": load_frames(["dormir_1.png", "dormir_2.png"]),
    "play": load_frames(["jugar_1.png", "jugar_2.png"]),
}
needs = [0.2, 0.9, 0.2]
current_state = "neutral"
frame_index = 0
last_frame_time = pygame.time.get_ticks()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_h:
                needs[0] = min(1, needs[0] + 0.1)
            elif event.key == pygame.K_j:
                needs[0] = max(0, needs[0] - 0.1)
            elif event.key == pygame.K_e:
                needs[1] = min(1, needs[1] + 0.1)
            elif event.key == pygame.K_d:
                needs[1] = max(0, needs[1] - 0.1)
            elif event.key == pygame.K_a:
                needs[2] = min(1, needs[2] + 0.1)
            elif event.key == pygame.K_s:
                needs[2] = max(0, needs[2] - 0.1)
            elif event.key == pygame.K_r:
                needs = list(np.random.rand(3))
            elif event.key == pygame.K_n:
                needs = [0.2, 0.9, 0.2]

    probabilities = predict(np.array([needs]))[0]
    action = actions[int(np.argmax(probabilities))]
    if needs[0] < 0.4 and needs[1] > 0.6 and needs[2] < 0.4:
        state = "neutral"  # Display rule; the network has three outputs.
    else:
        state = action

    now = pygame.time.get_ticks()
    if state != current_state:
        current_state, frame_index, last_frame_time = state, 0, now
    elif now - last_frame_time >= 500:
        frame_index = (frame_index + 1) % len(frames[state])
        last_frame_time = now

    screen.fill((250, 245, 252))
    title = font.render(labels["title"], True, (55, 45, 75))
    screen.blit(title, (45, 28))

    colors = [(235, 125, 75), (70, 175, 115), (105, 125, 225)]
    for index, key in enumerate(("hunger", "energy", "boredom")):
        value = needs[index]
        y = 125 + index * 95
        text = small_font.render(
            f"{labels[key]}: {value * 100:.0f}%", True, (35, 40, 55)
        )
        screen.blit(text, (50, y - 27))
        pygame.draw.rect(screen, (215, 220, 230), (50, y, 290, 22), border_radius=7)
        pygame.draw.rect(
            screen, colors[index], (50, y, int(290 * value), 22), border_radius=7
        )

    pet = frames[state][frame_index]
    screen.blit(pet, pet.get_rect(center=(650, 235)))
    screen.blit(font.render(labels[state], True, (65, 50, 85)), (585, 420))
    for index, action_name in enumerate(actions):
        score = small_font.render(
            f"{labels[action_name]}: {probabilities[index] * 100:.1f}%",
            True, (55, 50, 75)
        )
        screen.blit(score, (50, 390 + index * 28))

    screen.blit(small_font.render(labels["controls1"], True, (70, 70, 90)), (50, 495))
    screen.blit(small_font.render(labels["controls2"], True, (70, 70, 90)), (50, 525))
    screen.blit(small_font.render(labels["controls3"], True, (70, 70, 90)), (50, 555))
    pygame.display.flip()
    clock.tick(30)

pygame.quit()