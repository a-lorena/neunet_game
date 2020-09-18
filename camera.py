import pygame as pg
from settings import *


class Camera:
    def __init__(self, width, height):
        # -- INICIJALIZACIJA KAMERE --
        self.camera = pg.Rect(0, 0, width, height)
        self.width = width
        self.height = height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def update(self, target):
        # -- AŽURIRANJE KOORDINATE x NA KOJOJ SE KAMERA NALAZI --
        x = -target.rect.x + int(WINDOW_WIDTH / 4)
        y = 0

        # -- OGRANIČAVANJE POMICANJA KAMERE --
        x = min(0, x)                               # Lijevo
        x = max(-(self.width - WINDOW_WIDTH), x)    # Desno

        self.camera = pg.Rect(x, y, self.width, self.height)
