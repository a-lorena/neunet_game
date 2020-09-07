import pygame as pg
from settings import *


class Ground(pg.sprite.Sprite):
    def __init__(self, game, x, y, type_of_ground):
        # -- DODAVANJE SLIKA U GRUPE --
        self.groups = game.all_sprites, game.all_ground, game.can_stand
        self.player = game.player
        pg.sprite.Sprite.__init__(self, self.groups)

        # -- UČITAVANJE SLIKA --
        if type_of_ground == '1':
            self.image = pg.image.load("Images/Ground/Dirt_32.png")
            self.rect = self.image.get_rect()
        elif type_of_ground == '2':
            self.image = pg.image.load("Images/Ground/Grass_32.png")
            self.rect = self.image.get_rect()

        # -- POSTAVLJANJE NA POZICIJU --
        self.x = x
        self.y = y
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE
