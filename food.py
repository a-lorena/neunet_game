import pygame as pg
from settings import *


class Ants(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        # -- DODAVANJE SLIKE U GRUPE --
        self.groups = game.all_sprites, game.all_food, game.all_ants
        pg.sprite.Sprite.__init__(self, self.groups)

        # -- UČITAVANJE SLIKE --
        self.image = pg.image.load("Images/Food/Ant_32.png")
        self.rect = self.image.get_rect()

        # -- POSTAVLJANJE NA POZICIJU --
        self.x = x
        self.y = y
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE


class Corn(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        # -- DODAVANJE SLIKE U GRUPE --
        self.groups = game.all_sprites, game.all_food, game.all_corns
        pg.sprite.Sprite.__init__(self, self.groups)

        # -- UČITAVANJE SLIKE --
        self.image = pg.image.load("Images/Food/Corn_32.png")
        self.rect = self.image.get_rect()

        # -- POSTAVLJANJE NA POZICIJU --
        self.x = x
        self.y = y
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE


class Worm(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        # -- DODAVANJE SLIKE U GRUPE --
        self.groups = game.all_sprites, game.all_food, game.all_worms
        pg.sprite.Sprite.__init__(self, self.groups)

        # -- UČITAVANJE SLIKE --
        self.image = pg.image.load("Images/Food/Worm_32.png")
        self.rect = self.image.get_rect()

        # -- POSTAVLJANJE NA POZICIJU --
        self.x = x
        self.y = y
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE


class Tomato(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        # -- DODAVANJE SLIKE U GRUPE --
        self.groups = game.all_sprites, game.all_food, game.all_tomatos
        pg.sprite.Sprite.__init__(self, self.groups)

        # -- UČITAVANJE SLIKE --
        self.image = pg.image.load("Images/Food/Tomato_32.png")
        self.rect = self.image.get_rect()

        # -- POSTAVLJANJE NA POZICIJU --
        self.x = x
        self.y = y
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE


class Flag(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        # -- DODAVANJE SLIKE U GRUPE --
        self.groups = game.all_sprites, game.flag
        pg.sprite.Sprite.__init__(self, self.groups)

        # -- UČITAVANJE SLIKE --
        self.image = pg.image.load("Images/Food/Flag_32.png")
        self.rect = self.image.get_rect()

        # -- POSTAVLJANJE NA POZICIJU --
        self.x = x
        self.y = y
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE
