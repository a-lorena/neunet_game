import pygame as pg
from settings import *

vec = pg.math.Vector2


class Player(pg.sprite.Sprite):
    def __init__(self, game):
        super(Player, self).__init__()
        self.game = game

        # -- UČITAVANJE SLIKE --
        self.image = pg.image.load("Images/Player/Player_idle_right.png")
        self.rect = self.image.get_rect()
        self.rect.x = 96

        self.pos = vec(96, WINDOW_HEIGHT-64)  # Početna pozicija
        self.vel = vec(0, 0)                  # Brzina
        self.acc = vec(0, 0)                  # Ubrzanje
        self.x = 50
        self.y = 128

    # -- METODA SKAKANJA --
    def jump(self):
        hits = pg.sprite.spritecollide(self, self.game.all_ground, False)

        if hits:
            self.vel.y = -PLAYER_JUMP

    # -- METODA ZA KRETANJE IGRAČA --
    def update(self, input_actions):
        self.acc = vec(0, PLAYER_GRAVITY)

        # -- IGRAČ SE NEPRESTANO KREĆE UDESNO --
        if self.game.runs:
            self.acc.x = PLAYER_ACCELERATION
            self.game.reward += 0.1
        # if input_actions[2] == 1:
            # self.acc.x = -PLAYER_ACCELERATION

        # Primijeni trenje
        self.acc.x += self.vel.x * PLAYER_FRICTION
        # Jednadžba gibanja
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        self.rect.midbottom = self.pos
