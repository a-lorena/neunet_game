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
        #self.rect.center = (70, WINDOW_HEIGHT-64)

        self.pos = vec(96, WINDOW_HEIGHT-64)  # Position
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.x = 0
        self.y = 0

    def jump(self):
        hits = pg.sprite.spritecollide(self, self.game.can_stand, False)

        if hits:
            self.vel.y = -PLAYER_JUMP

    def update(self, input_actions):
        self.acc = vec(0, PLAYER_GRAVITY)

        # -- IGRAČ SE NEPRESTANO KREĆE UDESNO --
        if self.game.runs:
            self.acc.x = PLAYER_ACCELERATION
            self.game.reward += 0.1
        # if input_actions[2] == 1:
            # self.acc.x = -PLAYER_ACCELERATION

        # -- TRENJE --
        self.acc.x += self.vel.x * PLAYER_FRICTION
        # --JEDNADŽBE GIBANJA --
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc
        # -- AŽURIRANJE POZICIJE IGRAČA --
        self.rect.midbottom = self.pos
