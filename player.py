import pygame as pg
import time
from settings import *

vec = pg.math.Vector2


class Player(pg.sprite.Sprite):
    def __init__(self, game):
        # -- DODAVANJE OBJEKTA U GRUPE --
        self.groups = game.all_sprites, game.all_player
        pg.sprite.Sprite.__init__(self, self.groups)

        super(Player, self).__init__()

        self.game = game

        # -- SLIKE MIROVANJA --
        self.idle_images = []
        img = pg.image.load("Images/Player/Player_idle_right.png")
        self.idle_images.append(img)
        img = pg.image.load("Images/Player/Player_idle_left.png")
        self.idle_images.append(img)

        # -- SLIKE KRETANJA UDESNO --
        self.mr_images = []
        image = pg.image.load("Images/Player/Player_idle_right.png")
        self.mr_images.append(image)
        image = pg.image.load("Images/Player/Player_walk_right.png")
        self.mr_images.append(image)

        # -- SLIKE KRETANJA ULIJEVO --
        self.ml_images = []
        image = pg.image.load("Images/Player/Player_idle_left.png")
        self.ml_images.append(image)
        image = pg.image.load("Images/Player/Player_walk_left.png")
        self.ml_images.append(image)

        # -- SLIKE SKAKANJA --
        self.jump_images = []
        img = pg.image.load("Images/Player/Player_jump_right.png")
        self.jump_images.append(img)
        img = pg.image.load("Images/Player/Player_jump_left.png")
        self.jump_images.append(img)

        self.image = self.mr_images[0]
        self.rect = self.image.get_rect()
        self.rect.x = 50

        self.pos = vec(50, WINDOW_HEIGHT - 64)  # Početna pozicija
        self.vel = vec(0, 0)                    # Brzina
        self.acc = vec(0, 0)                    # Ubrzanje
        self.x = 50
        self.y = 128

        self.wait = 0           # Vrijeme promjene slike, koristeno za računanje koliko je vremena prošlo od promjene
        self.dir = 'r'          # Smjer kretanja igrača
        self.sprite = 0         # Redni broj trenutačne slike
        self.jumping = False    # Da li igrač skače

    # -- METODA SKAKANJA --
    def jump(self):
        hits = pg.sprite.spritecollide(self, self.game.all_ground, False)

        if hits:
            self.vel.y = -PLAYER_JUMP

    # -- METODA ANIMACIJE KRETANJA --
    # Mora voditi računa o smjeru kretanja, trenutnoj slici i kada je slika promijenjena
    def animate(self, sprite):
        if self.dir == 'r':
            if sprite == 0:
                self.image = self.mr_images[1]
                self.sprite = 1
                self.wait = time.time()
            elif sprite == 1:
                self.image = self.mr_images[0]
                self.sprite = 0
                self.wait = time.time()

        elif self.dir == 'l':
            if sprite == 0:
                self.image = self.ml_images[1]
                self.sprite = 1
                self.wait = time.time()
            elif sprite == 1:
                self.image = self.ml_images[0]
                self.sprite = 0
                self.wait = time.time()

    # -- METODA ZA KRETANJE IGRAČA --
    def update(self):
        self.acc = vec(0, PLAYER_GRAVITY)

        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            if not self.jumping:
                self.dir = 'l'
                if time.time() - self.wait > 0.2:
                    self.animate(self.sprite)
            self.acc.x = -PLAYER_ACCELERATION
        if keys[pg.K_RIGHT]:
            if not self.jumping:
                self.dir = 'r'
                if time.time() - self.wait > 0.2:
                    self.animate(self.sprite)
            self.acc.x = PLAYER_ACCELERATION
        if keys[pg.K_UP]:
            self.jumping = True
            if self.dir == 'r':
                self.image = self.jump_images[0]
            else:
                self.image = self.jump_images[1]
            self.jump()

        # Primijeni trenje
        self.acc.x += self.vel.x * PLAYER_FRICTION
        # Jednadžba gibanja
        self.vel += self.acc
        self.pos += 0.5 * self.acc + self.vel

        self.rect.midbottom = self.pos
