import pygame as pg
import time
from settings import *


class Enemy(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        # -- DODAVANJE SLIKA U GRUPE --
        self.groups = game.all_sprites, game.all_enemies
        pg.sprite.Sprite.__init__(self, self.groups)

        # -- SLIKE ZA KRETANJE UDESNO --
        self.mr_images = []
        self.image = pg.image.load("Images/Enemy/Enemy_right_1.png")
        self.mr_images.append(self.image)
        self.image = pg.image.load("Images/Enemy/Enemy_right_2.png")
        self.mr_images.append(self.image)

        # -- SLIKE ZA KRETANJE ULIJEVO --
        self.ml_images = []
        self.image = pg.image.load("Images/Enemy/Enemy_left_1.png")
        self.ml_images.append(self.image)
        self.image = pg.image.load("Images/Enemy/Enemy_left_2.png")
        self.ml_images.append(self.image)

        self.image = self.mr_images[0]
        self.rect = self.image.get_rect()

        # -- POSTAVLJANJE NA POČETNU POZICIJU --
        self.x = x
        self.y = y
        self.rect.x = x * TILE_SIZE
        self.rect.y = y * TILE_SIZE

        # Varijable potrebne za kretanje
        self.counter = 0                # Udaljeost koju je neprijatelj prošao
        self.pocetno = self.rect.x      # Početna pozicija neprijatelja

        # Varijable potrebne za animaciju
        self.sprite = 0                 # Redni broj slike u animaciji
        self.wait = 0                   # U kojem trenutku je promjenjena slika, potrebno za računanje trajanja svake slike
        self.direction = 'r'            # Smjer kretanja

    # -- FUNKCIJA ZA ANIMCIJU NEPRIJATELJA --
    def animate(self, sprite, direction):
        if direction == 'r':
            if sprite == 0:
                self.image = self.mr_images[1]
                self.sprite = 1
                self.wait = time.time()
            elif sprite == 1:
                self.image = self.mr_images[0]
                self.sprite = 0
                self.wait = time.time()

        elif direction == 'l':
            if sprite == 0:
                self.image = self.ml_images[1]
                self.sprite = 1
                self.wait = time.time()
            elif sprite == 1:
                self.image = self.ml_images[0]
                self.sprite = 0
                self.wait = time.time()

    # -- FUNKCIJA ZA KRETANJE NEPRIJATELJA --
    def move(self):
        distance = 125
        speed = 3

        if self.counter >= 0 and self.counter <= distance:
            self.rect.x += speed
            if self.direction == 'l':
                self.image = self.mr_images[0]
                self.direction = 'r'
            if time.time() - self.wait > 0.4:
                self.animate(self.sprite, self.direction)

        if self.counter >= distance and self.counter <= distance*2:
            self.rect.x -= speed
            if self.direction == 'r':
                self.image = self.ml_images[0]
                self.direction = 'l'
            if time.time() - self.wait > 0.4:
                self.animate(self.sprite, self.direction)

        if self.rect.x == self.pocetno:
            self.counter = 0

        self.counter += 1
