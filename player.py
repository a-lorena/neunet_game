import pygame as pg
import time
from settings import *

vec = pg.math.Vector2


class Player(pg.sprite.Sprite):
    def __init__(self, game):
        # -- DODAVANJE SLIKA U GRUPE --
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
        self.rect.x = 96
        # self.rect.center = (70, WINDOW_HEIGHT-64)

        self.pos = vec(96, WINDOW_HEIGHT-64)  # Position
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.x = 96
        self.y = 128

        self.wait = 0
        self.dir = 'r'
        self.frame = 0
        self.sprite = 0
        self.jumping = False

    # -- FUNKCIJA SKAKANJA --
    def jump(self):
        hits = pg.sprite.spritecollide(self, self.game.can_stand, False)

        if hits:
            self.vel.y = -PLAYER_JUMP

    # -- FUNKCIJA ANIMACIJE KRETANJA --
    # Mora voditi računa o smjeru kretanja, trenutnoj slici i kada je slika promijenjena
    def animate(self, sprite, dir):
        if dir == 'r':
            if sprite == 0:
                self.image = self.mr_images[1]
                self.sprite = 1
                self.wait = time.time()
            elif sprite == 1:
                self.image = self.mr_images[2]
                self.sprite = 2
                self.wait = time.time()
            elif sprite == 2:
                self.image = self.mr_images[3]
                self.sprite = 3
                self.wait = time.time()
            elif sprite == 3:
                self.image = self.mr_images[0]
                self.sprite = 0
                self.wait = time.time()

        elif dir == 'l':
            if sprite == 0:
                self.image = self.ml_images[1]
                self.sprite = 1
                self.wait = time.time()
            elif sprite == 1:
                self.image = self.ml_images[2]
                self.sprite = 2
                self.wait = time.time()
            elif sprite == 2:
                self.image = self.ml_images[3]
                self.sprite = 3
                self.wait = time.time()
            elif sprite == 3:
                self.image = self.ml_images[0]
                self.sprite = 0
                self.wait = time.time()

    # -- FUNKCIJA ZA KRETANJE IGRAČA --
    def update(self):
        self.acc = vec(0, PLAYER_GRAVITY)

        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT]:
            self.game.direction = False
            self.acc.x = -PLAYER_ACCELERATION
        if keys[pg.K_RIGHT]:
            self.game.direction = True
            self.acc.x = PLAYER_ACCELERATION

        # apply friction
        self.acc.x += self.vel.x * PLAYER_FRICTION
        # equations of motion
        self.vel += self.acc
        self.pos += self.vel + 0.5 * self.acc

        self.rect.midbottom = self.pos
