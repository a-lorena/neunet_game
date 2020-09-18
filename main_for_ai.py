import pygame as pg
from settings import *
import time
from player_for_ai import *
from ground import *
from food import *
from tilemap_for_ai import *
from camera import *


class Game:
    def __init__(self):
        # -- INICIJALIZACIJA IGRE --
        pg.init()

        self.window = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption(TITLE)

        self.clock = pg.time.Clock()
        self.load_data()
        self.running = True
        self.player = None

        self.reward = 0.1
        self.terminal = False

    def load_data(self):
        # -- UČITAVANJE MAPE --
        self.map = Map()

    def new(self, input_actions):
        # -- NOVA IGRA --
        # -- GRUPE SLIKA --
        self.all_sprites = pg.sprite.Group()

        self.all_ground = pg.sprite.Group()
        self.all_platforms = pg.sprite.Group()

        self.all_food = pg.sprite.Group()
        self.all_ants = pg.sprite.Group()
        self.all_corns = pg.sprite.Group()
        self.all_worms = pg.sprite.Group()
        self.all_tomatos = pg.sprite.Group()
        self.flag = pg.sprite.Group()

        self.score = 0

        # -- UČITAVANJE MAPE --
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Ground(self, col, row, tile)
                if tile == '2':
                    Ground(self, col, row, tile)
                if tile == 'A':
                    Ant(self, col, row)
                if tile == 'C':
                    Corn(self, col, row)
                if tile == 'W':
                    Worm(self, col, row)
                if tile == 'T':
                    Tomato(self, col, row)
                if tile == 'F':
                    Flag(self, col, row)

        # -- INICIJALIZACIJA IGRAČA --
        self.player = Player(self)
        self.all_sprites.add(self.player)

        # -- INICIJALIZACIJA KAMERE --
        self.camera = Camera(self.map.width, self.map.height)
        self.hit = False

        podaci = self.run(input_actions)

        return podaci

    def run(self, input_actions):
        # -- TIJEK IGRE --
        self.playing = True
        self.terminal = False
        self.reward = 0.1
        while self.playing:
            self.events(input_actions)
            self.update(input_actions)
            podaci = self.draw()
            self.clock.tick(FPS)

        return podaci

    def events(self, input_actions):
        # -- PROVJERAVA IZLAZAK IZ IGRE --
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.playing = False
                self.running = False
                pg.quit()

        # -- PROVJERA AKO IGRAČ SKAČE --
        if input_actions[1] == 1:
            self.player.jump()

    def update(self, input_actions):
        # -- UPDATE --
        self.all_sprites.update(input_actions)
        self.all_ground.update()
        self.all_platforms.update()
        self.all_food.update()

        self.player.update(input_actions)

        # -- PROVJERA AKO SE IGRAČ NALAZI NA PLATFORMI --
        if self.player.vel.y > 0:
            hits_platform = pg.sprite.spritecollide(self.player, self.all_ground, False)
            if hits_platform:
                self.player.pos.y = hits_platform[0].rect.top
                self.player.vel.y = 0

        if pg.sprite.spritecollide(self.player, self.all_ground, False):
            self.player.rect.x -= 1

        if self.player.pos.y <= 472:
            self.reward += 0.1

        # -- AKO IGRAČ UPADNE U RUPU IGRA SE ZAVRŠAVA I POČINJE NOVA --
        if self.player.pos.y > (WINDOW_HEIGHT+50):
            self.reward -= 10
            self.terminal = True
            self.playing = False

        # -- PROVJERA I AŽURIRANJE OSVOJENIH BODOVA --
        if pg.sprite.spritecollide(self.player, self.all_ants, True):
            self.score += 1
        if pg.sprite.spritecollide(self.player, self.all_corns, True):
            self.score += 5
        if pg.sprite.spritecollide(self.player, self.all_worms, True):
            self.score += 10
        if pg.sprite.spritecollide(self.player, self.all_tomatos, True):
            self.score += 50
            self.reward += 5
        if pg.sprite.spritecollide(self.player, self.flag, True):
            self.hit = True
            self.pocetak = time.time()
            self.score += 100
            self.reward += 100
            self.playing = False

        # -- KAMERA UPDATE --
        self.camera.update(self.player)

        # KADA IGRAČ POKUPI ZASTAVICU POBJEDIO JE, TE NAKON 2 SEKUDNE KREĆE NOVA IGRA
        if self.hit == True:
            if time.time() - self.pocetak > 2:
                self.playing = False

    def draw_grid(self):
        # -- CRTANJE MREŽE --
        for x in range(0, WINDOW_WIDTH, TILE_SIZE):
            pg.draw.line(self.window, WHITE, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, TILE_SIZE):
            pg.draw.line(self.window, WHITE, (0, y), (WINDOW_WIDTH, y))

    def draw(self):
        # -- ISPUNA PROZORA CRNOM BOJOM (POZADINA) --
        self.window.fill(BLACK)

        # -- CRTANJE SVIH SLIČICA --
        for sprite in self.all_sprites:
            self.window.blit(sprite.image, self.camera.apply(sprite))

        # -- CRTANJE MREŽE --
        #self.draw_grid()

        # -- ISPIS UKUPNO OSTVARENIH BODOVA --
        font = pg.font.Font('freesansbold.ttf', 32)
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.window.blit(score_text, [10, 10])

        pg.display.flip()

        # -- PRIKUPLJA I VRAĆA PODATKE POTREBNE ZA NEURONSKU MREŽU --
        self.image_data = pg.surfarray.array3d((pg.display.get_surface()))

        podaci = []
        podaci.append(self.image_data)
        podaci.append(self.reward)
        podaci.append(self.terminal)

        return podaci
