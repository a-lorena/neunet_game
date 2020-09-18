import pygame as pg
import time
from settings import *
from player import *
from ground import *
from food import *
from tilemap import *
from camera import *
from enemy import *

from pygame import mixer

FPS = 60


class Game:
    def __init__(self):
        # -- INICIJALIZACIJA IGRE --
        pg.init()

        self.window = pg.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pg.display.set_caption(TITLE)

        self.icon = pg.image.load("Images/Icon.png")
        pg.display.set_icon(self.icon)

        mixer.music.load("Music/Grasslands Theme.mp3")
        mixer.music.play(-1)

        self.clock = pg.time.Clock()
        self.load_data()
        self.running = True
        self.player = None

        self.pocetak = 0            # Vrijeme pokupljanja zastavice
        self.score = 0              # Ukupno ostvareni bodovi
        self.win = False            # Uspješno odigrana igra
        self.life = 2               # Broj rezervnih života

        self.points_image = None    # Slika bodova
        self.draw_points = False    # Treba li crtati bodove
        self.points_appear = 0      # Vrijeme crtanja bodova

    def load_data(self):
        # -- UČITAVANJE MAPE --
        self.map = Map()

    def new(self):
        # -- NOVA IGRA --
        # -- GRUPE SLIKA --
        self.all_sprites = pg.sprite.Group()

        self.all_player = pg.sprite.Group()

        self.all_ground = pg.sprite.Group()
        self.all_platforms = pg.sprite.Group()

        self.all_food = pg.sprite.Group()
        self.all_ants = pg.sprite.Group()
        self.all_corns = pg.sprite.Group()
        self.all_worms = pg.sprite.Group()
        self.all_tomatos = pg.sprite.Group()

        self.flag = pg.sprite.Group()
        self.all_points = pg.sprite.Group()
        self.all_enemies = pg.sprite.Group()

        self.score = 0
        self.win = False
        self.direction = True       # Kako bi na početku svake igre bio okrenut udesno

        # -- UČITAVANJE MAPE --
        for row, tiles in enumerate(self.map.data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Ground(self, col, row, tile)
                if tile == '2':
                    Ground(self, col, row, tile)
                if tile == 'E':
                    Enemy(self, col, row)
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

        self.run()

    def run(self):
        # -- TIJEK IGRE --
        self.playing = True
        self.life = 2
        while self.playing:
            self.events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

    def events(self):
        # -- PROVJERAVANJE IZLASKA IZ IGRE --
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.playing = False
                self.running = False

    def update(self):
        # -- UPDATE --
        self.all_sprites.update()

        # -- PROVJERA AKO SE IGRAČ NALAZI NA PLATFORMI --
        if self.player.vel.y > 0:
            hits_platform = pg.sprite.spritecollide(self.player, self.all_ground, False)
            if hits_platform:
                self.player.pos.y = hits_platform[0].rect.top
                self.player.vel.y = 0
                self.player.jumping = False

        # -- AKO IGRAČ UPADNE U RUPU IGRA SE RESETIRA --
        if self.player.pos.y > (WINDOW_HEIGHT+50):
            self.playing = False

        # -- PROVJERA SAKUPLJENOG I AŽURIRANJE BODOVA --
        if pg.sprite.spritecollide(self.player, self.all_ants, True):
            self.score += 1
            self.points_appear = time.time()
            self.points_image = pg.image.load("Images/Points/Ant_points.png")
            self.draw_points = True
        if pg.sprite.spritecollide(self.player, self.all_corns, True):
            self.score += 5
            self.points_appear = time.time()
            self.points_image = pg.image.load("Images/Points/Corn_points.png")
            self.draw_points = True
            self.y = self.player.rect.y
        if pg.sprite.spritecollide(self.player, self.all_worms, True):
            self.score += 10
            self.points_appear = time.time()
            self.points_image = pg.image.load("Images/Points/Worm_points.png")
            self.draw_points = True
        if pg.sprite.spritecollide(self.player, self.all_tomatos, True):
            self.score += 50
            self.points_appear = time.time()
            self.points_image = pg.image.load("Images/Points/Tomato_points.png")
            self.draw_points = True
        if pg.sprite.spritecollide(self.player, self.flag, True):
            self.win = True
            self.hit = True
            self.pocetak = time.time()
            self.score += 100

        # -- PROVJERA AKO JE DOŠLO DO DOTICAJA SA NEPRIJATELJEM --
        if pg.sprite.spritecollide(self.player, self.all_enemies, True):
            self.life -= 1

            if self.life < 0:
                self.pocetak = time.time()

        # -- AŽURIRANJE KAMERE --
        self.camera.update(self.player)

        # KADA IGRAČ POKUPI ZASTAVICU POBJEDIO JE, TE NAKON 2 SEKUDNE KREĆE NOVA IGRA
        if self.hit:
            if time.time() - self.pocetak > 2:
                self.playing = False

    def draw_grid(self):
        # -- CRTANJE MREŽE --
        for x in range(0, WINDOW_WIDTH, TILE_SIZE):
            pg.draw.line(self.window, WHITE, (x, 0), (x, WINDOW_HEIGHT))
        for y in range(0, WINDOW_HEIGHT, TILE_SIZE):
            pg.draw.line(self.window, WHITE, (0, y), (WINDOW_WIDTH, y))

    def draw(self):
        # -- ISPUNA PROZORA PLAVOM BOJOM (NEBO) --
        self.window.fill(BLUE)

        # -- CRTANJE SVIH SLIČICA --
        for sprite in self.all_sprites:
            self.window.blit(sprite.image, self.camera.apply(sprite))

        # -- KRETANJE NEPRIJATELJA --
        for e in self.all_enemies:
            e.move()

        # -- CRTANJE MREŽE --
        # self.draw_grid()

        # -- ISPIS UKUPNO OSTVARENIH BODOVA --
        score_font = pg.font.Font('freesansbold.ttf', 22)
        score_text = score_font.render("Score: " + str(self.score), True, WHITE)
        self.window.blit(score_text, [10, 10])

        # -- ISPIS PREOSTALIH ŽIVOTA --
        if self.life >= 0:
            life_text = score_font.render("Life: " + str(self.life), True, WHITE)
            self.window.blit(life_text, [10, 40])

        # -- ISPIS BODOVA OSTVARENIH ZA HRANU POKUPLJENU U TOM TRENUTKU --
        if self.draw_points:
            if time.time() - self.points_appear < 1:
                self.window.blit(self.points_image, [score_text.get_width() + 20, 10])

        # -- ISPIS PORUKE ZA POBJEDU --
        if self.win:
            win_font = pg.font.Font('freesansbold.ttf', 40)
            score_text = win_font.render("You won!", True, WHITE)
            score_text_width = score_text.get_width()
            score_text_height = score_text.get_height()
            self.window.blit(score_text, [WINDOW_WIDTH / 2 - score_text_width / 2, WINDOW_HEIGHT / 2 - score_text_height / 2])

        # -- ISPIS PORUKE ZA PORAZ --
        if self.life < 0:
            win_font = pg.font.Font('freesansbold.ttf', 40)
            score_text = win_font.render("You lost!", True, WHITE)
            score_text_width = score_text.get_width()
            score_text_height = score_text.get_height()
            self.window.blit(score_text, [WINDOW_WIDTH / 2 - score_text_width / 2,
                                          WINDOW_HEIGHT / 2 - score_text_height / 2])

            self.player.pos.y = -1000       #  Pomiče igrača van kamere kada izgubi
            if time.time() - self.pocetak > 2:
                self.playing = False

        pg.display.flip()
        pg.display.update()


game = Game()

while game.running:
    game.new()

pg.quit()

