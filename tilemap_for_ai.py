import pygame as pg
from settings import *


class Map:
    def __init__(self):
        self.data = []

        # -- OTVARANJE DATOTEKE SAMO ZA ČITANJE --
        with open("map_for_ai.txt", 'rt') as f:
            for line in f:
                self.data.append(line.strip())

        # -- BROJ STUPACA I REDAKA --
        self.tile_width = len(self.data[0])
        self.tile_height = len(self.data)

        # -- DIMENZIJE CIJELE MAPE --
        self.width = self.tile_width * TILE_SIZE
        self.height = self.tile_height * TILE_SIZE
