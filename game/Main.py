# Main Runnable file for the CE889 Assignment
# Project built by Lewis Veryard and Hugo Leon-Garza
import ctypes
import pygame
import os

# Change to the script's directory so all relative paths work correctly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Set DPI awareness to prevent zoomed-in display on high-DPI screens
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

from GameLoop import GameLoop

pygame.init()
display_info = pygame.display.Info()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

def importConfigFile():
    keys = []
    values = []
    file = open("Files/Config.con", 'r')
    for line in file:
        line_split = line.split(',')
        for individual in line_split:
            individual = individual.replace(" ", "")
            individual = individual.replace("\n", "")
            content = individual.split('=')
            keys.append(content[0])
            values.append(content[1])
    return dict(zip(keys, values))


def start_game_window():
    game = GameLoop()
    game.init(config_data)
    game.main_loop(config_data)


config_data = importConfigFile()
start_game_window()