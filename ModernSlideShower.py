import datetime
import shutil
import imgui
import moderngl
import moderngl_window as mglw
import moderngl_window.context.base
import moderngl_window.context.pyglet.keys
import moderngl_window.meta
import moderngl_window.opengl.vao
import moderngl_window.timers.clock
import io
import rawpy
from moderngl_window.opengl import program
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from PIL import Image, ExifTags
from scipy.ndimage import gaussian_filter
import numpy as np
import math
import os
import sys
import random
import collections
import pyglet
import json
from mpmath import mp

# from scipy.interpolate import BSpline

# Settings names
# HIDE_BORDERS = 1
# TRANSITION_DURATION = 2
# INTER_BLUR = 3
# STARTING_ZOOM_FACTOR = 4
# PIXEL_SIZE = 5

LEVELS_BORDERS_IN_MIN = 0
LEVELS_BORDERS_IN_MAX = 1
LEVELS_BORDERS_GAMMA = 2
LEVELS_BORDERS_OUT_MIN = 3
LEVELS_BORDERS_OUT_MAX = 4
LEVELS_BORDERS_SATURATION = 5

INTERFACE_MODE_GENERAL = 50
INTERFACE_MODE_MENU = 51
INTERFACE_MODE_SETTINGS = 52
INTERFACE_MODE_LEVELS = 53
INTERFACE_MODE_TRANSFORM = 54
INTERFACE_MODE_MANDELBROT = 55

SWITCH_MODE_CIRCLES = 30
# SWITCH_MODE_GESTURES = 31
SWITCH_MODE_COMPARE = 32
SWITCH_MODE_TINDER = 33

BUTTON_STICKING_TIME = 0.3  # After passing this time button acts as temporary.
IMAGE_UN_UNSEE_TIME = 0.2  # Time needed to consider image as been seen


class Actions:
    IMAGE_NEXT = 1
    IMAGE_PREVIOUS = 2
    IMAGE_FOLDER_NEXT = 3
    IMAGE_FOLDER_PREVIOUS = 4
    IMAGE_FOLDER_FIRST = 5
    IMAGE_RANDOM_FILE = 6
    IMAGE_RANDOM_UNSEEN_FILE = 7
    IMAGE_RANDOM_IN_CURRENT_DIR = 8
    IMAGE_RANDOM_DIR_FIRST_FILE = 9
    IMAGE_RANDOM_DIR_RANDOM_FILE = 10
    FILE_MOVE = 11
    FILE_COPY = 12
    FILE_SAVE_WITH_SUFFIX = 13
    FILE_SAVE_AND_REPLACE = 14
    LIST_SAVE_WITH_COMPRESS = 18
    LIST_SAVE_NO_COMPRESS = 19

    PIC_ZOOM_100 = 20
    PIC_ZOOM_FIT = 21
    PIC_ROTATE_RIGHT = 22
    PIC_ROTATE_LEFT = 23
    AUTO_FLIP_TOGGLE = 24
    CENTRAL_MESSAGE_TOGGLE = 25
    REVERT_IMAGE = 26
    TOGGLE_IMAGE_INFO = 27
    APPLY_TRANSFORM = 28
    APPLY_ROTATION_90 = 29

    SWITCH_MODE_CIRCLES = 30
    # SWITCH_MODE_GESTURES = 31
    SWITCH_MODE_COMPARE = 32
    SWITCH_MODE_TINDER = 33

    LEVELS_APPLY = 40
    LEVELS_TOGGLE = 41
    LEVELS_EMPTY = 42
    LEVELS_PREVIOUS = 43
    LEVELS_NEXT_BAND_ROUND = 44
    LEVELS_PREVIOUS_BAND = 45
    LEVELS_NEXT_BAND = 46
    LEVELS_SELECT_RED = 47
    LEVELS_SELECT_GREEN = 48
    LEVELS_SELECT_BLUE = 49

    INTERFACE_MODE_GENERAL = 50
    INTERFACE_MODE_MENU = 51
    INTERFACE_MODE_SETTINGS = 52
    INTERFACE_MODE_LEVELS = 53
    INTERFACE_MODE_TRANSFORM = 54
    INTERFACE_MODE_MANDELBROT = 55

    KEYBOARD_MOVEMENT_ZOOM_IN_ON = 60
    KEYBOARD_MOVEMENT_ZOOM_OUT_ON = 61
    KEYBOARD_MOVEMENT_MOVE_UP_ON = 62
    KEYBOARD_MOVEMENT_MOVE_DOWN_ON = 63
    KEYBOARD_MOVEMENT_MOVE_LEFT_ON = 64
    KEYBOARD_MOVEMENT_MOVE_RIGHT_ON = 65
    KEYBOARD_MOVEMENT_LEFT_BRACKET_ON = 66
    KEYBOARD_MOVEMENT_RIGHT_BRACKET_ON = 67
    KEYBOARD_FLIPPING_FAST_NEXT_ON = 68
    KEYBOARD_FLIPPING_FAST_PREVIOUS_ON = 69

    KEYBOARD_MOVEMENT_ZOOM_IN_OFF = 70
    KEYBOARD_MOVEMENT_ZOOM_OUT_OFF = 71
    KEYBOARD_MOVEMENT_MOVE_UP_OFF = 72
    KEYBOARD_MOVEMENT_MOVE_DOWN_OFF = 73
    KEYBOARD_MOVEMENT_MOVE_LEFT_OFF = 74
    KEYBOARD_MOVEMENT_MOVE_RIGHT_OFF = 75
    KEYBOARD_MOVEMENT_LEFT_BRACKET_OFF = 76
    KEYBOARD_MOVEMENT_RIGHT_BRACKET_OFF = 77
    KEYBOARD_FLIPPING_OFF = 78
    KEYBOARD_MOVEMENT_MOVE_ALL_OFF = 79

    MANDEL_GOOD_ZONES_TOGGLE = 80
    MANDEL_DEBUG_TOGGLE = 81
    MANDEL_GOTO_TEST_ZONE = 82
    MANDEL_TOGGLE_AUTO_TRAVEL_NEAR = 83
    MANDEL_TOGGLE_AUTO_TRAVEL_FAR = 84
    MANDEL_START_NEAR_TRAVEL_AND_ZONES = 85

    ACTION_GENERAL_LEFT = 90
    ACTION_GENERAL_RIGHT = 91
    ACTION_GENERAL_UP = 92
    ACTION_GENERAL_DOWN = 93
    ACTION_GENERAL_SPACE = 94

    WINDOW_SWITCH_FULLSCREEN = 95

    CLOSE_PROGRAM = 100


class Configs:
    HIDE_BORDERS = "Hide borders"
    TRANSITION_DURATION = "Transition duration"
    INTER_BLUR = "Transition Blur"
    STARTING_ZOOM_FACTOR = "Initial zoom"
    PIXEL_SIZE = "Pixel squareness"
    i = [
        HIDE_BORDERS,
        TRANSITION_DURATION,
        INTER_BLUR,
        STARTING_ZOOM_FACTOR,
        PIXEL_SIZE,
        "Save",
        "Close"
    ]


CENRAL_WND_FLAGS = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | \
                   imgui.WINDOW_NO_INPUTS | imgui.WINDOW_NO_COLLAPSE
SIDE_WND_FLAGS = CENRAL_WND_FLAGS | imgui.WINDOW_NO_TITLE_BAR

LIST_FILE_TYPE = 'sldlist'
JPEGTRAN_EXE_PATH = "c:\\Soft\\libjpeg-turbo-gcc\\bin"
JPEGTRAN_EXE_FILE = "jpegtran.exe"
JPEGTRAN_OPTIONS = ' -optimize -rotate {0} -trim -copy all -outfile "{1}" "{1}"'

IMAGE_FILE_TYPES = ('jpg', 'png', 'jpeg', 'gif', 'tif', 'tiff', 'webp')
RAW_FILE_TYPES = ('nef', 'dng', 'arw')
ALL_FILE_TYPES = IMAGE_FILE_TYPES + RAW_FILE_TYPES
EMPTY_IMAGE_LIST = "Empty.jpg"
SAVE_FOLDER = ".\\SaveFolder\\"
Point = collections.namedtuple('Point', ['x', 'y'])
MANDEL_PREZOOM = 4e-3

ORIENTATION_DB = dict([
    (2, [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM]),
    (3, [Image.ROTATE_180, Image.FLIP_TOP_BOTTOM]),
    (4, [Image.FLIP_TOP_BOTTOM, Image.FLIP_TOP_BOTTOM]),
    (5, [Image.ROTATE_90]),
    (6, [Image.ROTATE_270, Image.FLIP_TOP_BOTTOM]),
    (7, [Image.ROTATE_270]),
    (8, [Image.ROTATE_90, Image.FLIP_TOP_BOTTOM])
])

# MAIN_MENU structure:
# name, shortcut label, checkbox condition, action
MAIN_MENU = (
    ('Circle mode', '', lambda x: x.switch_mode == SWITCH_MODE_CIRCLES, None, Actions.SWITCH_MODE_CIRCLES),
    # ('Gesture mode', 'G', lambda x: x.switch_mode == SWITCH_MODE_GESTURES, None, Actions.SWITCH_MODE_GESTURES),
    ('Compare mode', 'X', lambda x: x.switch_mode == SWITCH_MODE_COMPARE, None, Actions.SWITCH_MODE_COMPARE),
    ('Tinder mode', 'J', lambda x: x.switch_mode == SWITCH_MODE_TINDER, None, Actions.SWITCH_MODE_TINDER),
    "--",
    ('Automatic switching (slideshow)', 'A', lambda x: x.autoflip_speed != 0, None, Actions.AUTO_FLIP_TOGGLE),
    "--",
    ('Move file out', 'M', None, None, Actions.FILE_MOVE),
    ('Copy file', 'C', None, None, Actions.FILE_COPY),
    "--",
    ('Rotate image right', '>', None, None, Actions.PIC_ROTATE_RIGHT),
    ('Rotate image left', '<', None, None, Actions.PIC_ROTATE_LEFT),
    ('Save rotation losslessly', 'Enter', None, lambda x: x.lossless_save_possible(), Actions.APPLY_ROTATION_90),
    "--",
    ('Show image information', 'I', lambda x: x.show_image_info != 0, None, Actions.TOGGLE_IMAGE_INFO),
    "--",
    ('Adjust levels', 'L', None, None, Actions.INTERFACE_MODE_LEVELS),
    ('Transform image', 'T', None, None, Actions.INTERFACE_MODE_TRANSFORM),
    ('Settings', 'S', None, None, Actions.INTERFACE_MODE_SETTINGS),
    ('Mandelbrot set', 'U', None, None, Actions.INTERFACE_MODE_MANDELBROT),
    "--",
    ('Keyboard shortcuts', 'F1, H', lambda x: x.central_message_showing, None, Actions.CENTRAL_MESSAGE_TOGGLE),
    ('Quit', 'Esc', None, None, Actions.CLOSE_PROGRAM),
)

# KEYBOARD_SHORTCUTS structure (dict):
# (action, interface_mode, ctrl, shift, alt, key): action_code
KEY = pyglet.window.key
prs = mglw.context.base.BaseKeys.ACTION_PRESS
rls = mglw.context.base.BaseKeys.ACTION_RELEASE
KEYBOARD_SHORTCUTS = {
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.S): Actions.INTERFACE_MODE_SETTINGS,
    (prs, INTERFACE_MODE_SETTINGS, False, False, False, KEY.S): Actions.INTERFACE_MODE_GENERAL,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.T): Actions.INTERFACE_MODE_TRANSFORM,
    (prs, INTERFACE_MODE_TRANSFORM, False, False, False, KEY.T): Actions.INTERFACE_MODE_GENERAL,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.L): Actions.INTERFACE_MODE_LEVELS,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.L): Actions.INTERFACE_MODE_GENERAL,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.U): Actions.INTERFACE_MODE_MANDELBROT,

    # (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.G): Actions.SWITCH_MODE_GESTURES,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.X): Actions.SWITCH_MODE_COMPARE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.J): Actions.SWITCH_MODE_TINDER,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.A): Actions.AUTO_FLIP_TOGGLE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.I): Actions.TOGGLE_IMAGE_INFO,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.C): Actions.FILE_COPY,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.M): Actions.FILE_MOVE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_0): Actions.FILE_MOVE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_INSERT): Actions.FILE_MOVE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.BACKSLASH): Actions.FILE_MOVE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.COMMA): Actions.PIC_ROTATE_LEFT,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.PERIOD): Actions.PIC_ROTATE_RIGHT,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F3): Actions.LIST_SAVE_NO_COMPRESS,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F4): Actions.LIST_SAVE_WITH_COMPRESS,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F9): Actions.FILE_SAVE_WITH_SUFFIX,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F12): Actions.FILE_SAVE_AND_REPLACE,

    (prs, INTERFACE_MODE_GENERAL, True, False, False, KEY.R): Actions.REVERT_IMAGE,

    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.ENTER): Actions.APPLY_ROTATION_90,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.SPACE): Actions.ACTION_GENERAL_SPACE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.RIGHT): Actions.ACTION_GENERAL_RIGHT,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.LEFT): Actions.ACTION_GENERAL_LEFT,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.UP): Actions.ACTION_GENERAL_UP,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.DOWN): Actions.ACTION_GENERAL_DOWN,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.PAGEUP): Actions.IMAGE_FOLDER_PREVIOUS,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.PAGEDOWN): Actions.IMAGE_FOLDER_NEXT,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F5): Actions.IMAGE_RANDOM_FILE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F6): Actions.IMAGE_RANDOM_IN_CURRENT_DIR,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F7): Actions.IMAGE_RANDOM_DIR_FIRST_FILE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F8): Actions.IMAGE_RANDOM_DIR_RANDOM_FILE,

    (prs, INTERFACE_MODE_GENERAL, True, False, False, KEY.SPACE): Actions.KEYBOARD_FLIPPING_FAST_NEXT_ON,
    (prs, INTERFACE_MODE_GENERAL, True, False, False, KEY.RIGHT): Actions.KEYBOARD_FLIPPING_FAST_NEXT_ON,
    (prs, INTERFACE_MODE_GENERAL, True, False, False, KEY.LEFT): Actions.KEYBOARD_FLIPPING_FAST_PREVIOUS_ON,

    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_MULTIPLY): Actions.PIC_ZOOM_FIT,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_DIVIDE): Actions.PIC_ZOOM_100,

    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.RIGHT): Actions.KEYBOARD_MOVEMENT_MOVE_RIGHT_ON,
    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.LEFT): Actions.KEYBOARD_MOVEMENT_MOVE_LEFT_ON,
    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.UP): Actions.KEYBOARD_MOVEMENT_MOVE_UP_ON,
    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.DOWN): Actions.KEYBOARD_MOVEMENT_MOVE_DOWN_ON,

    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.SPACE): Actions.KEYBOARD_FLIPPING_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.RIGHT): Actions.KEYBOARD_FLIPPING_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.LEFT): Actions.KEYBOARD_FLIPPING_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.UP): Actions.KEYBOARD_FLIPPING_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.DOWN): Actions.KEYBOARD_FLIPPING_OFF,

    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,

    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,

    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,
    (prs, INTERFACE_MODE_GENERAL, False, True, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,

    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,

    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,

    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,

    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.SEMICOLON): Actions.LEVELS_TOGGLE,
    (rls, INTERFACE_MODE_LEVELS, False, False, False, KEY.SEMICOLON): Actions.LEVELS_TOGGLE,

    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.ENTER): Actions.LEVELS_APPLY,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.P): Actions.LEVELS_PREVIOUS,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.O): Actions.LEVELS_EMPTY,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.TAB): Actions.LEVELS_NEXT_BAND_ROUND,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.R): Actions.LEVELS_SELECT_RED,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.G): Actions.LEVELS_SELECT_GREEN,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.B): Actions.LEVELS_SELECT_BLUE,

    (prs, INTERFACE_MODE_TRANSFORM, False, False, False, KEY.ENTER): Actions.APPLY_TRANSFORM,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F): Actions.WINDOW_SWITCH_FULLSCREEN,

    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.RIGHT): Actions.KEYBOARD_MOVEMENT_MOVE_RIGHT_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.LEFT): Actions.KEYBOARD_MOVEMENT_MOVE_LEFT_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.UP): Actions.KEYBOARD_MOVEMENT_MOVE_UP_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, True, False, KEY.DOWN): Actions.KEYBOARD_MOVEMENT_MOVE_DOWN_OFF,

    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.LSHIFT): Actions.KEYBOARD_MOVEMENT_MOVE_ALL_OFF,
    (rls, INTERFACE_MODE_GENERAL, False, False, False, KEY.RSHIFT): Actions.KEYBOARD_MOVEMENT_MOVE_ALL_OFF,

    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.RIGHT): Actions.KEYBOARD_MOVEMENT_MOVE_RIGHT_ON,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.LEFT): Actions.KEYBOARD_MOVEMENT_MOVE_LEFT_ON,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.UP): Actions.KEYBOARD_MOVEMENT_MOVE_UP_ON,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.DOWN): Actions.KEYBOARD_MOVEMENT_MOVE_DOWN_ON,

    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.RIGHT): Actions.KEYBOARD_MOVEMENT_MOVE_RIGHT_OFF,
    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.LEFT): Actions.KEYBOARD_MOVEMENT_MOVE_LEFT_OFF,
    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.UP): Actions.KEYBOARD_MOVEMENT_MOVE_UP_OFF,
    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.DOWN): Actions.KEYBOARD_MOVEMENT_MOVE_DOWN_OFF,

    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.BRACKETLEFT): Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_ON,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.BRACKETRIGHT): Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_ON,
    (rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.BRACKETLEFT): Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_OFF,
    (
    rls, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.BRACKETRIGHT): Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_OFF,

    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.Z): Actions.MANDEL_GOOD_ZONES_TOGGLE,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.D): Actions.MANDEL_DEBUG_TOGGLE,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.T): Actions.MANDEL_GOTO_TEST_ZONE,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.A): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_NEAR,
    (prs, INTERFACE_MODE_MANDELBROT, True, False, False, KEY.A): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_FAR,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.SPACE): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_NEAR,
    (prs, INTERFACE_MODE_MANDELBROT, True, False, False, KEY.SPACE): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_FAR,

    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_GENERAL, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_MENU, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_MENU, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_SETTINGS, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_SETTINGS, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_LEVELS, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_TRANSFORM, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_TRANSFORM, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, INTERFACE_MODE_MANDELBROT, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,

}

LEVEL_BORDER_NAMES = [
    'lvl_i_min',
    'lvl_i_max',
    'lvl_gamma',
    'lvl_o_min',
    'lvl_o_max',
    'saturation'
]

POP_MESSAGE_TEXT = [
    "File {file_name}\nwas moved to folder\n{new_folder}",  # 0
    "File {file_name}\nwas copied folder\n{new_folder}",  # 1
    "Image rotated by {angle} degrees. \nPress Enter to save losslessly",  # 2
    "Rotation saved losslessly",  # 3
    "Levels correction applied. \nPress F12 to save image with replacement",  # 4
    "File {file_name}\nsaved with overwrite.",  # 5
    "File saved with new name\n{file_name}",  # 6
    "First image in the list",  # 7
    "Entering folder {current_folder}",  # 8
    "Autoflipping ON, speed = {autoflip_speed:.2f}",  # 9
    "Autoflipping OFF",  # 10
    "Gesture sort mode",  # 11
    "Entering folder {dir_index} of {dir_count}",  # 12
    "File was moved",  # 13
    "File was copied",  # 14
    "Main Menu",  # 15
    "Settings",  # 16
    "Levels adjustment",  # 17
    "Transform image",  # 18
    "Mandelbrot mode",  # 19
    "Compare mode",  # 20
    "All images were shown {many_times}",  # 21
    "Tinder mode",  # 22
    "Settings saved",  # 23
    "No images in folder. Press Esc to close. Switching to Mandelbrot mode in {show_time} seconds",  # 24
]

CENTRAL_MESSAGE = ["",
                   '''
Usage shortcuts


[F1], [H] : show/hide this help window
[Escape], [middle mouse button] : exit program
[drag with left mouse button] : move image on the screen
[drag with right mouse button] : zoom image
[drag with left and right mouse button] : rotate image
[,] [.] : rotate image left and right by 90°
[Space], [right arrow] : show next image
[left arrow] : show previous image
[move mouse in circles clockwise] : show next image
[move mouse in circles counterclockwise] : show previous image   
[page up] : show first image in previous folder
[page down] : show first image in next folder
[M], [backslash] : move image file out of folder tree into '--' subfolder
[C] : copy image file out of folder tree into '++' subfolder
[F12] : save current image as .jpg file with replacement
[F9] : save current image as *_e.jpg file without replacing original
[F3] : save current playlist without compression (plain text)
[F4] : save current playlist with compression
[F5], [arrow up] : move to random image in the list
[F6] : move to random image in current directory
[F7] : move to first image in random directory
[F8] : move to random image in random directory

[S] : show settings window
[A] : start slideshow (automatic image flipping)
[L] : show/hide levels edit interface
[T] : enter image transform mode
[I] : show/hide basic image information
[F] : switch full screen mode
[Ctrl+R] : revert changes to original image

''',
                   '''
No images found

No loadable images found
You may pass a directory as a first argument to this 
script, or put this script in a directory with images.
Press H or F1 to close this window and to show help message.
''',
                   '''
No images found
No valid images found. Maybe all images were moved or deleted.     
Press H or F1 to close this message and to show help message.                
'''
                   ]


def sigmoid(x, mi, mx):
    return mi + (mx - mi) * (lambda t: (1 + 200 ** (-t + 0.5)) ** (-1))((x - mi) / (mx - mi))


def smootherstep_ease(x):
    #  average between smoothstep and smootherstep
    if x <= 0:
        return 0
    elif x >= 1:
        return 1
    else:
        return x * x * (x * (x * (x * 6 - 15) + 8) + 3) / 2


def mix(a, b, amount=.5):
    return a * (1 - amount) + b * amount


def restrict(val, min_val, max_val):
    if val < min_val: return min_val
    if val > max_val: return max_val
    return val


def split_complex(in_value):
    # split_constant = 67108865  # 2 ^ 26 + 1
    # split_constant = 1
    out_real, out_imaginary = [], []
    current_value = in_value
    for _ in range(4):
        with mp.workprec(52):
            t = current_value  # * split_constant
            c_hi = t - (t - current_value)
            out_real.append(float(c_hi.real))
            out_imaginary.append(float(c_hi.imag))
        current_value -= c_hi
    return out_real, out_imaginary


def format_bytes(size):
    # 2**10 = 1024
    power = 2 ** 10
    n = 0
    power_labels = {0: 'B', 1: ' KB', 2: ' MB', 3: ' GB', 4: ' TB'}
    f_size = size
    while f_size > power * 10:
        f_size /= power
        n += 1
    ret = f'{f_size:,.0f}'.replace(",", " ") + power_labels[n] + "  (" + f'{size:,d}'.replace(",", " ") + " B)"
    return ret


class ModernSlideShower(mglw.WindowConfig):
    gl_version = (4, 3)
    title = "ModernSlideShower"
    aspect_ratio = None
    clear_color = (0.0, 0.0, 0.0, 0.0)
    wnd = mglw.context.base.BaseWindow

    start_with_random_image = False
    average_frame_time = 0
    last_image_load_time = 0
    previous_image_duration = 0

    picture_vao = moderngl.VertexArray
    round_vao = moderngl.VertexArray
    file_operation_vao = moderngl.VertexArray
    ret_vertex_buffer = moderngl.Buffer
    jpegtran_exe = os.path.join(JPEGTRAN_EXE_PATH, JPEGTRAN_EXE_FILE)
    image_count = 0
    dir_count = 0
    image_index = 0
    new_image_index = 0
    previous_image_index = 0
    dir_index = 0
    images_in_folder = 0
    index_in_folder = 0
    dir_list = []
    file_list = []
    dir_to_file = []
    file_to_dir = []
    image_categories = []
    tinder_stats = np.zeros(3, dtype=int)
    tinder_last_choice = 0

    unseen_images = set()
    current_image_is_unseen = True
    all_images_seen_times = -1

    imgui_io = imgui.get_io
    imgui_style = imgui.get_style

    image_original_size = Point(0, 0)

    common_path = ""
    im_object = Image

    pic_pos_current = mp.mpc()
    pic_pos_future = mp.mpc()
    pic_move_speed = mp.mpc()
    pic_pos_fading = mp.mpc()

    pic_zoom = .5
    pic_zoom_future = .2
    pic_angle = 0.
    pic_angle_future = 0.
    gl_program_pic = [moderngl.program] * 2
    gl_program_borders = moderngl.program
    gl_program_round = moderngl.program
    gl_program_mandel = [moderngl.program] * 3
    gl_program_crop = moderngl.program
    gl_program_browse = moderngl.program
    gl_program_compare = moderngl.program
    mandel_id = 0
    program_id = 0
    image_texture = moderngl.Texture
    thumb_textures = {0: moderngl.Texture}
    # image_texture_hd = [moderngl.Texture] * 3
    current_texture = moderngl.Texture
    current_texture_old = moderngl.Texture
    curve_texture = moderngl.Texture
    max_keyboard_flip_speed = .3
    mouse_move_atangent = 0.
    mouse_move_atangent_delta = 0.
    mouse_move_cumulative = 0.
    mouse_unflipping_speed = 1.
    last_image_folder = None

    transition_center = (.4, .4)

    reset_frame_timer = True

    interface_mode = INTERFACE_MODE_GENERAL
    switch_mode = SWITCH_MODE_CIRCLES

    split_line = 0

    right_click_start = 0
    left_click_start = 0
    next_message_top = 0
    menu_top = 0
    menu_bottom = 1
    menu_clicked_last_time = 0

    current_frame_start_time = 0
    last_frame_duration = 0

    mandel_stat_buffer = moderngl.Buffer
    mandel_stat_texture_id = 0
    mandel_stat_texture_swapped = False
    mandel_zones_mask = np.empty
    mandel_stat_pull_time = 0
    mandel_good_zones = np.empty((32, 32))
    mandel_look_for_good_zones = False
    mandel_chosen_zone = (0, 0)
    mandel_pos_future = mp.mpc()
    mandel_move_acceleration = mp.mpc()
    mandel_auto_travel_mode = 0
    mandel_auto_travel_limit = 0
    mandel_auto_travel_speed = 1
    mandel_zones_hg = np.empty
    mandel_show_debug = False
    mandel_auto_complexity = 0.
    mandel_auto_complexity_speed = 0.
    mandel_auto_complexity_target = 0.
    mandel_auto_complexity_fill_target = 620
    # mandel_use_beta = False

    configs = {
        Configs.HIDE_BORDERS: 100,
        Configs.TRANSITION_DURATION: .75,
        Configs.INTER_BLUR: 30.,
        Configs.STARTING_ZOOM_FACTOR: .98,
        Configs.PIXEL_SIZE: 0
    }

    config_descriptions = {
        Configs.HIDE_BORDERS: "Hide image borders",
        Configs.TRANSITION_DURATION: "Duration of transition between images",
        Configs.INTER_BLUR: "Blur during transition",
        Configs.STARTING_ZOOM_FACTOR: "Zoom of newly shown image",
        Configs.PIXEL_SIZE: "Pixel shape in case of extreme zoom",
    }

    config_formats = {
        Configs.HIDE_BORDERS: (0, 1000, '%.1f', 2),
        Configs.TRANSITION_DURATION: (0.01, 10, '%.3f', 2),
        Configs.INTER_BLUR: (0, 1000, '%.1f', 4),
        Configs.STARTING_ZOOM_FACTOR: (0, 5, '%.3f', 4),
        Configs.PIXEL_SIZE: (0, 100, '%.1f', 4),
    }

    last_key_press_time = 0
    setting_active = 0
    autoflip_speed = 0.

    run_reduce_flipping_speed = 0.
    run_key_flipping = 0
    key_flipping_next_time = 0.
    run_flip_once = 0
    pressed_mouse = 0

    pop_db = []

    histogram_array = np.empty
    histo_texture = moderngl.texture
    histo_texture_empty = moderngl.buffer

    levels_borders = [[0.] * 4, [1.] * 4, [1.] * 4, [0.] * 4, [1.] * 4, [1.] * 4]
    levels_borders_previous = []

    # levels_open = False
    levels_enabled = True
    levels_edit_band = 3
    levels_edit_parameter = 0
    levels_edit_group = 0

    gesture_mode_timeout = 0

    key_picture_movement = [False] * 8

    transform_mode = 0
    crop_borders_active = 0
    pic_screen_borders = np.array([0.] * 4)
    crop_borders = np.array([0.] * 4)
    resize_xy = 1
    resize_x = 1
    resize_y = 1

    # gesture_sort_mode = False

    transition_stage = 1.

    mouse_buffer = np.array([0., 0.])

    show_image_info = 0
    current_image_file_size = 0

    central_message_showing = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.init_program()

    def init_program(self):
        self.ret_vertex_buffer = self.ctx.buffer(reserve=16)
        # self.ret_vertex_buffer.bind_to_uniform_block(6)
        mp.prec = 120

        self.imgui_io = imgui.get_io()
        self.imgui_style = imgui.get_style()
        self.imgui_io.ini_file_name = np.empty(0).tobytes()

        x = np.arange(0, 32)
        y = x[:, np.newaxis]
        self.mandel_zones_mask = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / 50).astype(float, order='F')

        self.picture_vao = mglw.opengl.vao.VAO("main_image", mode=moderngl.POINTS)

        Image.MAX_IMAGE_PIXELS = 10000 * 10000 * 3
        random.seed()

        def get_program_text(filename, program_text=""):
            if os.path.isfile(filename):
                with open(filename, 'r') as fd:
                    program_text = fd.read()
            return program_text

        picture_program_text = get_program_text('picture.glsl')
        mandel_program_text = get_program_text('mandelbrot.glsl')

        program_single = mglw.opengl.program.ProgramShaders.from_single
        # program_separate = mglw.opengl.program.ProgramShaders.from_separate
        program_source = mglw.opengl.program.ShaderSource

        picture_vertex_text = program_source('PICTURE_VETREX', "picture_vertex", picture_program_text).source
        picture_geometry_text = program_source('PICTURE_GEOMETRY', "picture_geometry", picture_program_text).source
        picture_fragment_text = program_source('PICTURE_FRAGMENT', "picture_fragment", picture_program_text).source
        crop_geometry_text = program_source('CROP_GEOMETRY', "crop_geometry", picture_program_text).source
        crop_fragment_text = program_source('CROP_FRAGMENT', "crop_fragment", picture_program_text).source
        browse_geometry_text = program_source('BROWSE_GEOMETRY', "browse_geometry", picture_program_text).source
        browse_fragment_text = program_source('BROWSE_FRAGMENT', "browse_fragment", picture_program_text).source
        compare_vertex_text = program_source('COMPARE_VETREX', "compare_vertex", picture_program_text).source
        compare_geometry_text = program_source('COMPARE_GEOMETRY', "compare_geometry", picture_program_text).source
        compare_fragment_text = program_source('COMPARE_FRAGMENT', "compare_fragment", picture_program_text).source
        round_vertex_text = program_source('ROUND_VERTEX', "round_vertex", picture_program_text).source
        round_fragment_text = program_source('ROUND_FRAGMENT', "round_fragment", picture_program_text).source

        self.gl_program_pic = [self.ctx.program(vertex_shader=picture_vertex_text,
                                                geometry_shader=picture_geometry_text,
                                                fragment_shader=picture_fragment_text),
                               self.ctx.program(vertex_shader=picture_vertex_text,
                                                geometry_shader=picture_geometry_text,
                                                fragment_shader=picture_fragment_text)
                               ]
        self.gl_program_borders = self.ctx.program(vertex_shader=picture_vertex_text, varyings=['crop_borders'])
        self.gl_program_crop = self.ctx.program(vertex_shader=picture_vertex_text,
                                                geometry_shader=crop_geometry_text,
                                                fragment_shader=crop_fragment_text)
        self.gl_program_browse = self.ctx.program(vertex_shader=picture_vertex_text,
                                                  geometry_shader=browse_geometry_text,
                                                  fragment_shader=browse_fragment_text)
        self.gl_program_compare = self.ctx.program(vertex_shader=compare_vertex_text,
                                                   geometry_shader=compare_geometry_text,
                                                   fragment_shader=compare_fragment_text)
        self.gl_program_round = self.ctx.program(vertex_shader=round_vertex_text, fragment_shader=round_fragment_text)

        p_d = moderngl_window.meta.ProgramDescription

        self.gl_program_mandel = []
        self.gl_program_mandel.append(program_single(p_d(defines={"definition": 0}), mandel_program_text).create())
        self.gl_program_mandel.append(program_single(p_d(defines={"definition": 1}), mandel_program_text).create())
        self.gl_program_mandel.append(program_single(p_d(defines={"definition": 2}), mandel_program_text).create())

        self.mandel_stat_buffer = self.ctx.buffer(reserve=(32 * 64 * 4))
        self.mandel_stat_buffer.bind_to_storage_buffer(4)
        self.histo_texture = self.ctx.texture((256, 5), 1, dtype='u4')
        self.histo_texture.bind_to_image(7, read=True, write=True)
        self.histo_texture_empty = self.ctx.buffer(reserve=(256 * 5 * 4))
        self.histogram_array = np.zeros((5, 256), dtype=np.float32)
        self.generate_round_geometry()
        self.empty_level_borders()
        self.empty_level_borders()
        self.find_jpegtran()
        self.load_settings()

    def post_init(self):
        self.get_images()
        if "-r" in sys.argv or "-F7" in sys.argv:
            self.start_with_random_image = True

        self.window_size = self.wnd.size
 
        if self.image_count == 0:
            self.central_message_showing = 2
            self.switch_interface_mode(INTERFACE_MODE_MANDELBROT)
            return

        # self.unseen_images = set(range(self.image_count))
        self.image_categories = np.zeros(self.image_count, dtype=int)
        self.tinder_stats[1] = self.image_count
        # self.update_tinder_stats()
        self.find_common_path()
        if self.start_with_random_image:
            self.random_image(Actions.IMAGE_RANDOM_DIR_FIRST_FILE if "-F7" in sys.argv else None)
        else:
            self.load_image()
            self.unschedule_pop_message(7)
            self.unschedule_pop_message(8)
        self.current_texture.use(5)
        self.transition_stage = 1
        if "-tinder_mode" in sys.argv:
            self.switch_swithing_mode(SWITCH_MODE_TINDER)
        # self.reset_pic_position(False)

        # --- Temp part. Fill thumbs base
        new_image = Image.new("RGB", (256, 256))
        # text here

        # image_bytes = new_image.tobytes()
        thumb_texture = self.ctx.texture(Point(256, 256), 3, new_image.tobytes())
        self.thumb_textures[0] = thumb_texture

    def previous_level_borders(self):
        self.levels_borders = self.levels_borders_previous
        self.update_levels()

    def empty_level_borders(self):
        self.levels_borders_previous = self.levels_borders
        self.levels_borders = [[0.] * 4, [1.] * 4, [1.] * 4, [0.] * 4, [1.] * 4, [1.] * 4]
        self.update_levels()

    def generate_round_geometry(self):
        points_count = 50

        def generate_points(points_array_x, points_array_y, scale, move):
            points_array = np.empty((100,), dtype=points_array_x.dtype)
            points_array[0::2] = points_array_x
            points_array[1::2] = points_array_y
            points_array *= scale
            points_array += move
            return points_array

        points_array = generate_points(
            np.sin(np.linspace(0., math.tau, points_count, endpoint=False)),
            np.cos(np.linspace(0., math.tau, points_count, endpoint=False)),
            40, 70
        )

        self.round_vao = mglw.opengl.vao.VAO("round", mode=moderngl.POINTS)
        self.round_vao.buffer(points_array.astype('f4'), '2f', ['in_position'])

        points_array = generate_points(
            (np.linspace(2, 0, points_count, endpoint=False)),
            (np.linspace(6, 0, points_count, endpoint=False) % 3),
            30, 20
        )

        self.file_operation_vao = mglw.opengl.vao.VAO("round", mode=moderngl.POINTS)
        self.file_operation_vao.buffer(points_array.astype('f4'), '2f', ['in_position'])

    def update_levels(self, edit_parameter=None):
        if edit_parameter is None:
            for i in range(6):
                self.update_levels(i)
            return

        border_name = LEVEL_BORDER_NAMES[edit_parameter]
        self.gl_program_pic[self.program_id][border_name] = tuple(self.levels_borders[edit_parameter])

    def find_jpegtran(self):
        if not os.path.isfile(self.jpegtran_exe):
            if os.path.isfile(JPEGTRAN_EXE_FILE):
                self.jpegtran_exe = os.path.abspath(JPEGTRAN_EXE_FILE)
            else:
                self.jpegtran_exe = None

    def get_images(self):
        file_arguments = []
        dir_arguments = []
        if len(sys.argv) > 1:
            for argument in sys.argv[1:]:
                if os.path.isdir(argument.rstrip('\\"')):
                    dir_arguments.append(os.path.abspath(argument.rstrip('\\"')))
                if os.path.isfile(argument):
                    file_arguments.append(os.path.abspath(argument))

            if len(dir_arguments):
                [self.scan_directory(directory) for directory in dir_arguments]
            if len(file_arguments):
                if len(dir_arguments) == 0 and len(file_arguments) == 1:
                    if file_arguments[0].lower().endswith(ALL_FILE_TYPES):
                        self.scan_directory(os.path.dirname(file_arguments[0]), file_arguments[0])
                    else:
                        self.scan_file(file_arguments[0])
                else:
                    [self.scan_file(file) for file in file_arguments]
        else:
            self.scan_directory(os.path.abspath('.\\'))

        print(self.image_count, "total images found")

    def find_common_path(self):
        # todo: resolve situation when paths are on different drives
        if len(self.dir_list) == 0:
            return
        if len(self.dir_list) > 10000:
            self.common_path = os.path.commonpath(self.dir_list[::len(self.dir_list) // 100])
        else:
            self.common_path = os.path.commonpath(self.dir_list)
        parent_path = self.dir_list[0]
        if self.common_path == parent_path:
            self.common_path = os.path.dirname(self.common_path)

    def scan_directory(self, dirname, look_for_file=None):
        print("Searching for images in", dirname)
        for root, dirs, files in os.walk(dirname):
            file_count = 0
            first_file = self.image_count
            for f in files:
                if f.lower().endswith(ALL_FILE_TYPES):
                    img_path = os.path.join(root, f)
                    self.image_count += 1
                    file_count += 1
                    self.file_list.append(f)
                    self.file_to_dir.append(self.dir_count)
                    if not self.image_count % 1000:
                        print(self.image_count, "images found", end="\r")
                    if look_for_file:
                        if img_path == look_for_file:
                            self.new_image_index = self.image_count - 1
            if file_count:
                self.dir_list.append(root)
                self.dir_to_file.append([first_file, file_count])
                self.dir_count += 1

    def scan_file(self, filename):
        if filename.lower().endswith(ALL_FILE_TYPES):
            file_dir = os.path.dirname(filename)
            last_dir = ""
            if len(self.dir_list):
                last_dir = self.dir_list[-1]
            if file_dir == last_dir:
                self.file_to_dir.append(self.dir_count - 1)
                self.dir_to_file[-1][1] += 1
            else:
                self.dir_list.append(file_dir)
                self.dir_to_file.append([self.image_count, 1])
                self.file_to_dir.append(self.dir_count)
                self.dir_count += 1
            self.file_list.append(os.path.basename(filename))
            self.image_count += 1

        elif filename.lower().endswith(LIST_FILE_TYPE):
            self.load_list_file(filename)

    def load_list_file(self, filename):
        print("Opening list", filename)
        with open(filename, 'r', encoding='utf-8') as file_handle:
            loaded_list = [current_place.rstrip() for current_place in file_handle.readlines()]
        print("Image list decompression")

        current_dir = ""
        current_dir_index = self.dir_count
        current_dir_file_count = 0
        current_dir_file_start = self.image_count
        previous_line = ""
        for line in loaded_list:
            if line[-1] == "\\":
                if current_dir_file_count:
                    self.dir_to_file.append([current_dir_file_start, current_dir_file_count])
                    self.dir_list.append(current_dir)
                    self.dir_count += 1
                current_dir_file_count = 0
                current_dir = line[:-1]
                current_dir_index = self.dir_count
                current_dir_file_start = self.image_count
            else:
                if line[0] == ":":
                    length_end = 2 if line[2] == " " or line[2] == ":" else 3
                    full_line = previous_line[:int(line[1:length_end])] + line[length_end + 1:]
                    if line[length_end] == ":":
                        full_line += previous_line[len(full_line):]
                else:
                    full_line = line
                previous_line = full_line
                self.file_list.append(full_line)
                self.file_to_dir.append(current_dir_index)
                current_dir_file_count += 1
                self.image_count += 1
        self.dir_to_file.append([current_dir_file_start, current_dir_file_count])
        self.dir_list.append(current_dir)
        self.dir_count += 1
        if "_r" in os.path.basename(filename):
            self.start_with_random_image = True

    def save_list_file(self, compress=True):
        if not os.path.isdir(SAVE_FOLDER):
            try:
                os.makedirs(SAVE_FOLDER)
            except Exception as e:
                print("Could not create folder ", e)

        new_list_file_name = SAVE_FOLDER + 'list_' + datetime.datetime.now().strftime(
            "%Y-%m-%d_%H.%M.%S") + '.' + LIST_FILE_TYPE

        with open(new_list_file_name, 'w', encoding='utf-8') as file_handle:
            for dir_num, dir_name in enumerate(self.dir_list):
                first, count = self.dir_to_file[dir_num]
                file_handle.write("{}\\\n".format(dir_name))
                if not compress:
                    file_handle.writelines("{}\n".format(name) for name in self.file_list[first:first + count])
                else:
                    previous_file_name = ""
                    for file_name in self.file_list[first:first + count]:
                        common_start = 0
                        start_increment = 1
                        common_end = 0
                        for pair in zip(previous_file_name, file_name):
                            if pair[0] == pair[1]:
                                common_start += start_increment
                                common_end -= 1
                            else:
                                start_increment = 0
                                common_end = 0
                        if 4 < common_start - common_end and common_start < 99:
                            if len(previous_file_name) != len(file_name) or common_end == 0:
                                common_end = 1000
                                separator = ' '
                            else:
                                separator = ':'
                            short_name = f":{common_start:d}{separator}{file_name[common_start:common_end]}\n"
                        else:
                            short_name = file_name + "\n"
                        file_handle.write(short_name)
                        previous_file_name = file_name

    def get_file_path(self, index=None):
        if index is None:
            index = self.new_image_index
        if index >= self.image_count:
            index = self.image_count - 1
        if self.image_count == 0:
            dir_index = - 1
        else:
            dir_index = self.file_to_dir[index]
        if dir_index == -1:
            return EMPTY_IMAGE_LIST
        dir_name = self.dir_list[dir_index]
        file_name = self.file_list[index]
        img_path = os.path.join(dir_name, file_name)
        return img_path

    def save_current_settings(self):
        with open("settings.json", 'w') as f:
            json.dump(self.configs, f)
        self.schedule_pop_message(23)

    def load_settings(self):
        if os.path.isfile("settings.json"):
            with open("settings.json", 'r') as f:
                # json.dump(self.configs, f)
                self.configs = json.load(f)

    def reorient_image(self, im):
        image_orientation = 0

        try:
            im_exif = im.getexif()
            image_orientation = im_exif.get(274, 0)
        except (KeyError, AttributeError, TypeError, IndexError):
            # print(KeyError)
            pass

        set_of_operations = ORIENTATION_DB.get(image_orientation, [Image.FLIP_TOP_BOTTOM])
        for operation in set_of_operations:
            im = im.transpose(operation)
        return im

    def release_texture(self, texture):
        if self.image_texture == texture:
            return
        if type(texture) is moderngl.texture.Texture:
            try:
                texture.release()
            except Exception:
                pass

    def prepare_to_mandelbrot(self):
        self.image_original_size = Point(self.wnd.width, self.wnd.height)
        # image_bytes = np.empty(self.wnd.width * self.wnd.height * 3, dtype=np.uint8)
        self.show_image_info = 1
        self.wnd.title = "ModernSlideShower: Mandelbrot mode"
        # self.reset_pic_position()
        self.pic_angle_future = -30
        self.mandel_auto_complexity = 2
        self.pic_zoom = 1e-3
        self.pic_zoom_future = .2
        self.discrete_actions(Actions.SWITCH_MODE_CIRCLES)
        self.unschedule_pop_message(21)

    def load_image(self):
        self.previous_image_duration = self.timer.time - self.last_image_load_time
        # if self.previous_image_duration > .1:
        #     self.unseen_images.discard(self.image_index)
        image_path = self.get_file_path()

        if not os.path.isfile(image_path):
            if image_path == EMPTY_IMAGE_LIST:
                self.central_message_showing = 3
                self.switch_interface_mode(INTERFACE_MODE_MANDELBROT)
            else:
                self.load_next_existing_image()
            return

        try:
            if sum([image_path.lower().endswith(ex) for ex in RAW_FILE_TYPES]):
                f = open(image_path, 'rb', buffering=0)  # This is a workaround for opening cyrillic file names
                thumb = rawpy.imread(f).extract_thumb()
                img_to_read = io.BytesIO(thumb.data)
            else:
                img_to_read = image_path

            with Image.open(img_to_read) as img_buffer:
                if img_buffer.mode == "RGB":
                    self.im_object = self.reorient_image(img_buffer)
                else:
                    self.im_object = self.reorient_image(img_buffer).convert(mode="RGB")
                image_bytes = self.im_object.tobytes()
                self.image_original_size = Point(self.im_object.width, self.im_object.height)
                self.current_image_file_size = os.stat(image_path).st_size
        except Exception as e:
            print("Error loading ", self.get_file_path(), e)
            self.load_next_existing_image()
            return

        self.wnd.title = "ModernSlideShower: " + image_path
        if self.image_texture != self.current_texture:
            self.release_texture(self.image_texture)
        self.current_texture_old = self.current_texture

        self.image_texture = self.ctx.texture(self.image_original_size, 3, image_bytes)
        self.image_texture.repeat_x = False
        self.image_texture.repeat_y = False
        self.image_texture.build_mipmaps()
        self.image_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.current_texture = self.image_texture
        self.previous_image_index = self.image_index
        self.image_index = self.new_image_index
        self.dir_index = self.file_to_dir[self.image_index]
        self.images_in_folder = self.dir_to_file[self.dir_index][1]
        self.index_in_folder = self.image_index + 1 - self.dir_to_file[self.dir_index][0]

        if self.switch_mode == SWITCH_MODE_COMPARE:
            self.program_id = 1 - self.program_id
            if self.mouse_move_cumulative >= 0:
                self.split_line = 0
            else:
                self.split_line = 1
            self.mouse_move_cumulative *= 0.01
        else:
            self.reset_pic_position()
        self.check_folder_change()

        self.reset_frame_timer = True
        self.current_image_is_unseen = True
        self.last_image_load_time = self.timer.time

    def check_folder_change(self):
        current_folder = self.file_to_dir[self.image_index]

        if self.image_index == 0 and not self.interface_mode == INTERFACE_MODE_MANDELBROT:
            self.unschedule_pop_message(8)
            # self.schedule_pop_message(7, 5)
        elif current_folder != self.last_image_folder:
            # self.schedule_pop_message(8, 5, current_folder=self.dir_list[current_folder:current_folder + 1])
            self.schedule_pop_message(12, 5, dir_index=current_folder + 1, dir_count=self.dir_count)
            self.pic_pos_current += self.current_texture.width / 3
        else:
            self.unschedule_pop_message(7)

        self.last_image_folder = current_folder

    def load_next_existing_image(self):
        start_number = self.new_image_index
        increment = 1
        if self.image_index - self.new_image_index == 1:
            increment = -1
        while True:
            self.new_image_index = (self.new_image_index + increment) % self.image_count
            if start_number == self.new_image_index:
                self.file_list[self.new_image_index] = EMPTY_IMAGE_LIST
                # self.central_message_showing = 3
                break
            if os.path.isfile(self.get_file_path()):
                break
        self.load_image()

    def file_copy_move_routine(self, do_copy=False):

        mouse_cumulative = self.mouse_move_cumulative
        split_line = self.split_line
        im_index_current = self.image_index
        im_index_previous = self.previous_image_index
        # im_index = self.image_index

        # if self.switch_mode == SWITCH_MODE_COMPARE:
        #     if abs(self.mouse_move_cumulative) < 50:
        #         im_index = self.previous_image_index
        #
        # if self.switch_mode == SWITCH_MODE_TINDER:
        #     for score, label in ((-1, "..\\--"), (1, "..\\++")):
        #         indices = np.asarray(self.image_categories == score).nonzero()
        #         for i in indices[0][::-1]:
        #             self.file_operation(i, label, do_copy)
        # else:
        #     self.file_operation(im_index, ["-", "+"][do_copy], do_copy)

        for score, label in ((-1, "..\\--"), (1, "..\\++")):
            indices = np.asarray(self.image_categories == score).nonzero()
            for i in indices[0][::-1]:
                self.file_operation(i, label, do_copy)
        
        self.image_categories = np.zeros(self.image_count, dtype=int)
        self.tinder_stats[0] = 0
        self.tinder_stats[2] = 0
        
        if not self.switch_mode == SWITCH_MODE_COMPARE:
            self.mouse_move_cumulative = self.mouse_move_cumulative * .05

        if not do_copy:
            if self.image_count > 0:
                if self.switch_mode == SWITCH_MODE_COMPARE:
                    self.new_image_index = im_index_previous % self.image_count
                    self.load_image()
                    self.new_image_index = im_index_current % self.image_count
                    self.load_image()
                    self.mouse_move_cumulative = mouse_cumulative
                    self.split_line = split_line
                else:
                    self.new_image_index = self.image_index % self.image_count
                    self.load_image()
                # self.schedule_pop_message(0, duration=10, file_name=short_name, new_folder=new_folder)
                self.schedule_pop_message(13, duration=10)
            else:
                self.new_image_index = 0
                self.pic_zoom_future = 1e-12
                self.schedule_pop_message(24, duration=8, show_time=8)
                self.central_message_showing = 3
                # self.load_image()
                # self.switch_interface_mode(Actions.INTERFACE_MODE_MANDELBROT)
        else:
            # self.schedule_pop_message(1, duration=10, file_name=short_name, new_folder=new_folder)
            self.schedule_pop_message(14, duration=10)

    def file_operation(self, im_index, prefix_subfolder, do_copy=False):
        full_name = self.get_file_path(im_index)
        parent_folder = os.path.dirname(full_name)
        own_subfolder = parent_folder[len(self.common_path):]
        if not own_subfolder.startswith("\\"):
            own_subfolder = "\\" + own_subfolder
        new_folder = os.path.join(self.common_path, prefix_subfolder) + own_subfolder
        if not os.path.isdir(new_folder):
            try:
                os.makedirs(new_folder)
            except Exception as e:
                print("Could not create folder", e)

        file_operation = [shutil.move, shutil.copy][do_copy]
        try:
            file_operation(full_name, new_folder)
            if not do_copy:
                # self.delete_image_from_dbs(im_index)
                if not os.listdir(parent_folder):
                    os.rmdir(parent_folder)

        except Exception as e:
            # todo good message here
            print("Could not complete file " + ["move", "copy"][do_copy], e)
            return

    def delete_image_from_dbs(self, im_index):
        dir_index = self.file_to_dir[im_index]
        self.dir_to_file[dir_index][1] -= 1
        for fix_dir in range(dir_index + 1, self.dir_count):
            self.dir_to_file[fix_dir][0] -= 1

        self.file_list.pop(im_index)
        self.file_to_dir.pop(im_index)
        self.unseen_images.discard(im_index)

        self.image_categories = np.delete(self.image_categories, im_index)
        self.update_tinder_stats()
        new_unseen_set = {i if i < im_index else i - 1 for i in self.unseen_images}
        self.unseen_images = new_unseen_set
        self.image_count -= 1
        # print(im_index, self.image_index)
        if im_index < self.image_index:
            self.image_index -= 1

    def lossless_save_possible(self):
        save_possible = self.pic_angle_future % 360 and self.jpegtran_exe and not (self.pic_angle_future % 90)
        return save_possible

    def save_rotation_90(self):
        if self.lossless_save_possible():
            rotate_command = self.jpegtran_exe + JPEGTRAN_OPTIONS.format(round(360 - self.pic_angle_future % 360),
                                                                         self.get_file_path(self.image_index))
            os.system(rotate_command)

            self.load_image()
            self.schedule_pop_message(3)

    def rotate_image_90(self, left=False):
        remainder = self.pic_angle_future % 90
        self.pic_angle_future = round(self.pic_angle_future - remainder + 90 * (left - (not left) * (remainder == 0)))

        if self.pic_angle_future % 180:
            self.pic_zoom_future = min(self.window_size[1] / self.current_texture.width,
                                       self.window_size[0] / self.current_texture.height) * .99
        else:
            self.pic_zoom_future = min(self.window_size[0] / self.current_texture.width,
                                       self.window_size[1] / self.current_texture.height) * .99

        if self.pic_angle_future % 360:  # not at original angle
            if not (self.pic_angle_future % 90) and self.jpegtran_exe:  # but at 90°-ish
                self.schedule_pop_message(2, duration=8000000, angle=360 - self.pic_angle_future % 360)
        else:
            self.unschedule_pop_message(2)

    def mandel_move_to_good_zone(self, frame_time_chunk):
        displacement = self.mandel_pos_future - self.pic_pos_future
        speed_x = smootherstep_ease(abs(displacement.real * self.pic_zoom / self.window_size[0])) * 2
        speed_y = smootherstep_ease(abs(displacement.imag * self.pic_zoom / self.window_size[1])) * 2
        displacement = mp.mpc(displacement.real * speed_x, displacement.imag * speed_y)
        self.mandel_move_acceleration = mix(self.mandel_move_acceleration, displacement, frame_time_chunk)
        self.mandel_move_acceleration *= .99
        # self.mandel_move_acceleration *= 1 - frame_time_chunk
        # self.mandel_move_acceleration += displacement * frame_time_chunk
        self.pic_pos_future += self.mandel_move_acceleration * frame_time_chunk * self.mandel_auto_travel_speed

    def mandel_adjust_complexity(self, frame_time_chunk):
        # chunk10 = (sigmoid(frame_time_chunk * self.mandel_auto_travel_speed, 0, .9))
        # chunk10 = smootherstep_ease(frame_time_chunk * self.mandel_auto_travel_speed)
        chunk10 = restrict(frame_time_chunk * self.mandel_auto_travel_speed, 0, 1)
        zoom_rate = smootherstep_ease(self.pic_zoom_future / 1000) * chunk10

        complexity_goal = self.mandel_auto_complexity_target / 1000
        zoom_rate2 = zoom_rate * (1.02 + 5 * sigmoid(complexity_goal, -.20, .10))
        self.mandel_auto_complexity_speed = mix(self.mandel_auto_complexity_speed, complexity_goal, zoom_rate2)
        if self.mandel_auto_complexity_speed < 0:
            self.mandel_auto_complexity_speed *= .9

        self.mandel_auto_complexity += self.mandel_auto_complexity_speed * zoom_rate
        self.mandel_auto_complexity *= smootherstep_ease(math.log10(self.pic_zoom_future) + 1.5)

    def mandel_travel(self, frame_time_chunk):
        if self.mandel_auto_travel_mode == 2:
            self.pic_zoom_future = self.pic_zoom_future * (1. - frame_time_chunk * self.mandel_auto_travel_speed)
            if self.pic_zoom < .1:
                self.mandel_auto_travel_mode = 1
        elif self.mandel_auto_travel_mode == 1:
            self.pic_zoom_future = self.pic_zoom_future * (1. + frame_time_chunk * self.mandel_auto_travel_speed / 2)
            if self.pic_zoom_future > [2.5e12, 1.3e30][self.mandel_auto_travel_limit]:
                self.mandel_auto_travel_mode = 2
            if self.mandel_good_zones.max() < .001:
                self.mandel_auto_travel_mode = 3  # backstep mode
        elif self.mandel_auto_travel_mode == 3:
            #  Desmos: 1\ +\ \frac{x-4}{\left(x+2\right)^{2}}
            unzoom = 1
            if self.pic_zoom_future < 4:
                unzoom = 1 + (self.pic_zoom_future - 4) / ((self.pic_zoom_future + 2) ** 2)
            unzoom = 1. - frame_time_chunk * self.mandel_auto_travel_speed * unzoom
            self.pic_zoom_future = self.pic_zoom_future * unzoom
            if self.mandel_good_zones.max() > .002:
                self.mandel_auto_travel_mode = 1  # continue forward

    def move_image(self, dx=0, dy=0):
        self.pic_move_speed += mp.mpc(dx, dy) / self.pic_zoom

    def compute_transition(self, frame_time):
        if self.transition_stage < 0:
            self.transition_stage = 0
            return

        transition_time = 1 / min(self.previous_image_duration * .7, self.configs[Configs.TRANSITION_DURATION])  # / 60
        transition_step = (1.2 - self.transition_stage) * transition_time * frame_time  # * 60
        to_target_stage = abs(self.mouse_move_cumulative) / 100 - self.transition_stage
        if to_target_stage > 0:
            self.transition_stage += to_target_stage * frame_time * 10
        self.transition_stage += transition_step
        if self.transition_stage > .99999:
            self.transition_stage = 1
            self.release_texture(self.current_texture_old)
            self.tinder_last_choice = 0

    def compute_movement(self, chunk):
        chunk10 = restrict(chunk * 10, 0, .5)

        self.pic_pos_future += self.check_image_vilible() * chunk10
        self.pic_pos_future += self.pic_move_speed * 10 * chunk10
        self.pic_pos_current = mix(self.pic_pos_current, self.pic_pos_future, chunk10)
        self.pic_move_speed = mix(self.pic_move_speed, 0, chunk10 / 2)

    def compute_zoom_rotation(self, chunk):
        scale_disproportion = abs(self.pic_zoom_future / self.pic_zoom - 1)
        scale_disproportion = smootherstep_ease(scale_disproportion) ** .5 * .1
        rate = scale_disproportion * self.transition_stage ** 2 * chunk * 60
        self.pic_zoom = mix(self.pic_zoom, self.pic_zoom_future, rate)
        self.pic_angle = mix(self.pic_angle, self.pic_angle_future, 5 * chunk)

    def check_image_vilible(self):
        correction_vector = mp.mpc()
        if self.interface_mode == INTERFACE_MODE_MANDELBROT:
            border = 1000 * 1.4
            x_re = self.pic_pos_current.real - 800
            if abs(x_re) > border:
                correction_vector += math.copysign(border, x_re) - x_re

            x_im = self.pic_pos_current.imag
            if abs(x_im) > border:
                correction_vector += 1j * (math.copysign(border, x_im) - x_im)
        else:
            right_edge = 1
            if self.interface_mode in {INTERFACE_MODE_LEVELS, INTERFACE_MODE_TRANSFORM}:
                right_edge = 0.8
            scr_center_w, scr_center_h = self.window_size[0] / 2 * right_edge, self.window_size[1] / 2

            borders = self.pic_screen_borders
            correction_vector += (scr_center_w - borders[0]) * (borders[0] > scr_center_w)
            correction_vector += (scr_center_w - borders[2]) * (borders[2] < scr_center_w)
            correction_vector += (scr_center_h - borders[1]) * (borders[1] > scr_center_h) * 1j
            correction_vector += (scr_center_h - borders[3]) * (borders[3] < scr_center_h) * 1j
            correction_vector *= 4

            if borders[2] - borders[0] < self.window_size[0] * right_edge * 1.1:
                correction_vector += .15 * (self.window_size[0] * right_edge - borders[0] - borders[2])

            if borders[3] - borders[1] < self.window_size[1] * 1.1:
                correction_vector += .15j * (self.window_size[1] - borders[1] - borders[3])

            if 0 < self.crop_borders_active < 5 and self.transform_mode == 2 and self.interface_mode == INTERFACE_MODE_TRANSFORM:
                safe_pix_x, safe_pix_y = self.window_size[0] * right_edge * .1, self.window_size[1] * .1
                if self.crop_borders_active & 1:
                    x1 = borders[self.crop_borders_active - 1]
                    correction_vector += (safe_pix_x - x1) * (x1 < safe_pix_x)
                    correction_vector += (self.window_size[0] - safe_pix_x - x1) * (
                            x1 > self.window_size[0] - safe_pix_x)
                else:
                    y1 = borders[self.crop_borders_active - 1]
                    correction_vector += (safe_pix_y - y1) * (y1 < safe_pix_y) * 1j
                    correction_vector += (self.window_size[1] - safe_pix_y - y1) * (
                            y1 > self.window_size[1] - safe_pix_y) * 1j

        return correction_vector

    def key_flipping(self):
        if self.key_flipping_next_time > self.current_frame_start_time:
            return
        self.run_flip_once = 1 if self.run_key_flipping > 0 else -1
        self.key_flipping_next_time = self.current_frame_start_time + .4 / abs(self.run_key_flipping)

    def first_directory_image(self, direction=0):
        dir_index = (self.file_to_dir[self.image_index] + direction) % self.dir_count
        self.new_image_index = self.dir_to_file[dir_index][0]
        self.load_image()

    def random_image(self, jump_type=Actions.IMAGE_RANDOM_UNSEEN_FILE):
        if jump_type == Actions.IMAGE_RANDOM_FILE:
            self.new_image_index = random.randrange(self.image_count)
        if jump_type == Actions.IMAGE_RANDOM_UNSEEN_FILE:
            if len(self.unseen_images):
                self.new_image_index = random.sample(list(self.unseen_images), 1)[0]
            else:
                self.new_image_index = random.randrange(self.image_count)
        elif jump_type == Actions.IMAGE_RANDOM_IN_CURRENT_DIR:
            dir_index = self.file_to_dir[self.image_index]
            self.new_image_index = self.dir_to_file[dir_index][0] + \
                                   random.randrange(self.dir_to_file[dir_index][1])
        elif jump_type == Actions.IMAGE_RANDOM_DIR_FIRST_FILE:
            dir_index = random.randrange(self.dir_count)
            self.new_image_index = self.dir_to_file[dir_index][0]
        elif jump_type == Actions.IMAGE_RANDOM_DIR_RANDOM_FILE:
            dir_index = random.randrange(self.dir_count)
            self.new_image_index = self.dir_to_file[dir_index][0] + \
                                   random.randrange(self.dir_to_file[dir_index][1])

        self.load_image()
        self.unschedule_pop_message(8)

    def first_image(self):
        self.new_image_index = 0
        self.load_image()

    def apply_transform(self):
        if self.resize_xy == self.resize_x == self.resize_y == 1 and self.pic_angle == 0:
            crop_tuple = (int(self.crop_borders[0]), int(self.crop_borders[1]),
                          self.current_texture.size[0] - int(self.crop_borders[2]) - int(self.crop_borders[0]),
                          self.current_texture.size[1] - int(self.crop_borders[1]) - int(self.crop_borders[3]))
            new_texture = self.ctx.texture((crop_tuple[2], crop_tuple[3]), 3)
            temp_buffer = self.ctx.buffer(reserve=crop_tuple[2] * crop_tuple[3] * 3)
            source_framebuffer = self.ctx.framebuffer([self.current_texture])
            source_framebuffer.read_into(temp_buffer, crop_tuple)
            new_texture.write(temp_buffer)
        else:
            new_texture_size = (int((self.pic_screen_borders[2] - self.pic_screen_borders[0]) / self.pic_zoom),
                                int((self.pic_screen_borders[3] - self.pic_screen_borders[1]) / self.pic_zoom))
            new_texture = self.ctx.texture(new_texture_size, 3)
            render_framebuffer = self.ctx.framebuffer([new_texture])
            render_framebuffer.clear()
            render_framebuffer.use()
            self.gl_program_pic[self.program_id]['process_type'] = 2
            self.picture_vao.render(self.gl_program_pic[self.program_id], vertices=1)
            self.gl_program_pic[self.program_id]['process_type'] = 0
            self.ctx.screen.use()

        self.reset_pic_position()

        # if self.current_texture != self.image_texture:
        #     self.release_texture(self.image_texture)
        self.release_texture(self.current_texture_old)
        self.current_texture_old = self.current_texture
        self.current_texture = new_texture
        new_texture.use(5)

        self.update_position()
        self.switch_interface_mode(INTERFACE_MODE_GENERAL)

    def apply_levels(self):
        new_texture = self.ctx.texture(self.current_texture.size, 3)
        render_framebuffer = self.ctx.framebuffer([new_texture])
        render_framebuffer.clear()
        render_framebuffer.use()
        self.gl_program_pic[self.program_id]['process_type'] = 4
        self.picture_vao.render(self.gl_program_pic[self.program_id], vertices=1)
        self.gl_program_pic[self.program_id]['process_type'] = 0
        self.ctx.screen.use()
        self.current_texture = new_texture
        self.current_texture.use(5)
        # self.levels_open = False
        self.switch_interface_mode(INTERFACE_MODE_GENERAL)
        self.levels_edit_band = 3
        self.empty_level_borders()
        self.schedule_pop_message(4, 8000000)
        # print(self.interface_mode)

    def save_current_texture(self, replace):
        texture_data = self.current_texture.read()
        new_image = Image.frombuffer("RGB", self.current_texture.size, texture_data)
        new_image = new_image.transpose(Image.FLIP_TOP_BOTTOM)
        if self.pic_angle_future % 360:  # not at original angle
            if not (self.pic_angle_future % 90):  # but at 90°-ish
                rotation_step = (self.pic_angle_future % 360) // 90 + 1
                new_image = new_image.transpose(rotation_step)

        # new_file_name = self.get_file_path(self.image_index)
        dir_index = self.file_to_dir[self.image_index]
        dir_name = self.dir_list[dir_index]
        file_name = self.file_list[self.image_index]
        if not replace:
            stripped_file_name = os.path.splitext(file_name)[0]
            file_name = stripped_file_name + "_e" + ".jpg"
        img_path = os.path.join(dir_name, file_name)

        orig_exif = self.im_object.getexif()
        print("Saving under name", img_path)
        new_image.save(img_path, quality=90, exif=orig_exif, optimize=True)
        pop_message = 5
        if not replace:
            pop_message = 6
            dir_index = self.file_to_dir[self.image_index]
            self.dir_to_file[dir_index][1] += 1
            for fix_dir in range(dir_index + 1, self.dir_count):
                self.dir_to_file[fix_dir][0] += 1
            self.file_list.insert(self.image_index + 1, file_name)
            self.file_to_dir.pop(self.image_index)
            self.image_count -= 1
            self.new_image_index = self.image_index % self.image_count

        self.schedule_pop_message(pop_message, duration=8, file_name=os.path.basename(img_path))

    def revert_image(self):
        self.current_texture_old = self.current_texture
        self.current_texture = self.image_texture
        # self.current_texture.use(5)
        self.reset_pic_position()
        self.unschedule_pop_message(4)

    def reset_pic_position(self, full=True, reduced_width=False):
        wnd_width, wnd_height = self.window_size
        if reduced_width:
            wnd_width *= .78

        if self.current_texture != moderngl.Texture:
            self.pic_zoom_future = min(wnd_width / self.current_texture.width,
                                       wnd_height / self.current_texture.height) * .99

        self.mouse_move_cumulative = 0
        self.gesture_mode_timeout = self.timer.time + .2

        if full:
            self.unschedule_pop_message(2)
            self.unschedule_pop_message(4)
            self.program_id = 1 - self.program_id
            self.pic_zoom = self.pic_zoom_future * self.configs[Configs.STARTING_ZOOM_FACTOR]
            self.pic_pos_current = mp.mpc(0)
            self.pic_pos_future = mp.mpc(0)
            self.pic_move_speed = mp.mpc(0)
            self.pic_angle = 0.
            self.pic_angle_future = 0.
            self.resize_xy = 1
            self.resize_x = 1
            self.resize_y = 1
            self.crop_borders *= 0
            self.transition_stage = -1
            self.transition_center = (.3 + .4 * random.random(), .3 + .4 * random.random())

            self.gl_program_pic[self.program_id]['transparency'] = 0

            if self.interface_mode == INTERFACE_MODE_MANDELBROT:
                self.pic_angle_future = -30
                self.pic_zoom = .1
                self.pic_zoom_future = .2

    def move_picture_with_key(self, time_interval):
        dx = time_interval * (self.key_picture_movement[1] - self.key_picture_movement[3]) * 100
        dy = time_interval * (self.key_picture_movement[0] - self.key_picture_movement[2]) * 100
        self.pic_zoom_future *= 1 + (time_interval * (self.key_picture_movement[5] - self.key_picture_movement[4]))
        self.mandel_auto_travel_speed *= 1 + .2 * (
                time_interval * (self.key_picture_movement[7] - self.key_picture_movement[6]))
        self.move_image(dx, -dy)

    def unschedule_pop_message(self, pop_id, force=False):
        for item in self.pop_db:
            if pop_id == item['type']:
                item['end'] = self.timer.time + 1
                if force:
                    self.pop_db.remove(item)

    def update_pop_message(self, pop_id, **kwargs):
        message_text = POP_MESSAGE_TEXT[pop_id].format(**kwargs)
        for item in self.pop_db:
            if pop_id == item['type']:
                item['full_text'] = message_text
                item['start'] = self.timer.time - 1
                item['end'] = self.timer.time + item['duration']

    def schedule_pop_message(self, pop_id, duration=4., shortify=False, **kwargs):
        self.unschedule_pop_message(pop_id, True)
        message_text = POP_MESSAGE_TEXT[pop_id].format(**kwargs)

        new_line = dict.fromkeys(['type', 'alpha', 'duration', 'start', 'end'], 0.)
        new_line['full_text'] = message_text
        new_line['text'] = message_text
        new_line['duration'] = duration
        new_line['type'] = pop_id
        new_line['shortify'] = shortify

        self.pop_db.append(new_line)

    def pop_message_dispatcher(self):
        time = self.current_frame_start_time
        for item in self.pop_db:
            if item['end'] == 0:
                item['start'] = time
                item['end'] = time + item['duration']
            else:
                item['alpha'] = restrict((time - item['start']) * 2, 0, 1) * restrict(item['end'] - time, 0, 1)
                if item['shortify']:
                    cropped_symbols = int((item['start'] + 5 - time) * 20)
                    if cropped_symbols > -1:
                        cropped_symbols = None
                    item['text'] = item['full_text'][0] + item['full_text'][1:cropped_symbols]
            if time > item['end']:
                self.pop_db.remove(item)

    def do_auto_flip(self):
        self.mouse_move_cumulative += self.autoflip_speed
        self.run_reduce_flipping_speed = - .15
        if abs(self.mouse_move_cumulative) > 100:
            self.run_flip_once = 1 if self.mouse_move_cumulative > 0 else -1
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

    def flip_once(self):
        self.new_image_index += self.run_flip_once
        self.new_image_index %= self.image_count
        self.load_image()
        self.run_flip_once = 0

    def reduce_flipping_speed(self):
        if self.run_reduce_flipping_speed < 0:
            self.run_reduce_flipping_speed = self.current_frame_start_time + abs(self.run_reduce_flipping_speed)
        if self.run_reduce_flipping_speed > self.current_frame_start_time:
            return
        self.mouse_move_cumulative -= math.copysign(self.mouse_unflipping_speed, self.mouse_move_cumulative)
        self.mouse_buffer[1] *= .9
        self.mouse_unflipping_speed = self.mouse_unflipping_speed * .8 + .3
        if abs(self.mouse_move_cumulative) < 5 and abs(self.mouse_buffer[1]) < 3:
            self.run_reduce_flipping_speed = 0

    def autoflip_toggle(self):
        if self.autoflip_speed == 0:
            self.autoflip_speed = .5
            self.unschedule_pop_message(10)
            self.schedule_pop_message(9, 8000, True, autoflip_speed=self.autoflip_speed)
            # self.update_pop_message(9, autoflip_speed=self.autoflip_speed)
        else:
            self.autoflip_speed = 0
            self.unschedule_pop_message(9)
            self.schedule_pop_message(10)

    def update_position_mandel(self):
        mandel_complexity = abs(self.pic_angle) / 10 + self.mandel_auto_complexity
        mandel_complexity *= math.log2(self.pic_zoom) * .66 + 10
        vec4_pos_x, vec4_pos_y = split_complex(-self.pic_pos_current / 1000)
        # self.gl_program_mandel[self.mandel_id]['zoom'] = self.pic_zoom
        self.gl_program_mandel[self.mandel_id]['invert_zoom'] = MANDEL_PREZOOM / self.pic_zoom
        self.gl_program_mandel[self.mandel_id]['complexity'] = mandel_complexity
        self.gl_program_mandel[self.mandel_id]['mandel_x'] = tuple(vec4_pos_x)
        self.gl_program_mandel[self.mandel_id]['mandel_y'] = tuple(vec4_pos_y)

    def update_position(self):
        displacement = (self.pic_pos_current.real, self.pic_pos_current.imag)
        self.gl_program_pic[self.program_id]['displacement'] = displacement
        self.gl_program_pic[self.program_id]['zoom_scale'] = self.pic_zoom
        self.gl_program_pic[self.program_id]['angle'] = math.radians(self.pic_angle)
        self.gl_program_pic[self.program_id]['crop'] = tuple(self.crop_borders)
        self.gl_program_pic[self.program_id]['useCurves'] = (self.interface_mode == INTERFACE_MODE_LEVELS) and \
                                                            self.levels_enabled
        self.gl_program_pic[self.program_id]['count_histograms'] = self.interface_mode == INTERFACE_MODE_LEVELS
        self.gl_program_pic[self.program_id]['show_amount'] = self.transition_stage
        self.gl_program_pic[self.program_id]['hide_borders'] = self.configs[Configs.HIDE_BORDERS]
        self.gl_program_pic[self.program_id]['inter_blur'] = self.configs[Configs.INTER_BLUR]
        self.gl_program_pic[self.program_id]['pixel_size'] = self.configs[Configs.PIXEL_SIZE]
        self.gl_program_pic[self.program_id]['transition_center'] = self.transition_center
        self.gl_program_pic[self.program_id]['resize_xy'] = self.resize_xy - 1
        self.gl_program_pic[self.program_id]['resize_x'] = self.resize_x - 1
        self.gl_program_pic[self.program_id]['resize_y'] = self.resize_y - 1
        self.gl_program_pic[self.program_id]['process_type'] = 0

        if self.switch_mode == SWITCH_MODE_COMPARE:
            self.gl_program_pic[1 - self.program_id]['transparency'] = 0
            self.gl_program_pic[self.program_id]['half_picture'] = self.split_line - 1 * (
                    self.mouse_move_cumulative > 0)
        else:
            self.gl_program_pic[1 - self.program_id]['transparency'] = self.transition_stage
            self.gl_program_pic[self.program_id]['half_picture'] = 0

        if self.switch_mode == SWITCH_MODE_TINDER and self.transition_stage < 1:
            fading_pos = (self.pic_pos_fading.real +
                          smootherstep_ease(self.transition_stage) * self.wnd.width / 3 * self.tinder_last_choice,
                          self.pic_pos_fading.imag)
            self.gl_program_pic[1 - self.program_id]['displacement'] = fading_pos

        blur_target = 0
        blur_now = self.gl_program_pic[self.program_id]['transparency'].value
        if self.interface_mode == Actions.INTERFACE_MODE_SETTINGS:
            if Configs.i[self.setting_active] == Configs.INTER_BLUR:
                blur_target = .5
            self.gl_program_pic[self.program_id]['transparency'] = mix(blur_now, blur_target, 0.05)
        else:
            self.gl_program_pic[self.program_id]['transparency'] = 0

        self.gl_program_borders['displacement'] = displacement
        self.gl_program_borders['zoom_scale'] = self.pic_zoom
        self.gl_program_borders['angle'] = math.radians(self.pic_angle)
        self.gl_program_borders['resize_xy'] = self.resize_xy - 1
        self.gl_program_borders['resize_x'] = self.resize_x - 1
        self.gl_program_borders['resize_y'] = self.resize_y - 1
        self.gl_program_borders['crop'] = tuple(self.crop_borders)

        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

        if self.interface_mode == INTERFACE_MODE_TRANSFORM:
            self.gl_program_pic[self.program_id]['process_type'] = 1
            self.gl_program_crop['active_border_id'] = self.crop_borders_active
            self.gl_program_crop['crop'] = tuple(self.crop_borders)
            self.gl_program_crop['zoom_scale'] = self.pic_zoom
            self.gl_program_crop['displacement'] = displacement
            self.gl_program_crop['resize_xy'] = self.resize_xy - 1
            self.gl_program_crop['resize_x'] = self.resize_x - 1
            self.gl_program_crop['resize_y'] = self.resize_y - 1
            self.gl_program_crop['angle'] = math.radians(self.pic_angle)

        # self.gl_program_browse['active_border_id'] = self.crop_borders_active
        # self.gl_program_browse['crop'] = tuple(self.crop_borders)
        self.gl_program_browse['zoom_scale'] = self.pic_zoom
        self.gl_program_browse['displacement'] = displacement
        # self.gl_program_browse['resize_xy'] = self.resize_xy - 1
        # self.gl_program_browse['resize_x'] = self.resize_x - 1
        # self.gl_program_browse['resize_y'] = self.resize_y - 1
        # self.gl_program_browse['angle'] = math.radians(self.pic_angle)

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)
        self.window_size = Point(width, height)
        wnd_size = (width, height)
        half_wnd_size = (width / 2, height / 2)
        self.gl_program_mandel[0]['half_wnd_size'] = half_wnd_size
        self.gl_program_mandel[1]['half_wnd_size'] = half_wnd_size
        self.gl_program_mandel[2]['half_wnd_size'] = half_wnd_size
        self.gl_program_crop['wnd_size'] = wnd_size
        self.gl_program_browse['wnd_size'] = wnd_size
        self.gl_program_round['wnd_size'] = wnd_size
        self.gl_program_borders['wnd_size'] = wnd_size
        self.gl_program_pic[0]['wnd_size'] = wnd_size
        self.gl_program_pic[1]['wnd_size'] = wnd_size
        self.reset_pic_position(False)

    def switch_swithing_mode(self, new_mode, toggle=True):
        if toggle and self.switch_mode == new_mode:
            new_mode = SWITCH_MODE_CIRCLES
        self.switch_mode = new_mode
        self.mouse_buffer *= 0

        self.unschedule_pop_message(11)
        self.unschedule_pop_message(20)
        self.unschedule_pop_message(22)
        if new_mode == SWITCH_MODE_COMPARE:
            self.schedule_pop_message(20, 8000000, True)
            self.run_flip_once = 1
            self.flip_once()
            self.mouse_move_cumulative = 50
        # elif new_mode == SWITCH_MODE_GESTURES:
        #     self.schedule_pop_message(11, 8000000, True)
        elif new_mode == SWITCH_MODE_TINDER:
            self.schedule_pop_message(22, 8000000, True)

    def switch_interface_mode(self, new_mode, toggle=True):
        if toggle and self.interface_mode == new_mode:
            new_mode = INTERFACE_MODE_GENERAL

        self.unschedule_pop_message(self.interface_mode + 14 - 50)

        self.interface_mode = new_mode
        self.mouse_buffer *= 0
        if new_mode != INTERFACE_MODE_GENERAL:
            self.schedule_pop_message(self.interface_mode + 14 - 50, 8000, True)

        if new_mode == INTERFACE_MODE_MANDELBROT:
            self.prepare_to_mandelbrot()

        if new_mode == INTERFACE_MODE_LEVELS:
            self.reset_pic_position(full=False, reduced_width=True)
            self.levels_edit_band = 3
            self.update_levels()

    def mouse_gesture_tracking(self, dx, dy, speed=1, dynamic=True):
        if self.gesture_mode_timeout > self.timer.time:
            self.rearm_gesture_timeout(.1)
            self.mouse_buffer[1] = 0
            return
        mouse_cumulative = self.mouse_move_cumulative
        dy_antiforce = (self.switch_mode != SWITCH_MODE_COMPARE) * math.copysign(dy * speed, self.mouse_move_cumulative)
        self.mouse_move_cumulative += -dx * 1.3 * speed * (1 - abs(self.mouse_buffer[1]) / 100) - dy_antiforce
        self.mouse_buffer[1] -= math.copysign(dx, self.mouse_buffer[1])
        if dynamic:
            self.run_reduce_flipping_speed = - .45
        if abs(self.mouse_move_cumulative) > 100:
            self.run_flip_once = 1 if self.mouse_move_cumulative > 0 else -1
            self.mouse_buffer *= 0
            self.rearm_gesture_timeout()
        if self.switch_mode == SWITCH_MODE_COMPARE:
            if math.copysign(1, self.mouse_move_cumulative) != math.copysign(1, mouse_cumulative):
                direction = int(math.copysign(1, self.mouse_move_cumulative))
                mouse_cumulative = self.mouse_move_cumulative
                for _ in [1, 2]:
                    self.run_flip_once = direction
                    self.flip_once()
                    self.mouse_move_cumulative = mouse_cumulative

        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

        if abs(self.mouse_buffer[1]) > 100:
            if self.autoflip_speed:
                pass
            else:
                self.mark_image_and_switch(self.mouse_buffer[1] < 0, compare_mode=True)
            self.mouse_buffer *= 0
            self.rearm_gesture_timeout()

    def rearm_gesture_timeout(self, timeout=.2):
        self.gesture_mode_timeout = self.timer.time + timeout

    def mouse_tin_tracking(self, dx, dy, speed=1, dynamic=True):
        if self.gesture_mode_timeout > self.timer.time:
            self.rearm_gesture_timeout(.1)
            self.mouse_buffer[1] = 0
            return
        # mouse_cumulative = self.mouse_move_cumulative
        dy_antiforce = (self.switch_mode != SWITCH_MODE_COMPARE) * math.copysign(dy * speed, self.mouse_move_cumulative)
        self.mouse_move_cumulative += dx * 1.3 * speed - dy_antiforce
        self.mouse_buffer[1] -= math.copysign(dx, self.mouse_buffer[1])
        if dynamic:
            self.run_reduce_flipping_speed = - .45
        if abs(self.mouse_move_cumulative) > 100:
            self.mark_image_and_switch(self.mouse_move_cumulative > 0)
            self.rearm_gesture_timeout()

        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

        if abs(self.mouse_buffer[1]) > 100:
            self.run_flip_once = 1 if self.mouse_buffer[1] > 0 else -1
            self.mouse_buffer *= 0
            self.rearm_gesture_timeout()

    def mark_image_and_switch(self, go_right, compare_mode=False):
        if compare_mode and (abs(self.mouse_move_cumulative) < 50):
            working_index = self.previous_image_index
        else:
            working_index = self.image_index
        old_image_category = self.image_categories[working_index]
        new_image_category = restrict(old_image_category + (1 if go_right else -1), -1, 1)
        self.image_categories[working_index] = new_image_category
        self.tinder_last_choice = new_image_category
        # self.update_tinder_stats()
        self.tinder_stats[old_image_category + 1] -= 1
        self.tinder_stats[new_image_category + 1] += 1
        self.mouse_buffer *= 0
        if not compare_mode:
            self.next_unmarked_image()

    def next_unmarked_image(self):
        if self.tinder_stats[1] != 0:
            self.new_image_index = self.image_index
            while self.image_categories[self.new_image_index] != 0:
                self.new_image_index += 1
                self.new_image_index %= self.image_count
            self.load_image()
        else:
            self.run_flip_once = 1

    def update_tinder_stats(self):
        self.tinder_stats[0] = sum(self.image_categories == -1)
        self.tinder_stats[1] = sum(self.image_categories == 0)
        self.tinder_stats[2] = sum(self.image_categories == 1)

    def adjust_transform(self, amount, amount_xy):
        if self.transform_mode == 1 and self.crop_borders_active:  # resizing
            if self.crop_borders_active == 1:
                button_rate = 1 + (self.pressed_mouse - 1) * 40
                button_rate = (1 - (amount_xy[1] * 5 - amount_xy[0]) / 50000 * button_rate)
                self.resize_xy = restrict(self.resize_xy * button_rate - amount / 1000, 0.1, 10)
            elif self.crop_borders_active == 4:
                button_rate_x = -amount_xy[0] / 10000
                button_rate_y = amount_xy[1] / 10000
                self.resize_x = restrict(self.resize_x * (1 - button_rate_x) - button_rate_x, 0.1, 10)
                self.resize_y = restrict(self.resize_y * (1 - button_rate_y) - button_rate_y, 0.1, 10)

        elif self.transform_mode == 2 and self.crop_borders_active:  # cropping
            if self.crop_borders_active == 5:
                self.pic_angle_future += (- amount_xy[0] + amount_xy[1]) / 15
                self.unschedule_pop_message(2)
                return
            border_id = self.crop_borders_active - 1
            crops = self.crop_borders
            work_axis = border_id & 1
            crop_amount = amount_xy[work_axis] / 10
            actual_pic_size = Point((self.pic_screen_borders[2] - self.pic_screen_borders[0]) / self.pic_zoom,
                                    (self.pic_screen_borders[3] - self.pic_screen_borders[1]) / self.pic_zoom)

            crop_direction = 1 if border_id in {0, 3} else -1
            opposite_border_id = border_id ^ 2
            axis_speed = actual_pic_size[work_axis] / self.window_size[work_axis] + 5
            button_rate = 1 + (self.pressed_mouse - 1) * axis_speed * 5
            crop_change_amount = crop_amount * button_rate * crop_direction
            crops[border_id] += crop_change_amount
            if self.pic_screen_borders[2 + work_axis] - self.pic_screen_borders[0 + work_axis] < 5 * self.pic_zoom:
                crops[opposite_border_id] -= crop_change_amount + .3 * self.pic_zoom

    def adjust_levels(self, amount):
        edit_parameter = self.levels_edit_parameter - 1
        if edit_parameter > 2 and self.levels_edit_band < 4:
            amount = - amount
        if self.levels_edit_band == 4:
            self.levels_borders[5][edit_parameter] = restrict(
                self.levels_borders[5][edit_parameter] * (1 - amount), 0.01, 10)
            self.update_levels(5)
        elif edit_parameter == 2:
            self.levels_borders[edit_parameter][self.levels_edit_band] = restrict(
                self.levels_borders[edit_parameter][self.levels_edit_band] * (1 + amount), 0.01, 10)
        else:
            new_value = self.levels_borders[edit_parameter][self.levels_edit_band] + amount
            self.levels_borders[edit_parameter][self.levels_edit_band] = restrict(new_value, 0, 1)
        self.update_levels(edit_parameter)

    def mouse_circle_tracking(self):
        self.mouse_buffer *= .9
        mouse_speed = np.linalg.norm(self.mouse_buffer) ** .6
        move_atangent_new = math.atan(self.mouse_buffer[0] / (self.mouse_buffer[1] + .000111))
        move_atangent_delta_new = self.mouse_move_atangent - move_atangent_new
        if abs(move_atangent_delta_new) > .15:
            move_atangent_delta_new = self.mouse_move_atangent_delta * .5
        self.mouse_move_atangent_delta *= .9
        self.mouse_move_atangent_delta += move_atangent_delta_new * .1
        self.mouse_move_atangent = move_atangent_new

        if self.mouse_move_atangent_delta * self.mouse_move_cumulative > 0:
            self.mouse_unflipping_speed = .5
        mouse_move_delta = self.mouse_move_atangent_delta * (
                4 - math.copysign(2, self.mouse_move_atangent_delta * self.mouse_move_cumulative)) * mouse_speed

        if self.autoflip_speed != 0:
            self.autoflip_speed += mouse_move_delta * .01
        else:
            self.mouse_move_cumulative += mouse_move_delta
            self.mouse_move_cumulative *= .999
        self.run_reduce_flipping_speed = - .15
        if abs(self.mouse_move_cumulative) > 100:
            self.run_flip_once = 1 if self.mouse_move_cumulative > 0 else -1
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

    def mouse_position_event(self, x, y, dx, dy):
        self.mouse_buffer += [dx, dy]

        if self.interface_mode == INTERFACE_MODE_GENERAL:
            null_zoom = min(self.window_size[0] / self.current_texture.width,
                        self.window_size[1] / self.current_texture.height) * .99
            zoom_ratio = 1 - null_zoom / max(self.pic_zoom_future, 0.001)
            if zoom_ratio > 0.1:
                self.move_image(dx, -dy)
            else:
                if self.switch_mode == SWITCH_MODE_CIRCLES:
                    self.mouse_circle_tracking()
                elif self.switch_mode == SWITCH_MODE_COMPARE:
                    self.mouse_gesture_tracking(dx, dy, speed=.2, dynamic=False)
                elif self.switch_mode == SWITCH_MODE_TINDER:
                    self.mouse_tin_tracking(dx, dy)

        elif self.interface_mode == INTERFACE_MODE_MENU:
            if self.menu_bottom > 1:
                self.mouse_buffer[1] = restrict(self.mouse_buffer[1], self.menu_top, self.menu_bottom)
                self.imgui.mouse_position_event(20, self.mouse_buffer[1], 0, 0)

        elif self.interface_mode == INTERFACE_MODE_SETTINGS:
            if abs(self.mouse_buffer[1]) > 150:
                self.setting_active += 1 if self.mouse_buffer[1] > 0 else -1
                self.setting_active = self.setting_active % (len(self.configs) + 2)
                self.mouse_buffer *= 0

        elif self.interface_mode == INTERFACE_MODE_LEVELS:
            if abs(self.mouse_buffer[0]) > 200:
                self.levels_edit_group = 1 if self.mouse_buffer[0] > 0 else 0
                self.mouse_buffer *= 0
            if abs(self.mouse_buffer[1]) > 150:
                self.levels_edit_band += 1 if self.mouse_buffer[1] > 0 else -1
                self.levels_edit_band = restrict(self.levels_edit_band, 0, 4)
                self.mouse_buffer *= 0

        elif self.interface_mode == INTERFACE_MODE_TRANSFORM:
            if self.transform_mode == 2:  # cropping
                if self.mouse_buffer[0] > 200:
                    self.crop_borders_active = 3
                    self.mouse_buffer *= 0

                if self.mouse_buffer[0] < -200:
                    self.crop_borders_active = 1
                    self.mouse_buffer *= 0

                if self.mouse_buffer[1] > 200:
                    self.crop_borders_active = 2
                    self.mouse_buffer *= 0

                if self.mouse_buffer[1] < -200:
                    self.crop_borders_active = 4
                    self.mouse_buffer *= 0

        elif self.interface_mode == INTERFACE_MODE_MANDELBROT:
            pass

    def visual_move(self, dx, dy):
        if self.pressed_mouse == 1:
            self.move_image(dx, -dy)
        elif self.pressed_mouse == 2:
            self.pic_zoom_future *= 1 / (1 + 1.02 ** (- dx + dy)) + .5
        elif self.pressed_mouse == 3:
            self.pic_angle_future += (- dx + dy) / 15
            self.unschedule_pop_message(2)

    def mouse_drag_event(self, x, y, dx, dy):
        # self.mouse_buffer += [dx, dy]
        self.mouse_buffer[1] += dy * .5
        amount = (dy * 5 - dx) / 1500
        self.right_click_start -= (abs(dx) + abs(dy)) * .01
        if self.interface_mode == INTERFACE_MODE_GENERAL:            
            if self.pressed_mouse == 1:
                if self.mouse_buffer[1] > 50:
                    self.show_image_info += 1
                    self.mouse_buffer[1] = 0
                elif self.mouse_buffer[1] < -50:
                    self.show_image_info -= 1
                    self.mouse_buffer[1] = 0
                if self.show_image_info == 3:
                    # self.switch_interface_mode(INTERFACE_MODE_MENU, False)
                    self.interface_mode = INTERFACE_MODE_MENU
                    self.imgui.mouse_position_event(20, 5, 0, 0)
                self.show_image_info = restrict(self.show_image_info, 0, 2)
            else:
                self.visual_move(dx, dy)
        elif self.interface_mode == INTERFACE_MODE_MENU:
            # self.mouse_position_event(x, y, dx, dy)
            # self.imgui.mouse_position_event(20, self.mouse_buffer[1], 0, 0)
            # self.mouse_buffer[1] += dy * .5
            self.mouse_buffer[1] = restrict(self.mouse_buffer[1], self.menu_top, self.menu_bottom)
            self.imgui.mouse_position_event(20, self.mouse_buffer[1], 0, 0)
            if self.menu_bottom > 1:
                self.imgui.mouse_press_event(20, self.mouse_buffer[1], 1)

        elif self.interface_mode == INTERFACE_MODE_SETTINGS:
            # if self.setting_active == 0: return
            self.configs[Configs.i[self.setting_active]] = restrict(self.configs[Configs.i[self.setting_active]] *
                                                                    (1 - amount) - amount / 1000 *
                                                                    self.config_formats[Configs.i[self.setting_active]][
                                                                        1],
                                                                    self.config_formats[Configs.i[self.setting_active]][
                                                                        0],
                                                                    self.config_formats[Configs.i[self.setting_active]][
                                                                        1])

        elif self.interface_mode == INTERFACE_MODE_LEVELS:
            self.adjust_levels(amount)

        elif self.interface_mode == INTERFACE_MODE_TRANSFORM:
            if self.transform_mode == 3:
                self.visual_move(dx, dy)
            else:
                self.adjust_transform(amount, (dx, dy))

        elif self.interface_mode == INTERFACE_MODE_MANDELBROT:
            self.visual_move(dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        if self.interface_mode == INTERFACE_MODE_MANDELBROT:
            self.pic_zoom_future *= 1.3 if y_offset < 0 else .7
            return
        elif self.interface_mode == INTERFACE_MODE_LEVELS:
            if y_offset < 0:
                self.discrete_actions(Actions.LEVELS_NEXT_BAND)
            else:
                self.discrete_actions(Actions.LEVELS_PREVIOUS_BAND)
        elif self.interface_mode == INTERFACE_MODE_TRANSFORM:
            self.transform_mode -= 1 if y_offset > 0 else -1
            self.transform_mode = restrict(self.transform_mode, 1, 3)
            self.crop_borders_active = 0
        else:
            self.run_flip_once = 1 if y_offset < 0 else -1

    def mouse_press_event(self, x, y, button):
        button_code = 4 if button == 3 else button
        self.pressed_mouse = self.pressed_mouse | button_code
        if self.pressed_mouse == 4:
            self.wnd.close()
            return

        if self.pressed_mouse == 2:
            self.right_click_start = self.timer.time
        else:
            self.right_click_start = 0

        if self.interface_mode == INTERFACE_MODE_MENU:
            # self.imgui.mouse_position_event(20, self.mouse_buffer[1] / 5, None, None)
            self.imgui.mouse_press_event(20, self.mouse_buffer[1], button)
            pass
        elif self.interface_mode == INTERFACE_MODE_GENERAL:
            if button == 1:
                self.left_click_start = self.timer.time
                self.show_image_info = 1
            if self.pressed_mouse == 3:
                self.random_image()
        elif self.interface_mode == INTERFACE_MODE_LEVELS and self.levels_enabled:
            if self.levels_edit_band == 4:
                self.levels_edit_parameter = (self.levels_edit_group * 2 + self.pressed_mouse) % 5
            elif self.pressed_mouse < 4:
                self.levels_edit_parameter = (self.levels_edit_group * 3 + self.pressed_mouse) % 6
        elif self.interface_mode == INTERFACE_MODE_TRANSFORM:
            if self.transform_mode == 1:
                if self.pressed_mouse == 1:
                    self.crop_borders_active = 1
                elif self.pressed_mouse == 2:
                    self.crop_borders_active = 1
                elif self.pressed_mouse == 3:
                    self.crop_borders_active = 4
            elif self.transform_mode == 2:
                if self.pressed_mouse == 3:
                    self.crop_borders_active = 5

        if self.pressed_mouse == 6:
            self.autoflip_toggle()

    def mouse_release_event(self, x: int, y: int, button: int):
        # self.imgui.mouse_release_event(x, y, button)
        button_code = 4 if button == 3 else button
        if button_code == 2 and self.timer.time - self.right_click_start < .15:
            if self.interface_mode == INTERFACE_MODE_GENERAL:
                self.switch_interface_mode(INTERFACE_MODE_MENU)
            elif self.interface_mode != INTERFACE_MODE_MANDELBROT:
                self.switch_interface_mode(INTERFACE_MODE_GENERAL)
        if self.interface_mode == INTERFACE_MODE_GENERAL and button == 1:
            self.show_image_info = 0
            if self.timer.time - self.left_click_start < .2:
                self.reset_pic_position(False)

        if self.interface_mode == INTERFACE_MODE_MENU and button == 1:
            # self.imgui.mouse_press_event(20, self.mouse_buffer[1], button)
            self.imgui.mouse_release_event(20, self.mouse_buffer[1], button)
            self.menu_bottom = -1
            # self.switch_interface_mode(INTERFACE_MODE_GENERAL)
            # self.menu_clicked_last_time = self.timer.time

        if self.interface_mode == INTERFACE_MODE_SETTINGS:
            if self.pressed_mouse == 1:
                if self.setting_active == len(self.configs):
                    self.save_current_settings()
                elif self.setting_active == len(self.configs) + 1:
                    self.switch_interface_mode(Actions.INTERFACE_MODE_GENERAL)

        self.pressed_mouse = self.pressed_mouse & ~button_code
        self.right_click_start = 0
        self.levels_edit_parameter = 0
        if self.interface_mode == INTERFACE_MODE_TRANSFORM:
            if self.transform_mode == 1:
                self.crop_borders_active = 0

    def general_arrow_events(self, action):
        if self.switch_mode == Actions.SWITCH_MODE_TINDER:
            if action == Actions.ACTION_GENERAL_LEFT:
                self.mark_image_and_switch(0)
            elif action == Actions.ACTION_GENERAL_RIGHT:
                self.mark_image_and_switch(1)
            elif action == Actions.ACTION_GENERAL_UP:
                self.run_key_flipping = -1
            elif action == Actions.ACTION_GENERAL_DOWN:
                self.run_key_flipping = 1
            elif action == Actions.ACTION_GENERAL_SPACE:
                self.random_image(Actions.IMAGE_RANDOM_UNSEEN_FILE)

        else:
            if action == Actions.ACTION_GENERAL_LEFT:
                self.run_key_flipping = -1
            elif action == Actions.ACTION_GENERAL_RIGHT:
                self.run_key_flipping = 1
            elif action == Actions.ACTION_GENERAL_SPACE:
                self.run_key_flipping = 1
            elif action == Actions.ACTION_GENERAL_UP:
                self.random_image(Actions.IMAGE_RANDOM_UNSEEN_FILE)
            elif action == Actions.ACTION_GENERAL_DOWN:
                self.first_directory_image(0)

    def discrete_actions(self, action):

        if Actions.ACTION_GENERAL_LEFT <= action <= Actions.ACTION_GENERAL_SPACE:
            self.general_arrow_events(action)

        elif action == Actions.IMAGE_NEXT:
            self.run_key_flipping = 1

        elif action == Actions.IMAGE_PREVIOUS:
            self.run_key_flipping = -1

        elif action == Actions.WINDOW_SWITCH_FULLSCREEN:
            self.wnd.fullscreen = not self.wnd.fullscreen
            self.wnd.mouse_exclusivity = self.wnd.fullscreen

        elif action == Actions.FILE_COPY:
            self.file_copy_move_routine(do_copy=True)

        elif action == Actions.FILE_MOVE:
            self.file_copy_move_routine(do_copy=False)

        elif action == Actions.IMAGE_FOLDER_NEXT:
            self.first_directory_image(1)

        elif action == Actions.IMAGE_FOLDER_PREVIOUS:
            self.first_directory_image(-1)

        elif action == Actions.IMAGE_FOLDER_FIRST:
            self.first_directory_image(0)

        elif action == Actions.IMAGE_RANDOM_FILE:
            self.random_image(Actions.IMAGE_RANDOM_FILE)

        elif action == Actions.IMAGE_RANDOM_UNSEEN_FILE:
            self.random_image(Actions.IMAGE_RANDOM_UNSEEN_FILE)

        elif action == Actions.IMAGE_RANDOM_IN_CURRENT_DIR:
            self.random_image(Actions.IMAGE_RANDOM_IN_CURRENT_DIR)

        elif action == Actions.IMAGE_RANDOM_DIR_FIRST_FILE:
            self.random_image(Actions.IMAGE_RANDOM_DIR_FIRST_FILE)

        elif action == Actions.IMAGE_RANDOM_DIR_RANDOM_FILE:
            self.random_image(Actions.IMAGE_RANDOM_DIR_RANDOM_FILE)

        elif action == Actions.FILE_SAVE_WITH_SUFFIX:
            self.save_current_texture(False)

        elif action == Actions.FILE_SAVE_AND_REPLACE:
            self.save_current_texture(True)

        elif action == Actions.LIST_SAVE_WITH_COMPRESS:
            self.save_list_file()

        elif action == Actions.LIST_SAVE_NO_COMPRESS:
            self.save_list_file(compress=False)

        elif action == Actions.PIC_ZOOM_100:
            self.pic_zoom_future = 1

        elif action == Actions.PIC_ZOOM_FIT:
            self.reset_pic_position(False)

        elif action == Actions.PIC_ROTATE_LEFT:
            self.rotate_image_90(True)

        elif action == Actions.PIC_ROTATE_RIGHT:
            self.rotate_image_90()

        elif action == Actions.AUTO_FLIP_TOGGLE:
            self.autoflip_toggle()

        elif action == Actions.CENTRAL_MESSAGE_TOGGLE:
            self.central_message_showing = 0 if self.central_message_showing else 1

        elif action == Actions.REVERT_IMAGE:
            self.revert_image()

        elif action == Actions.TOGGLE_IMAGE_INFO:
            self.show_image_info = (self.show_image_info + 1) % 3

        elif action == Actions.APPLY_TRANSFORM:
            self.apply_transform()

        elif action == Actions.APPLY_ROTATION_90:
            self.save_rotation_90()

        elif action == Actions.SWITCH_MODE_CIRCLES:
            self.switch_swithing_mode(SWITCH_MODE_CIRCLES)

        # elif action == Actions.SWITCH_MODE_GESTURES:
        #     self.switch_swithing_mode(SWITCH_MODE_GESTURES)

        elif action == Actions.SWITCH_MODE_COMPARE:
            self.switch_swithing_mode(SWITCH_MODE_COMPARE)

        elif action == Actions.SWITCH_MODE_TINDER:
            self.switch_swithing_mode(SWITCH_MODE_TINDER)

        elif action == Actions.LEVELS_APPLY:
            self.apply_levels()

        elif action == Actions.LEVELS_TOGGLE:
            if self.timer.time - self.last_key_press_time > BUTTON_STICKING_TIME:
                self.levels_enabled = not self.levels_enabled

        elif action == Actions.LEVELS_EMPTY:
            self.empty_level_borders()

        elif action == Actions.LEVELS_PREVIOUS:
            self.previous_level_borders()

        elif action == Actions.LEVELS_NEXT_BAND_ROUND:
            self.levels_edit_band = (self.levels_edit_band + 1) % 5

        elif action == Actions.LEVELS_PREVIOUS_BAND:
            self.levels_edit_band = restrict(self.levels_edit_band - 1, 0, 4)

        elif action == Actions.LEVELS_NEXT_BAND:
            self.levels_edit_band = restrict(self.levels_edit_band + 1, 0, 4)

        elif action == Actions.LEVELS_SELECT_RED:
            self.levels_edit_band = 3 if self.levels_edit_band == 0 else 0

        elif action == Actions.LEVELS_SELECT_GREEN:
            self.levels_edit_band = 3 if self.levels_edit_band == 1 else 1

        elif action == Actions.LEVELS_SELECT_BLUE:
            self.levels_edit_band = 3 if self.levels_edit_band == 2 else 2

        elif action == Actions.CLOSE_PROGRAM:
            self.wnd.close()

        elif INTERFACE_MODE_GENERAL <= action <= INTERFACE_MODE_MANDELBROT:
            self.switch_interface_mode(action)

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON:
            self.key_picture_movement[5] = True

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON:
            self.key_picture_movement[4] = True

        elif action == Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_ON:
            self.key_picture_movement[6] = True

        elif action == Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_ON:
            self.key_picture_movement[7] = True

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_UP_ON:
            self.key_picture_movement[0] = True

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_DOWN_ON:
            self.key_picture_movement[2] = True

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_LEFT_ON:
            self.key_picture_movement[1] = True

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_RIGHT_ON:
            self.key_picture_movement[3] = True

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF:
            self.key_picture_movement[5] = False

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF:
            self.key_picture_movement[4] = False

        elif action == Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_OFF:
            self.key_picture_movement[6] = False

        elif action == Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_OFF:
            self.key_picture_movement[7] = False

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_UP_OFF:
            self.key_picture_movement[0] = False

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_DOWN_OFF:
            self.key_picture_movement[2] = False

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_LEFT_OFF:
            self.key_picture_movement[1] = False

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_RIGHT_OFF:
            self.key_picture_movement[3] = False

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_ALL_OFF:
            self.key_picture_movement[0] = False
            self.key_picture_movement[1] = False
            self.key_picture_movement[2] = False
            self.key_picture_movement[3] = False

        elif action == Actions.KEYBOARD_FLIPPING_FAST_NEXT_ON:
            self.run_key_flipping = 4

        elif action == Actions.KEYBOARD_FLIPPING_FAST_PREVIOUS_ON:
            self.run_key_flipping = -4

        elif action == Actions.KEYBOARD_FLIPPING_OFF:
            self.run_key_flipping = 0

        elif action == Actions.MANDEL_GOOD_ZONES_TOGGLE:
            self.mandel_look_for_good_zones = not self.mandel_look_for_good_zones
            self.mandel_move_acceleration = 0

        elif action == Actions.MANDEL_DEBUG_TOGGLE:
            self.mandel_show_debug = not self.mandel_show_debug
            self.central_message_showing = 0

        elif action == Actions.MANDEL_GOTO_TEST_ZONE:
            self.mandel_goto_test_zone()

        elif action == Actions.MANDEL_TOGGLE_AUTO_TRAVEL_NEAR:
            self.mandel_auto_travel_mode = 0 if self.mandel_auto_travel_mode else 1
            self.mandel_auto_travel_limit = 0

        elif action == Actions.MANDEL_TOGGLE_AUTO_TRAVEL_FAR:
            self.mandel_auto_travel_mode = 0 if self.mandel_auto_travel_mode else 1
            self.mandel_auto_travel_limit = 1

        elif action == Actions.MANDEL_START_NEAR_TRAVEL_AND_ZONES:
            self.mandel_look_for_good_zones = True
            self.mandel_auto_travel_limit = 0
            self.mandel_auto_travel_mode = 1

    def key_event(self, key, action, modifiers):
        # self.imgui.key_event(key, action, modifiers)
        find_key = (action, self.interface_mode, modifiers.ctrl, modifiers.shift, modifiers.alt, key)
        found_action = KEYBOARD_SHORTCUTS.get(find_key)
        # print("find_key = ", find_key)
        # print("found_action = ", found_action)

        if found_action:
            self.discrete_actions(found_action)

        if action == self.wnd.keys.ACTION_PRESS:
            self.last_key_press_time = self.timer.time

    def unseen_image_routine(self):
        if self.current_frame_start_time > self.last_image_load_time + IMAGE_UN_UNSEE_TIME:
            self.unseen_images.discard(self.image_index)
            self.current_image_is_unseen = False

            if len(self.unseen_images) == 0:
                self.unseen_images = set(range(self.image_count))
                self.current_image_is_unseen = True
                self.all_images_seen_times += 1
                if self.all_images_seen_times > 1:
                    self.schedule_pop_message(21, 8000, many_times=f'{self.all_images_seen_times:d} times')
                elif self.all_images_seen_times == 1:
                    self.schedule_pop_message(21, 8000, many_times='')

    def read_and_clear_histo(self):
        hg_raw = self.histo_texture.read()
        hg = np.frombuffer(hg_raw, dtype=np.uint32).reshape(5, 256).copy()
        hg[4] = hg[1] + hg[2] + hg[3]
        self.histogram_array = hg.astype(np.float32)
        self.histo_texture.write(self.histo_texture_empty)

    def mandel_goto_test_zone(self):
        self.pic_zoom_future = 2 ** 100
        self.pic_zoom = self.pic_zoom_future
        self.pic_pos_future = mp.mpc("1.2926031417650986620649279496560493",
                                     "0.43839664593583653478781074400281723")
        self.pic_pos_future *= 1000
        self.pic_pos_current = self.pic_pos_future
        self.pic_angle_future = 26
        self.pic_angle = 26
        self.mandel_move_acceleration *= 0
        self.pic_move_speed *= 0

    def mandel_stat_analysis(self):
        # start = self.timer.time
        sum_tex = self.mandel_stat_buffer.read()
        hg = np.frombuffer(sum_tex, dtype=np.uint32).reshape(64, 32).copy(order='F')
        self.mandel_zones_hg = hg / (hg.max() + 1)
        self.mandel_stat_buffer.clear()

        dark_zones, light_zones = np.vsplit(np.flipud(self.mandel_zones_hg), 2)
        light_zones_sum, dark_zones_sum = float(np.sum(light_zones)), float(np.sum(dark_zones))
        complexity = -(light_zones_sum - dark_zones_sum)
        self.mandel_auto_complexity_target = self.mandel_auto_complexity_fill_target - complexity

        best_zones_table = light_zones * dark_zones * (1 + dark_zones)
        best_zones_table_blurred = 20 * gaussian_filter(best_zones_table, sigma=6)

        self.mandel_good_zones = best_zones_table * self.mandel_zones_mask * best_zones_table_blurred
        self.mandel_stat_texture_swapped = False

        if best_zones_table.T[self.mandel_chosen_zone] < best_zones_table.max() * .8:
            chosen_zone = np.unravel_index(np.argmax(best_zones_table), best_zones_table.shape, order='F')
            self.mandel_chosen_zone = (chosen_zone[0].item(), chosen_zone[1].item())

            reposition = mp.mpc((chosen_zone[0].item() / 16 - 1), (1 - chosen_zone[1].item() / 16))
            reposition_random = mp.mpc(random.random() - .5, random.random() - .5)
            reposition += reposition_random * (.1 + 2 / (1 + .5 * abs(math.log10(self.pic_zoom_future))))

            reposition = mp.mpc(reposition.real * self.window_size[0],
                                reposition.imag * self.window_size[1])
            reposition /= self.pic_zoom
            self.mandel_pos_future = self.pic_pos_future - reposition

    def mandelbrot_routine(self, time: float, frame_time_chunk: float):
        if self.mandel_look_for_good_zones or self.mandel_show_debug or self.mandel_auto_travel_mode:
            self.mandel_stat_analysis()
        if self.mandel_auto_travel_mode:
            self.mandel_travel(frame_time_chunk)
            self.mandel_adjust_complexity(frame_time_chunk)
        if self.mandel_look_for_good_zones:
            self.mandel_move_to_good_zone(frame_time_chunk)

        if self.pic_zoom > 4e28:
            self.mandel_id = 0
        elif self.pic_zoom > 2e13:
            self.mandel_id = 1
        else:
            self.mandel_id = 2

    def render(self, time=0.0, frame_time=0.0):
        self.current_frame_start_time, self.last_frame_duration = self.timer.next_frame()
        if self.reset_frame_timer:
            self.last_frame_duration = 1 / 60
        # frame_time_chunk = (frame_time / 3 + .02) ** .5 - .14  # Desmos: \ \left(\frac{x}{3}+.02\right)^{.5}-.14
        frame_time_chunk = (self.last_frame_duration / 3 + .02) ** .5 - .14
        if frame_time_chunk > .5:
            frame_time_chunk = .5

        self.wnd.swap_buffers()
        self.wnd.clear()

        if self.interface_mode == INTERFACE_MODE_MANDELBROT:
            self.mandelbrot_routine(self.current_frame_start_time, frame_time_chunk)
            self.update_position_mandel()
            self.picture_vao.render(self.gl_program_mandel[self.mandel_id], vertices=1)
        else:
            self.update_position()
            if self.interface_mode == INTERFACE_MODE_LEVELS:
                self.read_and_clear_histo()
            if self.transition_stage < 1:
                if type(self.current_texture_old) is moderngl.texture.Texture:
                    self.current_texture_old.use(5)
                    self.picture_vao.render(self.gl_program_pic[1 - self.program_id], vertices=1)

            self.current_texture.use(5)
            self.picture_vao.render(self.gl_program_pic[self.program_id], vertices=1)

            if self.switch_mode == SWITCH_MODE_COMPARE:
                if type(self.current_texture_old) is moderngl.texture.Texture:
                    self.program_id = 1 - self.program_id
                    self.update_position()
                    self.gl_program_compare['line_position'] = 1 - self.split_line
                    self.gl_program_pic[self.program_id]['half_picture'] = self.split_line - 1 * (
                            self.mouse_move_cumulative < 0)
                    self.split_line = mix(self.split_line, self.mouse_move_cumulative / 100 % 1, .2)
                    self.current_texture_old.use(5)
                    self.picture_vao.render(self.gl_program_pic[self.program_id], vertices=1)
                    self.program_id = 1 - self.program_id
                    self.picture_vao.render(self.gl_program_compare, vertices=1)

            self.picture_vao.transform(self.gl_program_borders, self.ret_vertex_buffer, vertices=1)
            self.pic_screen_borders = np.frombuffer(self.ret_vertex_buffer.read(), dtype=np.float32)

            if self.interface_mode == INTERFACE_MODE_TRANSFORM:
                self.picture_vao.render(self.gl_program_crop, vertices=1)

            pic_w = self.pic_screen_borders[2] - self.pic_screen_borders[0]
            pic_h = self.pic_screen_borders[3] - self.pic_screen_borders[1]
            small_zoom_w = pic_w < self.window_size[0] * .9
            small_zoom_h = pic_h < self.window_size[1] * .9
            small_zoom = self.pic_screen_borders[2] != self.pic_screen_borders[0]
            small_zoom &= small_zoom_w or small_zoom_h

            if small_zoom:
                thumb_size = max(pic_w, pic_h)
                thumb_rows = int(self.window_size.y / thumb_size / 2 + .5)
                thumb_rows = min(7, thumb_rows)
                row_elements = int(self.window_size.x / thumb_size / 2 + .5)
                row_elements = min(11, row_elements)
                for row in range(-thumb_rows, thumb_rows + 1):
                    self.thumb_textures[0].use(8)
                    self.gl_program_browse['row'] = row
                    self.gl_program_browse['row_elements'] = row_elements
                    self.picture_vao.render(self.gl_program_browse, vertices=1)

            self.ctx.enable_only(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
            if self.switch_mode != SWITCH_MODE_COMPARE:
                self.round_vao.render(self.gl_program_round)
            # if self.gesture_sort_mode and not (self.transform_mode or self.levels_open or self.setting_active):
            if self.switch_mode != SWITCH_MODE_CIRCLES and self.interface_mode == INTERFACE_MODE_GENERAL:
                self.gl_program_round['finish_n'] = self.mouse_buffer[1]
                if self.switch_mode == SWITCH_MODE_COMPARE:
                    self.gl_program_round['move_to_right'] = bool((self.mouse_move_cumulative % 100) > 50)
                else:
                    self.gl_program_round['move_to_right'] = False
                self.file_operation_vao.render(self.gl_program_round)

        self.render_ui()
        self.average_frame_time = mix(self.average_frame_time, self.last_frame_duration, .01)

        self.compute_movement(frame_time_chunk)
        self.compute_zoom_rotation(frame_time_chunk)
        self.pop_message_dispatcher()
        if self.transition_stage < 1:
            self.compute_transition(self.last_frame_duration)
        if abs(self.run_reduce_flipping_speed):
            self.reduce_flipping_speed()
        if abs(self.run_flip_once):
            self.flip_once()
        if self.run_key_flipping:
            self.key_flipping()
        if True in self.key_picture_movement:
            self.move_picture_with_key(frame_time_chunk)
        if self.autoflip_speed != 0 and self.pressed_mouse == 0 and self.interface_mode == INTERFACE_MODE_GENERAL:
            self.do_auto_flip()
        if self.current_image_is_unseen:
            self.unseen_image_routine()
        if self.pic_zoom < 1e-6:
            self.discrete_actions(Actions.CLOSE_PROGRAM)

    def render_ui(self):
        imgui.new_frame()
        self.imgui_style.alpha = 1

        if self.central_message_showing:
            self.imgui_central_message()

        # Settings window
        if self.interface_mode == INTERFACE_MODE_SETTINGS:
            self.imgui_settings()

        # Levels window
        if self.interface_mode == INTERFACE_MODE_LEVELS:
            self.imgui_levels()

        self.next_message_top = 10

        if self.interface_mode == INTERFACE_MODE_MANDELBROT:
            self.imgui_mandelbrot()
        elif self.show_image_info:
            self.imgui_image_info()

        if self.interface_mode == INTERFACE_MODE_TRANSFORM:
            self.imgui_transforms()

        # Upper stats in tinder mode
        if self.switch_mode == SWITCH_MODE_TINDER:
            self.imgui_tinder_stats()

        # Menu window
        if self.interface_mode == INTERFACE_MODE_MENU:
            self.imgui_menu()

        # Dual labels in compare mode
        if self.switch_mode == SWITCH_MODE_COMPARE:
            # self.imgui_tinder_stats_compact()
            self.imgui_tinder_stats(True)
            self.imgui_compare()

        if len(self.pop_db) > 0:
            self.imgui_popdb()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def imgui_settings(self):
        pos_x, pos_y = self.imgui_io.display_size.x * .5, self.imgui_io.display_size.y * .5
        imgui.set_next_window_position(pos_x, pos_y, 1, pivot_x=.5, pivot_y=0.5)
        imgui.set_next_window_bg_alpha(.9)
        imgui.begin("Settings", False, CENRAL_WND_FLAGS)

        for key in self.configs.keys():
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, .2 + .5 * (Configs.i[self.setting_active] == key), .2,
                                   .2)
            imgui.slider_float(self.config_descriptions[key], self.configs[key],
                               self.config_formats[key][0],
                               self.config_formats[key][1],
                               self.config_formats[key][2],
                               self.config_formats[key][3])
            imgui.pop_style_color()
        imgui.dummy(10, 10)
        imgui.push_style_color(imgui.COLOR_BUTTON, .2 + .5 * (self.setting_active == len(self.configs)), .2, .2)
        imgui.small_button("Save settings as default")
        imgui.pop_style_color()
        # imgui.same_line(spacing=50)
        imgui.push_style_color(imgui.COLOR_BUTTON, .2 + .5 * (self.setting_active == len(self.configs) + 1), .2, .2)
        imgui.small_button("Close")
        imgui.pop_style_color()
        imgui.end()

    def imgui_menu(self):
        menu_clicked = False
        self.imgui_style.alpha = .9
        self.next_message_top += 10
        self.menu_top = self.next_message_top
        imgui.set_next_window_position(10, self.next_message_top)
        imgui.set_next_window_bg_alpha(.9)

        imgui.open_popup("main_menu")

        if imgui.begin_popup("main_menu"):
            imgui.set_window_font_scale(1.6)

            for item in MAIN_MENU:

                if item == "--":
                    imgui.separator()
                else:
                    checked = False if item[2] is None else item[2](self)
                    enabled = True if item[3] is None else item[3](self)

                    clicked, selected = imgui.menu_item(item[0], item[1], checked, enabled)
                    if clicked:
                        self.discrete_actions(item[4])
                        menu_clicked = True
            
            if self.menu_bottom <  0:
                menu_clicked = True
            self.next_message_top += imgui.get_window_height()
            self.menu_bottom = self.next_message_top
            imgui.end_popup()

        if menu_clicked:
            if self.interface_mode == INTERFACE_MODE_MENU:
                self.menu_bottom = 1
                self.show_image_info = 0
                self.switch_interface_mode(INTERFACE_MODE_GENERAL)

    def imgui_central_message(self):
        pos_x, pos_y = self.imgui_io.display_size.x * .5, self.imgui_io.display_size.y * .5
        imgui.set_next_window_position(pos_x, pos_y, 1, pivot_x=0.5, pivot_y=0.5)
        _, caption, *message = CENTRAL_MESSAGE[self.central_message_showing].splitlines()
        imgui.set_next_window_size(0, 0)
        imgui.set_next_window_bg_alpha(.8)
        imgui.begin(caption, False, CENRAL_WND_FLAGS)
        imgui.set_window_font_scale(1.5)
        for text in message:
            imgui.text(text)
        imgui.end()

    def imgui_levels(self):
        line_height = imgui.get_text_line_height_with_spacing()

        def add_cells_with_text(texts, list_of_selected):
            for n, text in enumerate(texts):
                letters_blue = .5 if n in list_of_selected else 1
                imgui.push_style_color(imgui.STYLE_ALPHA, 1, 1, letters_blue)
                imgui.text(text)
                imgui.pop_style_color()
                imgui.next_column()

        pos_x, pos_y = self.imgui_io.display_size.x, self.imgui_io.display_size.y * .5
        imgui.set_next_window_position(pos_x, pos_y, 1, pivot_x=1, pivot_y=0.5)
        imgui.set_next_window_size(0, 0)

        hg_size = (self.imgui_io.display_size.x * .2, (self.imgui_io.display_size.y - 9 * line_height * 2) / 6)

        imgui.set_next_window_bg_alpha(.8)
        imgui.begin("Levels settings", True, SIDE_WND_FLAGS)

        hg_names = ["Gray", "Red", "Green", "Blue", "RGB"]
        hg_colors = [[0.8, 0.8, 0.8],
                     [0.7, 0.3, 0.3],
                     [0.3, 0.7, 0.3],
                     [0.3, 0.3, 0.7],
                     [0.7, 0.7, 0.7]]

        for hg_num in range(5):
            bg_color = .3
            if hg_num - 1 == self.levels_edit_band:
                bg_color = .5
            imgui.text(hg_names[hg_num] + " Histogram")
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, bg_color, bg_color, bg_color, bg_color)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, hg_colors[hg_num][0], hg_colors[hg_num][1],
                                   hg_colors[hg_num][2])
            imgui.plot_histogram("", self.histogram_array[hg_num], graph_size=hg_size)
            imgui.pop_style_color(2)

        self.imgui_style.alpha = .2 + .6 * self.levels_enabled

        imgui.columns(3)
        add_cells_with_text(["", "Input", "   Output"], [self.levels_edit_group + 1])

        imgui.columns(6)
        selected = range(self.levels_edit_group * 3 + 1, self.levels_edit_group * 3 + 4)
        add_cells_with_text(["", " min", " max", "gamma", " min", " max"], selected)

        def add_one_slider(column, row_number, wide, column_is_active, band_active, sell_active):
            if sell_active and band_active:
                bg_color = (.7, .2, .2)
            elif column_is_active and band_active:
                bg_color = (.2, .2, .6)
            else:
                bg_color = (.2, .2, .2)
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, bg_color[0], bg_color[1], bg_color[2])
            imgui.slider_float("", self.levels_borders[column][row_number], 0, [1, 10][wide], '%.2f', [1, 3][wide])
            imgui.pop_style_color()
            imgui.next_column()

        def add_grid_elements(row_name, row_number, active_columns):
            imgui.text(row_name)
            imgui.next_column()
            for column in range(5):
                add_one_slider(column, row_number, column == 2, column in active_columns,
                               self.levels_edit_band == row_number, column == self.levels_edit_parameter - 1)

        active_columns = range(self.levels_edit_group * 3, self.levels_edit_group * 3 + 3)
        add_grid_elements("Red", 0, active_columns)
        add_grid_elements("Green", 1, active_columns)
        add_grid_elements("Blue", 2, active_columns)
        add_grid_elements("RGB", 3, active_columns)

        imgui.columns(2)
        add_cells_with_text(["Pre-levels Saturation", "Post-levels Saturation"], [self.levels_edit_group])
        imgui.columns(4)
        selected = self.levels_edit_group * 2
        add_cells_with_text(["Hard", "Soft"] * 2, [selected, selected + 1])

        for column in range(4):
            add_one_slider(5, column, True, column // 2 == self.levels_edit_group,
                           self.levels_edit_band == 4, column == self.levels_edit_parameter - 1)

        imgui.set_window_font_scale(1)
        imgui.end()

    def imgui_show_info_window(self, text_list, win_name):
        imgui.set_next_window_position(10, self.next_message_top, 1, pivot_x=0, pivot_y=0)
        imgui.begin(win_name, True, SIDE_WND_FLAGS)
        for text_ in text_list:
            imgui.text(text_)
        self.next_message_top += imgui.get_window_height()
        imgui.end()

    def imgui_image_info(self):
        self.imgui_style.alpha = .7
        info_text = ["Folder: " + os.path.dirname(self.get_file_path(self.image_index))]
        self.imgui_show_info_window(info_text, "Directory")
        im_mp = self.image_original_size.x * self.image_original_size.y / 1000000
        # dir_index = self.file_to_dir[self.image_index]
        # dirs_in_folder = self.dir_to_file[dir_index][1]
        # index_in_folder = self.image_index + 1 - self.dir_to_file[dir_index][0]
        info_text = ["File name: " + os.path.basename(self.get_file_path(self.image_index)),
                     "File size: " + format_bytes(self.current_image_file_size),
                     "Image size: " + f"{self.image_original_size.x} x {self.image_original_size.y}",
                     "Image size: " + f"{im_mp:.2f} megapixels",
                     "Image # (current folder): " + f"{self.index_in_folder:d} of {self.images_in_folder:d}",
                     "Image # (all list): " + f"{self.image_index + 1:d} of {self.image_count:d}",
                     "Folder #: " + f"{self.dir_index + 1:d} of {self.dir_count:d}"]

        self.imgui_show_info_window(info_text, "File props")

        if self.show_image_info == 2:
            info_text = [f"Current zoom: {self.pic_zoom:.2f}",
                         f"Visual rotation angle: {self.pic_angle:.2f}°",
                         ]
            self.imgui_show_info_window(info_text, "File props extended")
            self.next_message_top += 10

    def imgui_mandelbrot(self):
        if self.show_image_info:
            self.imgui_style.alpha = .7
            info_text = [
                "Mandelbrot mode",
                "Position:",
                " x: " + mp.nstr(-self.pic_pos_current.real / 1000, int(math.log10(self.pic_zoom) + 5)),
                " y: " + mp.nstr(-self.pic_pos_current.imag / 1000, int(math.log10(self.pic_zoom) + 5)),
                f"Log2 Zoom: {math.log(self.pic_zoom, 2):.1f}",
                f"Actual Zoom: {self.pic_zoom:,.1f}",
                f"Actual Zoom: {self.pic_zoom:.1e}",
                f"Base complexity: {abs(self.pic_angle) / 10:.2f}",
                f"Auto complexity: {abs(self.mandel_auto_complexity) / 10:.2f}",
                f"Resulting complexity: {self.gl_program_mandel[self.mandel_id]['complexity'].value:.2f}",
                f"Auto travel speed: {self.mandel_auto_travel_speed:.2f}",
                f"FPS: {1 / (self.average_frame_time + .0001):.2f}"
            ]
            self.imgui_show_info_window(info_text, "File props")
            self.next_message_top += 10

        if self.mandel_show_debug:
            self.imgui_show_info_window(["Automatic Mandelbrot travel mode"], "Mandel Travel")
            self.next_message_top += 10
            pos_x, pos_y = self.imgui_io.display_size.x * .5, self.imgui_io.display_size.y * .5
            imgui.set_next_window_position(pos_x, pos_y, 1, pivot_x=.5, pivot_y=0.5)
            imgui.set_next_window_size(0, 0)

            hg_size = (self.imgui_io.display_size.x * .98, (self.imgui_io.display_size.y / 42))

            imgui.set_next_window_bg_alpha(.8)
            imgui.begin("Levels sett", True, SIDE_WND_FLAGS)
            hg_colors = [0.8, 0.8, 0.8]
            for hg_num in range(32):
                bg_color = .3
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, bg_color, bg_color, bg_color, bg_color)
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, hg_colors[0], hg_colors[1], hg_colors[2])
                imgui.plot_histogram("", self.mandel_good_zones[hg_num].astype(np.float32), graph_size=hg_size,
                                     scale_min=0, scale_max=self.mandel_good_zones.max())
                imgui.pop_style_color(2)
            imgui.set_window_font_scale(1)
            imgui.end()
        return self.next_message_top

    def imgui_transforms(self):
        next_message_top_r = 100
        pos_x = self.imgui_io.display_size.x

        def push_bg_color(border_id, transform_mode):
            if border_id == self.crop_borders_active - 1 and self.transform_mode == transform_mode:
                r = .7
            else:
                r = 0.2
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, r, .2, .2)

        imgui.set_next_window_position(pos_x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)
        imgui.set_next_window_size(0, 0)

        imgui.begin("Image tranformations", True, SIDE_WND_FLAGS)
        imgui.text("Image tranformations")
        self.imgui_style.alpha = .8
        imgui.set_window_font_scale(1.6)
        next_message_top_r += imgui.get_window_height()
        imgui.end()

        imgui.set_next_window_position(pos_x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)

        bg_color = 1  # if self.transform_mode == 1 else 0.5
        imgui.push_style_color(imgui.COLOR_BORDER, .5, .5, .5, .1)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .2, .2, .3, bg_color)

        imgui.begin("Image info", True, SIDE_WND_FLAGS)
        self.imgui_style.alpha = .7
        imgui.text("Image info")
        next_message_top_r += imgui.get_window_height()

        new_texture_size = Point((self.pic_screen_borders[2] - self.pic_screen_borders[0]) / self.pic_zoom,
                                 (self.pic_screen_borders[3] - self.pic_screen_borders[1]) / self.pic_zoom)
        new_image_width = self.current_texture.width - int(self.crop_borders[0]) - int(self.crop_borders[1])
        new_image_width *= self.resize_xy * self.resize_x
        new_image_height = self.current_texture.height - int(self.crop_borders[2]) - int(self.crop_borders[3])
        new_image_height *= self.resize_xy * self.resize_y
        im_mp_1 = self.image_original_size.x * self.image_original_size.y / 1000000
        im_mp_2 = self.current_texture.width * self.current_texture.height / 1000000
        im_mp_3 = int(new_image_width) * int(new_image_height) / 1000000
        im_mp_3 = int(new_texture_size.x) * int(new_texture_size.y) / 1000000
        imgui.set_window_font_scale(1)
        imgui.text("Original image size: " + f"{self.image_original_size.x} x {self.image_original_size.y}")
        imgui.text(f"Original image size: {im_mp_1:.2f} megapixels")
        imgui.text("Current image size: " + f"{self.current_texture.width} x {self.current_texture.height}")
        imgui.text(f"Current image size: {im_mp_2:.2f} megapixels")
        imgui.text("New image size: " + f"{int(new_texture_size.x)} x {int(new_texture_size.y)}")
        imgui.text(f"New image size: {im_mp_3:.2f} megapixels")

        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.end()

        bg_color = 1 if self.transform_mode == 1 else 0.5
        imgui.push_style_color(imgui.COLOR_BORDER, bg_color, bg_color, bg_color, bg_color)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .2, .2, .2 + .1 * bg_color, bg_color)

        imgui.set_next_window_position(pos_x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)
        imgui.set_next_window_bg_alpha(.8)
        imgui.begin("Resize window", True, SIDE_WND_FLAGS)
        self.imgui_style.alpha = .2 + .6 * self.levels_enabled * (self.transform_mode == 1)

        imgui.pop_style_color()
        imgui.pop_style_color()

        imgui.text("")
        imgui.text("Resize image")

        push_bg_color(0, 1)
        imgui.push_item_width(130)
        imgui.slider_float("Scale whole image", self.resize_xy * 100, 10, 1000, '%.2f', 6.66)
        imgui.pop_style_color()
        push_bg_color(3, 1)
        imgui.slider_float("Scale width", self.resize_x * 100, 10, 1000, '%.2f', 6.66)
        imgui.pop_style_color()
        push_bg_color(3, 1)
        imgui.slider_float("Scale height", self.resize_y * 100, 10, 1000, '%.2f', 6.66)
        imgui.pop_style_color()
        previous_message_top_r = next_message_top_r
        next_message_top_r += imgui.get_window_height()
        imgui.end()

        imgui.set_next_window_position(pos_x, previous_message_top_r, 1, pivot_x=1, pivot_y=0.0)

        bg_color = 1 if self.transform_mode == 1 else 0.5
        imgui.push_style_color(imgui.COLOR_BORDER, bg_color, bg_color, bg_color, bg_color)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .5, .5, bg_color, bg_color)

        imgui.begin("Resize label", True, SIDE_WND_FLAGS)
        imgui.text("Image resize")
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.end()

        bg_color = 1 if self.transform_mode == 2 else 0.5
        imgui.push_style_color(imgui.COLOR_BORDER, bg_color, bg_color, bg_color, bg_color)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .2, .2, .2 + .1 * bg_color, bg_color)

        imgui.set_next_window_position(pos_x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)
        imgui.set_next_window_bg_alpha(.8)
        imgui.begin("Crop image", True, SIDE_WND_FLAGS)
        self.imgui_style.alpha = .2 + .6 * self.levels_enabled * (self.transform_mode == 2)

        imgui.pop_style_color()
        imgui.pop_style_color()

        imgui.text("")
        imgui.text("Crop from borders")

        push_bg_color(3, 2)
        imgui.slider_int("Top", self.crop_borders[3], 0, self.current_texture.size[1], '%d')
        imgui.pop_style_color()
        imgui.columns(2)
        push_bg_color(0, 2)
        imgui.slider_int("Left", self.crop_borders[0], 0, self.current_texture.size[0], '%d')
        imgui.pop_style_color()
        imgui.next_column()
        push_bg_color(2, 2)
        imgui.slider_int("Right", self.crop_borders[2], 0, self.current_texture.size[0], '%d')
        imgui.pop_style_color()
        imgui.columns(1)
        push_bg_color(1, 2)
        imgui.slider_int("Bottom", self.crop_borders[1], 0, self.current_texture.size[1], '%d')
        imgui.pop_style_color()
        imgui.text("")
        push_bg_color(4, 2)
        imgui.slider_float("Angle", -self.pic_angle, -360, 360, '%.2f')
        imgui.pop_style_color()

        previous_message_top_r = next_message_top_r
        next_message_top_r += imgui.get_window_height()
        imgui.end()

        imgui.set_next_window_position(pos_x, previous_message_top_r, 1, pivot_x=1, pivot_y=0.0)
        bg_color = 1 if self.transform_mode == 2 else 0.5
        imgui.push_style_color(imgui.COLOR_BORDER, bg_color, bg_color, bg_color, bg_color)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .5, .5, bg_color, bg_color)
        imgui.begin("Crop_label", True, SIDE_WND_FLAGS)
        imgui.text("Crop & rotate")
        imgui.set_window_font_scale(1.2)
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.end()

        imgui.set_next_window_position(pos_x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)
        bg_color = 1 if self.transform_mode == 3 else 0.5
        imgui.set_next_window_bg_alpha(.2 + .6 * bg_color)
        self.imgui_style.alpha = .8
        imgui.push_style_color(imgui.COLOR_BORDER, bg_color, bg_color, bg_color, bg_color)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .5, .5, bg_color, bg_color)

        imgui.begin("Navigation", True, SIDE_WND_FLAGS)
        imgui.text("Navigation")
        imgui.set_window_font_scale(1.2)
        next_message_top_r += imgui.get_window_height()
        imgui.pop_style_color()
        imgui.pop_style_color()
        imgui.end()

    def imgui_popdb(self):
        for item in self.pop_db:
            self.imgui_style.alpha = item['alpha'] * .8
            imgui.set_next_window_position(10, self.next_message_top)
            imgui.begin(str(item['type']), True, SIDE_WND_FLAGS)
            imgui.set_window_font_scale(1.2)
            imgui.text(item['text'])
            self.next_message_top += imgui.get_window_height() * max(item['alpha'] ** .3, item['alpha'])
            imgui.end()

    def imgui_tinder_stats(self, right=False):
        if self.image_count == 0:
            return
        self.imgui_style.alpha = .4
        left_signum = -1 if right else 1
        text_colors = [[1, .6, .6], [.8, .8, .8], [.6, .6, 1]]
        top = self.next_message_top
        last_window_height = 0
        next_x = 10 * left_signum % self.window_size.x
        info_text = [f"-: {self.tinder_stats[0]}", f"  {self.tinder_stats[1]}  ", f"+: {self.tinder_stats[2]}"]
        if self.switch_mode == SWITCH_MODE_COMPARE:
            right_next = right if self.mouse_move_cumulative > 0 else not right
            image_category = self.image_categories[self.image_index if right_next else self.previous_image_index]
        else:
            image_category = self.image_categories[self.image_index]
        whole_alpha = .75 - (self.split_line * .5 - .25) * left_signum

        for i in [0, 1, 2][:: left_signum]:
            self.imgui_style.alpha = (1 if image_category == i - 1 else .5) * whole_alpha
            imgui.set_next_window_position(next_x, top, 1, pivot_x=(.5 - left_signum / 2), pivot_y=0)
            imgui.begin(f"Selection statistics {i:d}" + ("right" if right else "left"), True, SIDE_WND_FLAGS)
            imgui.set_window_font_scale(1.5)
            imgui.text_colored(info_text[i], text_colors[i][0], text_colors[i][1], text_colors[i][2], 1)
            next_x += imgui.get_window_width() * left_signum
            last_window_height = imgui.get_window_height()
            imgui.end()

        if right:
            self.imgui_tinder_stats()
        else:
            self.next_message_top += last_window_height

    def imgui_compare(self):
        def show_side_info(left_sig=0):
            image_index = [self.previous_image_index, self.image_index][(int_cumulative + left_sig) % 2]
            self.imgui_style.alpha = .65 + (self.split_line * .7 - .35) * (2 * left_sig - 1)
            dir_index = self.file_to_dir[image_index]
            dirs_in_folder = self.dir_to_file[dir_index][1]
            index_in_folder = image_index + 1 - self.dir_to_file[dir_index][0]
            info_text = ["File name: " + os.path.basename(self.get_file_path(image_index)),
                         "Image # (current folder): " + f"{index_in_folder:d} of {dirs_in_folder:d}",
                         "Image # (all list): " + f"{image_index + 1:d} of {self.image_count:d}",
                         "Folder #: " + f"{dir_index + 1:d} of {self.dir_count:d}"]

            imgui.set_next_window_position(self.window_size.x * left_sig, self.window_size.y * .9,
                                           1, pivot_x=left_sig, pivot_y=0.5)

            imgui.begin(("Left", "Right")[left_sig] + " image", True, SIDE_WND_FLAGS)
            for text_ in info_text:
                imgui.text(text_)
            imgui.end()

        int_cumulative = int(self.mouse_move_cumulative < 0)
        show_side_info(0)
        show_side_info(1)


def main_loop() -> None:
    # mglw.setup_basic_logging(20)  # logging.INFO
    start_fullscreen = True
    if "-f" in sys.argv:
        start_fullscreen = not start_fullscreen

    enable_vsync = True
    # window = mglw.get_local_window_cls('pyglet')(fullscreen=start_fullscreen, vsync=enable_vsync)
    window = mglw.get_local_window_cls('pyglet')(vsync=enable_vsync)
    if start_fullscreen:
        window.mouse_exclusivity = True
        window.fullscreen = True

    window.print_context_info()
    mglw.activate_context(window=window)
    timer = mglw.timers.clock.Timer()
    window.config = ModernSlideShower(ctx=window.ctx, wnd=window, timer=timer)

    timer.start()
    timer.next_frame()
    timer.next_frame()
    window.config.post_init()

    while not window.is_closing:
        window.render()

    _, duration = timer.stop()
    window.destroy()
    if duration > 0:
        mglw.logger.info(
            "Duration: {0:.2f}s @ {1:.2f} FPS".format(
                duration, window.frames / duration
            )
        )


if __name__ == '__main__':
    main_loop()
