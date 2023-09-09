import pyglet
import moderngl_window


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


LEVEL_BORDER_NAMES = ['lvl_i_min', 'lvl_i_max', 'lvl_gamma',
                      'lvl_o_min', 'lvl_o_max', 'saturation']


class InterfaceMode:
    GENERAL = 50
    MENU = 51
    SETTINGS = 52
    LEVELS = 53
    TRANSFORM = 54
    MANDELBROT = 55


SWITCH_MODE_CIRCLES = 30
# SWITCH_MODE_GESTURES = 31
SWITCH_MODE_COMPARE = 32
SWITCH_MODE_TINDER = 33


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

    # InterfaceMode.GENERAL = 50
    # InterfaceMode.MENU = 51
    # InterfaceMode.SETTINGS = 52
    # InterfaceMode.LEVELS = 53
    # InterfaceMode.TRANSFORM = 54
    # InterfaceMode.MANDELBROT = 55

    KEYBOARD_MOVEMENT_ZOOM_IN_ON = 60
    KEYBOARD_MOVEMENT_ZOOM_OUT_ON = 61
    KEYBOARD_UP_PRESS = 62
    KEYBOARD_DOWN_PRESS = 63
    KEYBOARD_LEFT_PRESS = 64
    KEYBOARD_RIGHT_PRESS = 65
    KEYBOARD_MOVEMENT_LEFT_BRACKET_ON = 66
    KEYBOARD_MOVEMENT_RIGHT_BRACKET_ON = 67
    KEYBOARD_FLIPPING_FAST_NEXT_ON = 68
    KEYBOARD_FLIPPING_FAST_PREVIOUS_ON = 69

    KEYBOARD_MOVEMENT_ZOOM_IN_OFF = 70
    KEYBOARD_MOVEMENT_ZOOM_OUT_OFF = 71
    KEYBOARD_UP_RELEASE = 72
    KEYBOARD_DOWN_RELEASE = 73
    KEYBOARD_LEFT_RELEASE = 74
    KEYBOARD_RIGHT_RELEASE = 75
    KEYBOARD_MOVEMENT_LEFT_BRACKET_OFF = 76
    KEYBOARD_MOVEMENT_RIGHT_BRACKET_OFF = 77
    ACTION_SPACE_RELEASE = 78
    KEYBOARD_MOVEMENT_MOVE_ALL_RELEASE = 79

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
    ACTION_SPACE_PRESS = 94

    WINDOW_SWITCH_FULLSCREEN = 95
    WINDOW_GOTO_NEXT_SCREEN = 96

    CLOSE_PROGRAM = 100


random_image_actions = (
    Actions.IMAGE_RANDOM_FILE,
    Actions.IMAGE_RANDOM_DIR_RANDOM_FILE,
    Actions.IMAGE_RANDOM_DIR_FIRST_FILE,
    Actions.IMAGE_RANDOM_IN_CURRENT_DIR,
    Actions.IMAGE_RANDOM_UNSEEN_FILE
)

# MAIN_MENU structure:
# name, shortcut label, checkbox condition, action
MAIN_MENU = (
    ('Circle mode', '', lambda x: x.switch_mode == Actions.SWITCH_MODE_CIRCLES, None, Actions.SWITCH_MODE_CIRCLES),
    # ('Gesture mode', 'G', lambda x: x.switch_mode == SWITCH_MODE_GESTURES, None, Actions.SWITCH_MODE_GESTURES),
    ('Compare mode', 'X', lambda x: x.switch_mode == Actions.SWITCH_MODE_COMPARE, None, Actions.SWITCH_MODE_COMPARE),
    ('Tinder mode', 'J', lambda x: x.switch_mode == Actions.SWITCH_MODE_TINDER, None, Actions.SWITCH_MODE_TINDER),
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
    ('Adjust levels', 'L', None, None, InterfaceMode.LEVELS),
    ('Transform image', 'T', None, None, InterfaceMode.TRANSFORM),
    ('Settings', 'S', None, None, InterfaceMode.SETTINGS),
    ('Mandelbrot set', 'U', None, None, InterfaceMode.MANDELBROT),
    "--",
    ('Random image in the list', 'F5', None, None, Actions.IMAGE_RANDOM_FILE),
    ('Random image in current directory', 'F6', None, None, Actions.IMAGE_RANDOM_IN_CURRENT_DIR),
    ('First image in random directory', 'F7', None, None, Actions.IMAGE_RANDOM_DIR_FIRST_FILE),
    ('Random image in random directory', 'F8', None, None, Actions.IMAGE_RANDOM_DIR_RANDOM_FILE),
    "--",
    ('Keyboard shortcuts', 'F1, H', lambda x: x.central_message_showing, None, Actions.CENTRAL_MESSAGE_TOGGLE),
    ('Quit', 'Esc', None, None, Actions.CLOSE_PROGRAM),
)

rapid_menu_actions = (('Circle\n mode', Actions.SWITCH_MODE_CIRCLES),
                      ('Compare\n mode', Actions.SWITCH_MODE_COMPARE),
                      ('Tinder\n mode', Actions.SWITCH_MODE_TINDER),
                      ('Adjust\nlevels', InterfaceMode.LEVELS),
                      ('Zoom\nto fit', Actions.PIC_ZOOM_FIT),
                      ('Slideshow\n toggle', Actions.AUTO_FLIP_TOGGLE),
                      ('Rotate\n left', Actions.PIC_ROTATE_LEFT),
                      ('   Save \n rotation\nlosslessly', Actions.APPLY_ROTATION_90),
                      ('Rotate\nright', Actions.PIC_ROTATE_RIGHT)
                      )

# KEYBOARD_SHORTCUTS structure (dict):
# (action, interface_mode, ctrl, shift, alt, key): action_code
KEY = pyglet.window.key
prs = moderngl_window.context.base.BaseKeys.ACTION_PRESS
rls = moderngl_window.context.base.BaseKeys.ACTION_RELEASE

KEYBOARD_SHORTCUTS = {
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.S): InterfaceMode.SETTINGS,
    (prs, InterfaceMode.SETTINGS, False, False, False, KEY.S): InterfaceMode.GENERAL,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.T): InterfaceMode.TRANSFORM,
    (prs, InterfaceMode.TRANSFORM, False, False, False, KEY.T): InterfaceMode.GENERAL,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.L): InterfaceMode.LEVELS,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.L): InterfaceMode.GENERAL,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.U): InterfaceMode.MANDELBROT,

    # (prs, InterfaceMode.GENERAL, False, False, False, KEY.G): Actions.SWITCH_MODE_GESTURES,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.X): Actions.SWITCH_MODE_COMPARE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.J): Actions.SWITCH_MODE_TINDER,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.A): Actions.AUTO_FLIP_TOGGLE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.I): Actions.TOGGLE_IMAGE_INFO,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.C): Actions.FILE_COPY,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.M): Actions.FILE_MOVE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.NUM_0): Actions.FILE_MOVE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.BACKSLASH): Actions.FILE_MOVE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.COMMA): Actions.PIC_ROTATE_LEFT,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.PERIOD): Actions.PIC_ROTATE_RIGHT,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F3): Actions.LIST_SAVE_NO_COMPRESS,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F4): Actions.LIST_SAVE_WITH_COMPRESS,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F9): Actions.FILE_SAVE_WITH_SUFFIX,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F12): Actions.FILE_SAVE_AND_REPLACE,

    (prs, InterfaceMode.GENERAL, True, False, False, KEY.R): Actions.REVERT_IMAGE,

    (prs, InterfaceMode.GENERAL, False, False, False, KEY.ENTER): Actions.APPLY_ROTATION_90,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.PAGEUP): Actions.IMAGE_FOLDER_PREVIOUS,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.PAGEDOWN): Actions.IMAGE_FOLDER_NEXT,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F5): Actions.IMAGE_RANDOM_FILE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F6): Actions.IMAGE_RANDOM_IN_CURRENT_DIR,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F7): Actions.IMAGE_RANDOM_DIR_FIRST_FILE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F8): Actions.IMAGE_RANDOM_DIR_RANDOM_FILE,

    (prs, InterfaceMode.GENERAL, True, False, False, KEY.SPACE): Actions.KEYBOARD_FLIPPING_FAST_NEXT_ON,
    (prs, InterfaceMode.GENERAL, True, False, False, KEY.RIGHT): Actions.KEYBOARD_FLIPPING_FAST_NEXT_ON,
    (prs, InterfaceMode.GENERAL, True, False, False, KEY.LEFT): Actions.KEYBOARD_FLIPPING_FAST_PREVIOUS_ON,

    (prs, InterfaceMode.GENERAL, False, False, False, KEY.NUM_MULTIPLY): Actions.PIC_ZOOM_FIT,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.NUM_DIVIDE): Actions.PIC_ZOOM_100,

    (prs, InterfaceMode.GENERAL, False, False, False, KEY.RIGHT): Actions.KEYBOARD_RIGHT_PRESS,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.LEFT): Actions.KEYBOARD_LEFT_PRESS,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.UP): Actions.KEYBOARD_UP_PRESS,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.DOWN): Actions.KEYBOARD_DOWN_PRESS,

    (rls, InterfaceMode.GENERAL, False, False, False, KEY.RIGHT): Actions.KEYBOARD_RIGHT_RELEASE,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.LEFT): Actions.KEYBOARD_LEFT_RELEASE,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.UP): Actions.KEYBOARD_UP_RELEASE,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.DOWN): Actions.KEYBOARD_DOWN_RELEASE,

    (prs, InterfaceMode.GENERAL, False, False, False, KEY.SPACE): Actions.ACTION_SPACE_PRESS,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.SPACE): Actions.ACTION_SPACE_RELEASE,

    (prs, InterfaceMode.GENERAL, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,

    (rls, InterfaceMode.GENERAL, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,

    (prs, InterfaceMode.GENERAL, False, True, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, InterfaceMode.GENERAL, False, True, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, InterfaceMode.GENERAL, False, True, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,
    (prs, InterfaceMode.GENERAL, False, True, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,

    (rls, InterfaceMode.GENERAL, False, True, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, InterfaceMode.GENERAL, False, True, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, InterfaceMode.GENERAL, False, True, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, InterfaceMode.GENERAL, False, True, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,

    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON,

    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.NUM_ADD): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.EQUAL): Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF,
    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.NUM_SUBTRACT): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,
    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.MINUS): Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF,

    (prs, InterfaceMode.LEVELS, False, False, False, KEY.SEMICOLON): Actions.LEVELS_TOGGLE,
    (rls, InterfaceMode.LEVELS, False, False, False, KEY.SEMICOLON): Actions.LEVELS_TOGGLE,

    (prs, InterfaceMode.LEVELS, False, False, False, KEY.ENTER): Actions.LEVELS_APPLY,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.P): Actions.LEVELS_PREVIOUS,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.O): Actions.LEVELS_EMPTY,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.TAB): Actions.LEVELS_NEXT_BAND_ROUND,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.R): Actions.LEVELS_SELECT_RED,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.G): Actions.LEVELS_SELECT_GREEN,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.B): Actions.LEVELS_SELECT_BLUE,

    (prs, InterfaceMode.TRANSFORM, False, False, False, KEY.ENTER): Actions.APPLY_TRANSFORM,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F): Actions.WINDOW_SWITCH_FULLSCREEN,
    (prs, InterfaceMode.GENERAL, True, False, False, KEY.F): Actions.WINDOW_GOTO_NEXT_SCREEN,

    (rls, InterfaceMode.GENERAL, False, False, False, KEY.LSHIFT): Actions.KEYBOARD_MOVEMENT_MOVE_ALL_RELEASE,
    (rls, InterfaceMode.GENERAL, False, False, False, KEY.RSHIFT): Actions.KEYBOARD_MOVEMENT_MOVE_ALL_RELEASE,

    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.RIGHT): Actions.KEYBOARD_RIGHT_PRESS,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.LEFT): Actions.KEYBOARD_LEFT_PRESS,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.UP): Actions.KEYBOARD_UP_PRESS,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.DOWN): Actions.KEYBOARD_DOWN_PRESS,

    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.RIGHT): Actions.KEYBOARD_RIGHT_RELEASE,
    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.LEFT): Actions.KEYBOARD_LEFT_RELEASE,
    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.UP): Actions.KEYBOARD_UP_RELEASE,
    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.DOWN): Actions.KEYBOARD_DOWN_RELEASE,

    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.BRACKETLEFT): Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_ON,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.BRACKETRIGHT): Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_ON,
    (rls, InterfaceMode.MANDELBROT, False, False, False, KEY.BRACKETLEFT): Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_OFF,
    (
    rls, InterfaceMode.MANDELBROT, False, False, False, KEY.BRACKETRIGHT): Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_OFF,

    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.Z): Actions.MANDEL_GOOD_ZONES_TOGGLE,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.D): Actions.MANDEL_DEBUG_TOGGLE,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.T): Actions.MANDEL_GOTO_TEST_ZONE,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.A): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_NEAR,
    (prs, InterfaceMode.MANDELBROT, True, False, False, KEY.A): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_FAR,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.SPACE): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_NEAR,
    (prs, InterfaceMode.MANDELBROT, True, False, False, KEY.SPACE): Actions.MANDEL_TOGGLE_AUTO_TRAVEL_FAR,

    (prs, InterfaceMode.GENERAL, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.GENERAL, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.MENU, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.MENU, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.SETTINGS, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.SETTINGS, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.LEVELS, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.TRANSFORM, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.TRANSFORM, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.H): Actions.CENTRAL_MESSAGE_TOGGLE,
    (prs, InterfaceMode.MANDELBROT, False, False, False, KEY.F1): Actions.CENTRAL_MESSAGE_TOGGLE,
}

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
    "{count} files moved",  # 13
    "{count} files copied",  # 14
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

DUMMY_420_SHADER = """#version 420
                          #if defined PICTURE_VERTEX
                          void main() {}
                          #endif 
                          """
