import datetime
import shutil
import moderngl
import moderngl_window as mglw
import moderngl_window.context.base
import moderngl_window.context.pyglet.keys
import moderngl_window.meta
import argparse
import moderngl_window.timers.clock
from moderngl_window.opengl import program, vao
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import io
import rawpy
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import gaussian_filter
import numpy as np
import math
import os
# import sys
import random
import collections
import mpmath
import tomllib
import tomli_w
import multiprocessing
from mpmath import mp
from natsort import natsorted
from StaticData import *
import imgui
from enum import Enum

Point = collections.namedtuple('Point', ['x', 'y'])

ORIENTATION_DB = dict([
    (2, [Image.Transpose.FLIP_LEFT_RIGHT, Image.Transpose.FLIP_TOP_BOTTOM]),
    (3, [Image.Transpose.ROTATE_180, Image.Transpose.FLIP_TOP_BOTTOM]),
    (4, [Image.Transpose.FLIP_TOP_BOTTOM, Image.Transpose.FLIP_TOP_BOTTOM]),
    (5, [Image.Transpose.ROTATE_90]),
    (6, [Image.Transpose.ROTATE_270, Image.Transpose.FLIP_TOP_BOTTOM]),
    (7, [Image.Transpose.ROTATE_270]),
    (8, [Image.Transpose.ROTATE_90, Image.Transpose.FLIP_TOP_BOTTOM])
])


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
    out_real, out_imaginary = [], []
    current_value = in_value
    for _ in range(4):
        with mp.workprec(52):
            t = current_value
            c_hi = t - (t - current_value)
            out_real.append(float(c_hi.real))
            out_imaginary.append(float(c_hi.imag))
        current_value -= c_hi
    return out_real, out_imaginary


def split_complex_32(in_value):
    out_real, out_imaginary = [], []
    current_value = in_value
    for _ in range(4):
        with mp.workprec(26):
            t = current_value
            c_hi = t - (t - current_value)
            out_real.append(float(c_hi.real))
            out_imaginary.append(float(c_hi.imag))
        current_value -= c_hi
    return out_real, out_imaginary


def format_bytes_3(size):
    power = 2 ** 10  # 2**10 = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    f_size = size
    while f_size > power:
        f_size /= power
        n += 1
    digits = 3 - math.ceil(math.log10(f_size)) if f_size < 1000 else 0
    formatted_string = f'{f_size:.{digits:d}f} {power_labels[n]}  ({size:,d} B)'
    formatted_string = formatted_string.replace(",", " ")
    return formatted_string


def reorient_image(im):
    image_orientation = 0
    try:
        im_exif = im.getexif()
        image_orientation = im_exif.get(274, 0)
    except (KeyError, AttributeError, TypeError, IndexError):
        # print(KeyError)
        pass

    set_of_operations = ORIENTATION_DB.get(image_orientation, [Image.Transpose.FLIP_TOP_BOTTOM])
    for operation in set_of_operations:
        im = im.transpose(operation)
    return im


# def load_settings():
#     if os.path.isfile("settings.json"):
#         with open("settings.json", 'r') as f:
#             return json.load(f)


def load_image(path_target, do_thumb=False):
    def draw_dummy():
        im_object = Image.new("RGB", (256, 256), (20, 50, 80))
        d = ImageDraw.Draw(im_object)
        d.ellipse((10, 10, 200, 200))
        d.ellipse((100, 100, 255, 255))
        return im_object.tobytes()

    try:
        if not do_thumb and any((path_target.lower().endswith(ex) for ex in RAW_FILE_TYPES)):
            with open(path_target, 'rb', buffering=0) as f:  # This is a workaround for opening non-latin file names
                thumb = rawpy.imread(f).extract_thumb()
            path_target = io.BytesIO(thumb.data)

        with Image.open(path_target) as im_object:
            im_exif = im_object.info.get("exif", b'')

            if im_object.mode == "RGB":
                im_object = reorient_image(im_object)
            else:
                im_object = reorient_image(im_object).convert(mode="RGB")
            if do_thumb:
                im_object.thumbnail((256, 256))
                im_object = ImageOps.pad(im_object, (256, 256))
            return im_object.tobytes(), Point(im_object.width, im_object.height), im_exif
    except Exception as e:
        if do_thumb:
            return draw_dummy(), Point(0, 0), b''
        else:
            return None, e, b''


def image_loader_process(in_queue: multiprocessing.Queue, out_queue: multiprocessing.Queue, do_thumb):
    out_queue.put("Started")
    while True:
        task_data = in_queue.get()
        if task_data is None:
            print("Exit loader")
            return
        pic_data = load_image(task_data[1], do_thumb=do_thumb)
        try:
            out_queue.put((task_data, *pic_data), timeout=2)
        except Exception:
            pass


def init_image_loader(do_thumb=False):
    thumb_queue_tasks = multiprocessing.Queue()
    thumb_queue_data = multiprocessing.Queue()
    thumb_queue_tasks.cancel_join_thread()
    thumb_queue_data.cancel_join_thread()
    thumb_loader = multiprocessing.Process(target=image_loader_process,
                                           args=(thumb_queue_tasks, thumb_queue_data, do_thumb))
    thumb_loader.start()
    thumb_queue_data.get()
    return thumb_queue_tasks, thumb_queue_data, thumb_loader


class Configs(Enum):
    HIDE_BORDERS = "hide_borders"
    TRANSITION_DURATION = "transition_duration"
    INTER_BLUR = "transition_blur"
    STARTING_ZOOM_FACTOR = "initial_zoom"
    PIXEL_SIZE = "pixel_squareness"
    FULL_SCREEN_ID = "full_screen_monitor_id"

    Values = (HIDE_BORDERS,
              TRANSITION_DURATION,
              INTER_BLUR,
              STARTING_ZOOM_FACTOR,
              PIXEL_SIZE,
              FULL_SCREEN_ID,
              )

    DESCRIPTIONS = {
        HIDE_BORDERS: "Hide image borders",
        TRANSITION_DURATION: "Duration of transition between images",
        INTER_BLUR: "Blur during transition",
        STARTING_ZOOM_FACTOR: "Zoom of newly shown image",
        PIXEL_SIZE: "Pixel shape in case of extreme zoom",
        FULL_SCREEN_ID: "ID of the full screen monitor",
    }

    FORMATS = {
        HIDE_BORDERS: (0, 1000, '%.1f', 0),
        TRANSITION_DURATION: (0.01, 10, '%.3f', 0),
        INTER_BLUR: (0, 1000, '%.1f', 0),
        STARTING_ZOOM_FACTOR: (0, 5, '%.3f', 0),
        PIXEL_SIZE: (0, 100, '%.2f', 32),
        FULL_SCREEN_ID: (0, 3, '%.0f', 0),
    }


class Config:
    def __init__(self, config_file='settings.toml'):
        self.config_file = config_file
        self.defaults = {
            Configs.HIDE_BORDERS.value: 100,
            Configs.TRANSITION_DURATION.value: 0.75,
            Configs.INTER_BLUR.value: 30.0,
            Configs.STARTING_ZOOM_FACTOR.value: 0.98,
            Configs.PIXEL_SIZE.value: 0,
            Configs.FULL_SCREEN_ID.value: 1,
        }
        self.settings = self.defaults.copy()
        self.load_settings()

    def load_settings(self):
        if os.path.isfile(self.config_file):
            with open(self.config_file, 'rb') as f:
                loaded_settings = tomllib.load(f)
                for key, value in loaded_settings.items():
                    if key in self.defaults:
                        self.settings[key] = self.validate_setting(key, value)

    def save_settings(self):
        with open(self.config_file, 'wb') as f:
            tomli_w.dump(self.settings, f)

    def validate_setting(self, key, value):
        min_val, max_val, _, _ = Configs.FORMATS.value[key]
        if isinstance(value, (int, float)):
            if value < min_val:
                return min_val
            elif value > max_val:
                return max_val
            else:
                return value
        else:
            return self.defaults[key]

    def get(self, key):
        return self.settings.get(key.value, self.defaults.get(key.value))

    def set(self, key, value):
        if key.value in self.defaults:
            self.settings[key.value] = self.validate_setting(key.value, value)


class ModernSlideShower(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "ModernSlideShower"
    aspect_ratio = None
    clear_color = (0.0, 0.0, 0.0, 0.0)
    wnd = mglw.context.base.BaseWindow
    screens = []
    use_screen_id = 0

    start_with_random_image = False
    average_frame_time = 0
    last_image_load_time = 0
    previous_image_duration = 0
    init_end_time = 0

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
    dir_list = []
    file_list = []
    dir_to_file = []
    file_to_dir = []
    image_ratings = []
    image_categories = []
    tinder_stats_d = {-1: 0, 0: 0, 1: 0}
    tinder_last_choice = 0
    small_zoom = False
    move_sensitivity = 0.
    pic_square = 1
    thumbs_list = {}
    thumb_id = 0
    thumb_spacing = 5  # how many percent of space between thumbnails
    thumb_period = 1 + thumb_spacing / 100
    thumb_loader = multiprocessing.Process
    image_loader = multiprocessing.Process
    thumb_queue_tasks = multiprocessing.SimpleQueue
    thumb_queue_data = multiprocessing.SimpleQueue
    image_queue_tasks = multiprocessing.SimpleQueue
    image_queue_data = multiprocessing.SimpleQueue
    thumbs_for_image_requested = -1
    preloads_for_image_requested = -1
    image_cache = {}

    # random_folder_mode = False
    # exclude_plus_minus = False

    seen_images = np.zeros(0)
    current_image_is_unseen = True
    all_images_seen_times = -1

    imgui_io = imgui.get_io
    imgui_style = imgui.get_style

    image_original_size = Point(0, 0)
    im_exif = b''

    common_path = ""
    parent_path = ""  # all path lower than parent is considered as subfolders.

    pic_pos_current = mp.mpc()
    pic_pos_future = mp.mpc()
    pic_move_speed = mp.mpc()
    pic_pos_fading = mp.mpc()

    pic_zoom = .5
    pic_zoom_future = .2
    pic_zoom_null = .2
    pic_angle = 0.
    pic_angle_future = 0.
    big_zoom = False
    gl_program_pic = [moderngl.Program] * 2
    gl_program_borders = moderngl.Program
    gl_program_round = moderngl.Program
    gl_program_mandel = [moderngl.Program] * 3
    gl_program_crop = moderngl.Program
    gl_program_browse_squares = moderngl.Program
    gl_program_browse_pic = moderngl.Program
    gl_program_compare = moderngl.Program
    use_old_gl = False
    mandel_id = 0
    program_id = 0
    image_texture = moderngl.Texture
    thumb_textures = moderngl.TextureArray
    THUMBS_ROW_MAX = 7
    thumb_rows = 0
    thumb_row_elements = 0
    thumbs_backward_max = 100
    thumbs_forward_max = 200
    thumbs_count = thumbs_forward_max + thumbs_backward_max
    thumbs_backward = 0
    thumbs_forward = 1
    thumbs_shown = 1
    thumbs_displacement = mp.mpc()
    thumb_central_index = 0
    current_texture = moderngl.Texture
    current_texture_old = moderngl.Texture
    histo_texture = moderngl.Texture
    histo_texture_empty = moderngl.Buffer
    mandel_stat_empty = moderngl.Buffer
    histogram_array = np.empty

    max_keyboard_flip_speed = .3
    mouse_move_atangent = 0.
    mouse_move_atangent_delta = 0.
    mouse_move_cumulative = 0.
    mouse_unflipping_speed = 1.
    last_image_folder = None

    transition_center = (.4, .4)

    reset_frame_timer = True

    interface_mode = InterfaceMode.GENERAL
    switch_mode = SWITCH_MODE_CIRCLES

    # scan_all_files = False
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

    # configs = {
    #     Configs.HIDE_BORDERS: 100,
    #     Configs.TRANSITION_DURATION: .75,
    #     Configs.INTER_BLUR: 30.,
    #     Configs.STARTING_ZOOM_FACTOR: .98,
    #     Configs.PIXEL_SIZE: 0
    # }

    # config_descriptions = {
    #     Configs.HIDE_BORDERS: "Hide image borders",
    #     Configs.TRANSITION_DURATION: "Duration of transition between images",
    #     Configs.INTER_BLUR: "Blur during transition",
    #     Configs.STARTING_ZOOM_FACTOR: "Zoom of newly shown image",
    #     Configs.PIXEL_SIZE: "Pixel shape in case of extreme zoom",
    # }

    # config_formats = {
    #     Configs.HIDE_BORDERS: (0, 1000, '%.1f', 2),
    #     Configs.TRANSITION_DURATION: (0.01, 10, '%.3f', 2),
    #     Configs.INTER_BLUR: (0, 1000, '%.1f', 4),
    #     Configs.STARTING_ZOOM_FACTOR: (0, 5, '%.3f', 4),
    #     Configs.PIXEL_SIZE: (0, 100, '%.1f', 4),
    # }

    last_key_press_time = 0
    setting_active = 0
    autoflip_speed = 0.

    run_reduce_flipping_speed = 0.
    run_key_flipping = 0
    key_flipping_next_time = 0.
    run_flip_once = 0
    pressed_mouse = 0

    pop_db = []

    # levels_borders = [[0.] * 4, [1.] * 4, [1.] * 4, [0.] * 4, [1.] * 4, [1.] * 4]
    levels_borders = [[1. if i % 3 else 0.] * 4 for i in range(6)]
    levels_borders_previous = []

    levels_enabled = True
    levels_edit_band = 3
    levels_edit_parameter = 0
    levels_edit_group = 0

    gesture_mode_timeout = 0

    key_picture_movement = [False] * 8

    transform_mode = 2
    crop_borders_active = 0
    pic_screen_borders = np.array([0.] * 4)
    crop_borders = np.array([0.] * 4)
    resize_xy = 1
    resize_x = 1
    resize_y = 1

    transition_stage = 1.

    mouse_buffer = np.array([0., 0.])

    show_image_info = 0
    show_rapid_menu = False
    selected_rapid_action = -1
    current_image_file_size = 0

    central_message_showing = False
    # parser = argparse.ArgumentParser(description="ModernSlideShower", allow_abbrev=False)
    # program_args = dict
    entered_digits = ""
    digit_flop_time = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.init_program()

    def init_program(self):
        self.ret_vertex_buffer = self.ctx.buffer(reserve=16)
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

        # parser.add_argument("--folder_name", help="Specify the folder name")
        # self.program_args, unknown_args = parser.parse_known_args()

        def sub_program(program_name: str):
            return mglw.opengl.program.ShaderSource(program_name.upper(), program_name.lower(),
                                                    picture_program_text).source

        def get_program_text(filename, program_text=""):
            if os.path.isfile(filename):
                with open(filename, 'r') as fd:
                    program_text = fd.read()
            return program_text

        if program_args.old_gl:
            self.use_old_gl = True
        else:
            try:
                # test fot newer version compatibility
                picture_program_text = DUMMY_420_SHADER
                self.ctx.program(vertex_shader=sub_program("picture_vertex"))
            except moderngl.error.Error:
                self.use_old_gl = True

        prog_suffix = "_simple" if self.use_old_gl else ""
        picture_program_text = get_program_text("picture" + prog_suffix + ".glsl")
        mandel_program_text = get_program_text("mandelbrot" + prog_suffix + ".glsl")

        def compile_program(vertex_name, geometry_name, fragment_name):
            if geometry_name:
                geometry_name = sub_program(geometry_name)
            return self.ctx.program(vertex_shader=sub_program(vertex_name),
                                    geometry_shader=geometry_name,
                                    fragment_shader=sub_program(fragment_name))

        self.gl_program_pic = []

        for _ in (0, 0):
            self.gl_program_pic.append(compile_program("picture_vertex", "picture_geometry", "picture_fragment"))
            self.gl_program_pic[-1]['texture_image'] = 5

        self.gl_program_crop = compile_program("picture_vertex", "crop_geometry", "crop_fragment")
        self.gl_program_browse_squares = compile_program("picture_vertex", "browse_geometry", "browse_fragment")
        self.gl_program_browse_pic = compile_program("picture_vertex", "browse_pic_geometry", "browse_pic_fragment")
        self.gl_program_compare = compile_program("compare_vertex", "compare_geometry", "compare_fragment")
        self.gl_program_round = compile_program("round_vertex", None, "round_fragment")
        self.gl_program_borders = self.ctx.program(vertex_shader=sub_program("picture_vertex"),
                                                   varyings=['crop_borders'])
        self.gl_program_browse_pic['texture_thumb'] = 8
        self.gl_program_borders['texture_image'] = 5

        p_d = moderngl_window.meta.ProgramDescription
        program_single = mglw.opengl.program.ProgramShaders.from_single

        self.gl_program_mandel = []
        self.gl_program_mandel.append(program_single(p_d(defines={"definition": 0}), mandel_program_text).create())
        self.gl_program_mandel.append(program_single(p_d(defines={"definition": 1}), mandel_program_text).create())
        self.gl_program_mandel.append(program_single(p_d(defines={"definition": 2}), mandel_program_text).create())

        self.mandel_stat_buffer = self.ctx.texture((32, 64), 1, dtype='u4')
        self.mandel_stat_buffer.bind_to_image(4, read=True, write=True)
        self.mandel_stat_empty = self.ctx.buffer(reserve=(32 * 64 * 4))
        self.histo_texture = self.ctx.texture((256, 5), 1, dtype='u4')
        self.histo_texture.bind_to_image(7, read=True, write=True)
        self.histo_texture_empty = self.ctx.buffer(reserve=(256 * 5 * 4))
        self.histogram_array = np.zeros((5, 256), dtype=np.float32)
        self.generate_round_geometry()
        self.empty_level_borders()
        self.empty_level_borders()
        self.find_jpegtran()
        # self.load_settings()
        # loaded_settings = load_settings()
        # if loaded_settings:
        #     self.configs = load_settings()

    def post_init(self):
        self.window_size = self.wnd.size
        if program_args.random_image:
            self.start_with_random_image = Actions.IMAGE_RANDOM_FILE
        if program_args.random_dir:
            self.start_with_random_image = Actions.IMAGE_RANDOM_DIR_FIRST_FILE
        # if "-r" in sys.argv or "-F5" in sys.argv:
        #     self.start_with_random_image = Actions.IMAGE_RANDOM_FILE
        # if "-F6" in sys.argv:
        #     self.start_with_random_image = Actions.IMAGE_RANDOM_IN_CURRENT_DIR
        # if "-F7" in sys.argv:
        #     self.start_with_random_image = Actions.IMAGE_RANDOM_DIR_FIRST_FILE
        # if "-F8" in sys.argv:
        #     self.start_with_random_image = Actions.IMAGE_RANDOM_DIR_RANDOM_FILE
        self.get_images()

        if self.wnd.is_closing:
            return

        if self.image_count == 0:
            self.central_message_showing = 2
            self.switch_interface_mode(InterfaceMode.MANDELBROT)
            self.init_end_time = self.timer.time
            return

        self.image_categories = np.zeros(self.image_count, dtype=int)
        self.seen_images = np.zeros(self.image_count, dtype=bool)
        self.tinder_stats_d[0] = self.image_count
        self.find_common_path()
        if self.start_with_random_image:
            if self.start_with_random_image is True:
                self.start_with_random_image = Actions.IMAGE_RANDOM_FILE
            self.random_image(self.start_with_random_image)
        else:
            self.load_image()
            self.unschedule_pop_message(7)
            self.unschedule_pop_message(8)
        self.transition_stage = 1
        if program_args.tinder_mode:
            self.switch_swithing_mode(SWITCH_MODE_TINDER)

        # --- Temp part. Fill thumbs base
        new_image = Image.new("RGB", (256, 256), (25, 50, 80))

        d = ImageDraw.Draw(new_image)
        d.ellipse((10, 10, 200, 200))
        d.ellipse((100, 100, 255, 255))

        thumb_texture = self.ctx.texture_array((256, 256, self.thumbs_count), 3)
        for i in range(self.thumbs_count):
            d.ellipse(tuple(np.random.randint(0, 128, 2).tolist() + np.random.randint(128, 256, 2).tolist()))
            thumb_texture.write(new_image.tobytes(), (0, 0, i, 256, 256, 1))
        self.thumb_textures = thumb_texture
        self.gl_program_browse_pic['thumbs_count'] = self.thumbs_count
        self.gl_program_browse_squares['thumb_period'] = self.thumb_period
        self.gl_program_browse_pic['thumb_period'] = self.thumb_period
        self.thumbs_list = [''] * self.thumbs_count
        self.init_end_time = self.timer.time
        self.thumb_textures.use(location=8)

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
        if program_args.path:
            for argument in program_args.path:
                if os.path.isdir(argument.rstrip('\\"')):
                    dir_arguments.append(os.path.abspath(argument.rstrip('\\"')))
                if os.path.isfile(argument):
                    file_arguments.append(os.path.abspath(argument))

            if len(dir_arguments):
                [self.scan_directory(directory) for directory in dir_arguments]
            if len(file_arguments):
                if len(dir_arguments) == 0 and len(file_arguments) == 1:
                    if file_arguments[0].lower().endswith(ALL_FILE_TYPES) or program_args.ignore_extention:
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
            self.common_path = os.path.commonpath(self.dir_list[::len(self.dir_list) // 1000] + self.dir_list[-3:])
        else:
            self.common_path = os.path.commonpath(self.dir_list)
        parent_path = self.dir_list[0]
        if self.common_path == parent_path:
            self.common_path = os.path.dirname(self.common_path)
        if program_args.base_folder_name:
            search_folder = os.path.sep + program_args.base_folder_name + os.path.sep
            found_match = self.common_path.find(search_folder)
            if found_match >= 0:
                self.parent_path = self.common_path[:found_match + len(search_folder)]
        if not self.parent_path:
            self.parent_path = os.path.dirname(self.common_path)

        print(f"Common path: {self.common_path}")
        print(f"Parent path: {self.parent_path}")

    def scan_directory(self, dirname, look_for_file=None):
        print("Searching for images in", dirname)

        def adjust_step():
            if self.image_count > 15000:
                return 1000
            elif self.image_count > 5000:
                return 500
            return 100

        report_step = adjust_step()

        for root, dirs, files in os.walk(dirname):
            file_count = 0
            this_dir_file_list = []
            first_file = self.image_count
            if program_args.exclude_sorted:
                if "\\++" in root or "\\--" in root:
                    continue
            for f in files:
                if self.wnd.is_closing:
                    return
                if f.lower().endswith(ALL_FILE_TYPES) or program_args.ignore_extention:
                    img_path = os.path.join(root, f)
                    self.image_count += 1
                    file_count += 1
                    this_dir_file_list.append(f)
                    self.file_to_dir.append(self.dir_count)
                    if not self.image_count % report_step:
                        print(self.image_count, "images found", end="\r")
                        self.render()
                        report_step = adjust_step()
                    if look_for_file:
                        if img_path == look_for_file:
                            self.new_image_index = self.image_count - 1
            if file_count:
                self.dir_list.append(root)
                self.dir_to_file.append([first_file, file_count])
                self.dir_count += 1
                self.file_list += natsorted(this_dir_file_list)

    def scan_file(self, filename):
        if filename.lower().endswith(ALL_FILE_TYPES) or program_args.ignore_extention:
            file_dir = os.path.dirname(filename)
            if self.dir_list and file_dir == self.dir_list[-1]:
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
            if "_r." in os.path.basename(filename):
                self.start_with_random_image = self.start_with_random_image or True
        elif filename.lower().endswith(PLAIN_LIST_FILE_TYPE):
            self.load_plain_list_file(filename)
            if "_r." in os.path.basename(filename):
                self.start_with_random_image = self.start_with_random_image or True

    def load_plain_list_file(self, filename):
        print("Opening plain list", filename)
        with open(filename, 'r', encoding='utf-8') as file_handle:
            if program_args.skip_load:
                def skip_lines(lines):
                    for _ in range(lines):
                        next(file_handle, False)
                    return next(file_handle, False)
                loaded_list = []
                last_line = 0
                skip_step = 2 ** program_args.skip_load
                sp = (-skip_step // 50 - 1, skip_step // 30 + 1, skip_step // 2 + 1)
                skip_lines(random.randint(0, skip_step))
                while last_line := skip_lines(skip_step if last_line else 0):
                    loaded_list.append(last_line.rstrip())
                    skip_step = max(skip_step + random.randint(*sp[:2]), sp[2])

            else:
                loaded_list = [line.rstrip() for line in file_handle.readlines()]

        last_dir_index = 0
        for line in loaded_list:
            if not line.lower().endswith(ALL_FILE_TYPES):
                continue

            line_split = line.split(":::")
            image_rating = line_split[0] if len(line_split) > 1 else 0

            file_dir = os.path.dirname(line_split[-1])
            self.file_list.append(os.path.basename(line_split[-1]))

            if self.dir_list and file_dir == self.dir_list[last_dir_index]:
                self.dir_to_file[last_dir_index][1] += 1
            elif file_dir in self.dir_list:
                last_dir_index = self.dir_list.index(file_dir)
                self.dir_to_file[last_dir_index][1] += 1
            else:
                last_dir_index = self.dir_count
                self.dir_list.append(file_dir)
                self.dir_to_file.append([self.image_count, 1])
                self.dir_count += 1

            self.file_to_dir.append(last_dir_index)
            self.image_ratings.append(image_rating)
            self.image_count += 1

    def load_list_file(self, filename):
        print("Opening list", filename)
        with open(filename, 'r', encoding='utf-8') as file_handle:
            loaded_list = [line.rstrip() for line in file_handle.readlines()]
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

    def save_list_file(self, compress=True):
        if not os.path.isdir(SAVE_FOLDER):
            try:
                os.makedirs(SAVE_FOLDER)
            except Exception as e:
                print("Could not create folder ", e)

        new_list_file_name = os.path.basename(self.common_path) + '_' + \
                             datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S") + '.' + LIST_FILE_TYPE
        new_list_file_name = os.path.join(os.path.dirname(self.common_path), new_list_file_name)

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
            index = min(self.new_image_index, self.image_count - 1)
        if self.image_count == 0:
            dir_index = - 1
        else:
            dir_index = self.file_to_dir[index]
        if dir_index == -1:
            return EMPTY_IMAGE_LIST
        dir_name = self.dir_list[dir_index]
        file_name = self.file_list[index]
        img_path = os.path.join(dir_name, file_name)
        if os.path.isfile(img_path):
            return img_path
        else:
            return EMPTY_IMAGE_LIST

    def release_texture(self, texture):
        if self.image_texture == texture:
            return
        if type(texture) is moderngl.Texture:
            try:
                texture.release()
            except Exception:
                pass

    def prepare_to_mandelbrot(self):
        self.image_original_size = Point(self.wnd.width, self.wnd.height)
        self.show_image_info = 1
        self.wnd.title = "ModernSlideShower: Mandelbrot mode"
        self.pic_angle_future = -30
        self.mandel_auto_complexity = 2
        self.pic_zoom = 1e-3
        self.pic_zoom_future = .2
        self.discrete_actions(Actions.SWITCH_MODE_CIRCLES)
        self.unschedule_pop_message(21)

    def load_next_existing_image(self):
        start_number = self.new_image_index
        half_count = self.image_count // 2
        image_jump = (self.new_image_index - self.image_index + half_count) % self.image_count - half_count
        increment = -1 if image_jump < 0 else 1
        while True:
            self.new_image_index = (self.new_image_index + increment) % self.image_count
            if start_number == self.new_image_index:
                self.file_list[self.new_image_index] = EMPTY_IMAGE_LIST
                self.central_message_showing = 3
                self.switch_interface_mode(InterfaceMode.MANDELBROT)
                break
            if os.path.exists(self.get_file_path()):
                if not self.load_image(subloader=True):
                    break

    def load_image(self, subloader=False):
        def load_failed():
            if subloader:
                return 1
            else:
                self.load_next_existing_image()
                return

        self.previous_image_duration = self.timer.time - self.last_image_load_time + .01
        image_path = self.get_file_path()
        self.mouse_unflipping_speed = 1

        if not (image_load_result := self.image_cache.get(image_path, [])):
            image_load_result = load_image(image_path)
        if image_load_result[0] is None:
            if image_path != EMPTY_IMAGE_LIST:
                print("Error loading ", image_path, image_load_result[1])
            return load_failed()
        image_bytes, self.image_original_size, self.im_exif = image_load_result
        self.current_image_file_size = os.stat(image_path).st_size

        self.wnd.title = "ModernSlideShower: " + image_path
        if self.image_texture != self.current_texture:
            self.release_texture(self.image_texture)
        self.current_texture_old = self.current_texture

        self.image_texture = self.ctx.texture(self.image_original_size, 3, image_bytes)
        self.image_texture.repeat_x = False
        self.image_texture.repeat_y = False
        self.image_texture.build_mipmaps()
        # self.image_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.current_texture = self.image_texture
        self.previous_image_index = self.image_index
        self.image_index = self.new_image_index
        self.recalc_thumbs()
        # self.dir_index = self.file_to_dir[self.image_index]
        # self.images_in_folder = self.dir_to_file[self.dir_index][1]
        # self.index_in_folder = self.image_index + 1 - self.dir_to_file[self.dir_index][0]

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

    def recalc_thumbs(self):
        self.thumb_id = self.image_index % self.thumbs_count
        self.thumbs_backward = min(self.image_index, self.thumbs_count - min(self.image_count - self.image_index,
                                                                             self.thumbs_forward_max))
        self.thumbs_forward = min(self.image_count - self.image_index,
                                  self.thumbs_count - min(self.image_index, self.thumbs_backward_max))

    def request_thumbs(self):
        for thumb_number in range(self.thumbs_count * 10):
            thumb_offset = [1, -1][thumb_number % 2] * thumb_number // 2
            if thumb_offset < -self.thumbs_backward or thumb_offset > self.thumbs_forward - 1:
                continue
            current_thumb_id = (self.thumb_id + thumb_offset) % self.thumbs_count
            thumb_list_id = (self.image_index + thumb_offset) % self.image_count
            thumb_path_target = self.get_file_path(thumb_list_id)
            thumb_path_current = self.thumbs_list[current_thumb_id]
            if thumb_path_target != thumb_path_current:
                self.thumb_queue_tasks.put((current_thumb_id, thumb_path_target))
                self.thumbs_list[current_thumb_id] = thumb_path_target
                return
        self.thumbs_for_image_requested = self.image_index

    def get_loaded_thumbs(self):
        thumb_data = self.thumb_queue_data.get()
        self.thumb_textures.write(thumb_data[1], (0, 0, thumb_data[0][0], 256, 256, 1))

    def update_preload_cache(self):
        if self.image_queue_data.empty():
            if self.preloads_for_image_requested == self.image_index: return
            for offset in range(5):
                request_id = (self.image_index + offset) % self.image_count
                request_path = self.get_file_path(request_id)
                if request_path in self.image_cache: continue
                self.image_queue_tasks.put((request_id, request_path))
                self.image_cache[request_path] = []
                return
            self.preloads_for_image_requested = self.image_index
            return
        else:
            image_data = self.image_queue_data.get()
            self.image_cache[image_data[0][1]] = image_data[1:]
            if len(self.image_cache) > 10:
                self.image_cache.pop(next(iter(self.image_cache)))

    def check_folder_change(self):
        current_folder = self.file_to_dir[self.image_index]

        if self.image_index == 0 and not self.interface_mode == InterfaceMode.MANDELBROT:
            self.unschedule_pop_message(8)
        elif current_folder != self.last_image_folder and not len(self.image_ratings):
            self.schedule_pop_message(12, 5, dir_index=current_folder + 1, dir_count=self.dir_count)
            self.pic_pos_current += self.current_texture.width / 15
        else:
            self.unschedule_pop_message(7)

        self.last_image_folder = current_folder

    def file_copy_move_routine(self, do_copy=False):
        mouse_cumulative = self.mouse_move_cumulative
        split_line = self.split_line
        im_index_current = self.image_index
        im_index_previous = self.previous_image_index
        file_operations = 0

        for score, prefix in ((-1, "--"), (1, "++")):
            indices = np.asarray(self.image_categories == score).nonzero()
            for i in indices[0][::-1]:
                file_operations += self.file_operation(i, prefix, do_copy)

        self.image_categories = np.zeros(self.image_count, dtype=int)
        self.tinder_stats_d[-1] = 0
        self.tinder_stats_d[1] = 0

        if self.switch_mode != SWITCH_MODE_COMPARE:
            self.mouse_move_cumulative = self.mouse_move_cumulative * .05

        if do_copy:
            self.schedule_pop_message(14, count=file_operations, duration=10)
        else:
            if self.image_count:
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
                self.schedule_pop_message(13, count=file_operations, duration=10)
            else:
                self.new_image_index = 0
                self.pic_zoom_future = 1e-15
                self.schedule_pop_message(24, duration=8, show_time=8)
                self.central_message_showing = 3

    def file_operation(self, im_index, prefix, do_copy=False):
        full_name = self.get_file_path(im_index)
        parent_folder = os.path.dirname(full_name)
        # print(full_name, prefix)
        own_subfolder = parent_folder[len(self.parent_path):]
        if not own_subfolder.startswith("\\"):
            own_subfolder = "\\" + own_subfolder
        new_folder = os.path.join(self.parent_path, prefix) + own_subfolder
        # print(new_folder)
        if not os.path.isdir(new_folder):
            try:
                os.makedirs(new_folder)
            except Exception as e:
                print("Could not create folder", e)

        try:
            if do_copy:
                shutil.copy(full_name, new_folder)
            else:
                shutil.move(full_name, new_folder)
                if not os.listdir(parent_folder):
                    os.rmdir(parent_folder)
            return 1
        except Exception as e:
            # todo good message here
            print("Could not complete file " + ["move ", "copy "][do_copy], e)
            return 0

    def lossless_save_possible(self):
        return self.pic_angle_future % 360 and self.jpegtran_exe and not (self.pic_angle_future % 90)

    def save_rotation_90(self):
        if self.lossless_save_possible():
            rotate_command = self.jpegtran_exe + JPEGTRAN_OPTIONS.format(round(360 - self.pic_angle_future % 360),
                                                                         self.get_file_path(self.image_index))
            os.system(rotate_command)
            self.image_cache.pop(self.get_file_path(self.image_index), None)

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
        self.pic_pos_future += self.mandel_move_acceleration * frame_time_chunk * self.mandel_auto_travel_speed

    def mandel_adjust_complexity(self, frame_time_chunk):
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

    def move_image(self, dx=0, dy=0, accelerate=.3):
        self.move_sensitivity = mix(self.move_sensitivity, 1, accelerate)
        self.pic_move_speed += mp.mpc(dx * self.move_sensitivity, dy * self.move_sensitivity) / self.pic_zoom
        self.pic_zoom_future = mix(self.pic_zoom_future, self.pic_zoom, .1)

    def compute_transition(self):
        if self.transition_stage < 0:
            self.transition_stage = 0
            return

        transition_time = 1 / min(self.previous_image_duration * .7, config.get(Configs.TRANSITION_DURATION))
        transition_step = (1.2 - self.transition_stage) * transition_time * self.last_frame_duration
        to_target_stage = abs(self.mouse_move_cumulative) / 100 - self.transition_stage
        if to_target_stage > 0:
            self.transition_stage += to_target_stage * self.last_frame_duration * 10
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
        if self.interface_mode == InterfaceMode.GENERAL and self.small_zoom and self.image_count:
            optimal_row_elements = int(self.window_size.x / (self.pic_square * self.thumb_period) / 2)
            optimal_row_elements = restrict(optimal_row_elements, 1, self.thumb_row_elements) * 2 + 1
            zoom_correction = self.window_size.x / (optimal_row_elements * self.pic_square * self.thumb_period)
            self.pic_zoom_future = mix(self.pic_zoom_future, self.pic_zoom_future * zoom_correction, .015)
        self.pic_zoom = mix(self.pic_zoom, self.pic_zoom_future, rate)
        self.pic_angle = mix(self.pic_angle, self.pic_angle_future, 5 * chunk)

    def check_image_vilible(self):
        correction_vector = mp.mpc()
        if self.interface_mode == InterfaceMode.MANDELBROT:
            border = 1000 * 1.4
            x_re = self.pic_pos_current.real - 800
            if abs(x_re) > border:
                correction_vector += math.copysign(border, x_re) - x_re

            x_im = self.pic_pos_current.imag
            if abs(x_im) > border:
                correction_vector += 1j * (math.copysign(border, x_im) - x_im)
        elif self.small_zoom:
            unzoom_rate = self.pic_zoom / self.pic_zoom_future - 1
            if unzoom_rate > 0:
                absolute_distance = mpmath.fabs(self.pic_pos_current) / 100
                correction_force = sigmoid(absolute_distance, 0, 15) ** 2 * sigmoid(unzoom_rate * 10, 0, 1.5) ** 3
                correction_vector += - correction_vector * .2 - mpmath.sign(self.pic_pos_current) * correction_force
            if abs(self.pic_move_speed.imag) > 0 and abs(self.pic_move_speed.real) > 0:
                speed = smootherstep_ease((abs(self.pic_move_speed.imag) + 10) /
                                          (abs(self.pic_move_speed.real) + 10) / 8) / 1000
                # print(speed)
                # speed *= sigmoid(abs(self.pic_move_speed.imag) - 1, 0, 40)
                # if abs(self.pic_move_speed.real) and abs(self.pic_move_speed.imag) / abs(self.pic_move_speed.real) > .5:
                correction_vector -= self.pic_pos_current.real * speed * abs(self.pic_pos_current.real)
                # print(correction_vector)
        else:
            right_edge = 1
            if self.interface_mode in {InterfaceMode.LEVELS, InterfaceMode.TRANSFORM}:
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

            if 0 < self.crop_borders_active < 5 and self.transform_mode == 2 and self.interface_mode == InterfaceMode.TRANSFORM:
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

    # def random_image_old(self, jump_type=Actions.IMAGE_RANDOM_UNSEEN_FILE):
    #     if jump_type == Actions.IMAGE_RANDOM_FILE:
    #         self.new_image_index = random.randrange(self.image_count)
    #     elif jump_type == Actions.IMAGE_RANDOM_UNSEEN_FILE:
    #         list_of_not_seen = np.invert(self.seen_images).nonzero()[0]
    #         if list_of_not_seen.any():
    #             self.new_image_index = random.choice(list_of_not_seen)
    #         else:
    #             self.new_image_index = random.randrange(self.image_count)
    #     elif jump_type == Actions.IMAGE_RANDOM_IN_CURRENT_DIR:
    #         dir_first_img, dir_img_count = self.dir_to_file[self.file_to_dir[self.image_index]]
    #         self.new_image_index = dir_first_img + random.randrange(dir_img_count)
    #     elif jump_type == Actions.IMAGE_RANDOM_DIR_FIRST_FILE:
    #         dir_index = random.randrange(self.dir_count)
    #         self.new_image_index = self.dir_to_file[dir_index][0]
    #     elif jump_type == Actions.IMAGE_RANDOM_DIR_RANDOM_FILE:
    #         dir_first_img, dir_img_count = self.dir_to_file[random.randrange(self.dir_count)]
    #         self.new_image_index = dir_first_img + random.randrange(dir_img_count)
    #
    #     self.image_index = self.new_image_index
    #
    #     self.load_image()
    #     self.unschedule_pop_message(8)

    def rand_new_image_index(self, jump_type):
        match jump_type:
            case Actions.IMAGE_RANDOM_UNSEEN_FILE:
                list_of_not_seen = np.invert(self.seen_images).nonzero()[0]
                if list_of_not_seen.any():
                    return random.choice(list_of_not_seen)
                else:
                    return random.randrange(self.image_count)
            case Actions.IMAGE_RANDOM_DIR_FIRST_FILE:
                dir_index = random.randrange(self.dir_count)
                return self.dir_to_file[dir_index][0]
            case Actions.IMAGE_RANDOM_IN_CURRENT_DIR:
                dir_first_img, dir_img_count = self.dir_to_file[self.file_to_dir[self.image_index]]
                return dir_first_img + random.randrange(dir_img_count)
            case Actions.IMAGE_RANDOM_DIR_RANDOM_FILE:
                dir_first_img, dir_img_count = self.dir_to_file[random.randrange(self.dir_count)]
                return dir_first_img + random.randrange(dir_img_count)
            case _:
                return random.randrange(self.image_count)

    def random_image(self, jump_type=Actions.IMAGE_RANDOM_UNSEEN_FILE):
        self.new_image_index = self.rand_new_image_index(jump_type)
        self.image_index = self.new_image_index
        self.load_image()
        self.unschedule_pop_message(8)

    # def first_image(self):
    #     self.new_image_index = 0
    #     self.load_image()

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

        self.release_texture(self.current_texture_old)
        self.current_texture_old = self.current_texture
        self.current_texture = new_texture
        new_texture.use(5)

        self.update_position()
        self.switch_interface_mode(InterfaceMode.GENERAL)

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
        self.switch_interface_mode(InterfaceMode.GENERAL)
        self.levels_edit_band = 3
        self.levels_edit_group = 0
        self.empty_level_borders()
        self.schedule_pop_message(4, 8000000)

    def save_current_texture(self, replace):
        texture_data = self.current_texture.read()
        new_image = Image.frombuffer("RGB", self.current_texture.size, texture_data)
        new_image = new_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.pic_angle_future % 360:  # not at original angle
            if not (self.pic_angle_future % 90):  # but at 90°-ish
                rotation_step = (self.pic_angle_future % 360) // 90 + 1
                new_image = new_image.transpose(rotation_step)

        dir_index = self.file_to_dir[self.image_index]
        dir_name = self.dir_list[dir_index]
        file_name = self.file_list[self.image_index]
        if not replace:
            stripped_file_name = os.path.splitext(file_name)[0]
            file_name = stripped_file_name + "_e" + ".jpg"
        img_path = os.path.join(dir_name, file_name)

        print("Saving under name", img_path)
        new_image.save(img_path, quality=90, exif=self.im_exif, optimize=True)
        self.image_cache.pop(img_path, None)
        pop_message = 5 if replace else 6
        # if not replace:
        #     pop_message = 6
        #     dir_index = self.file_to_dir[self.image_index]
        #     self.dir_to_file[dir_index][1] += 1
        #     for fix_dir in range(dir_index + 1, self.dir_count):
        #         self.dir_to_file[fix_dir][0] += 1
        #     self.file_list.insert(self.image_index + 1, file_name)
        #     self.image_count += 1

        self.schedule_pop_message(pop_message, duration=8, file_name=os.path.basename(img_path))

    def revert_image(self):
        self.current_texture_old = self.current_texture
        self.current_texture = self.image_texture
        self.reset_pic_position()
        self.unschedule_pop_message(4)

    def reset_pic_position(self, full=True, reduced_width=False, reset_mouse=True):
        wnd_width, wnd_height = self.window_size
        if reduced_width:
            wnd_width *= .78

        if self.current_texture != moderngl.Texture:
            self.pic_zoom_future = min(wnd_width / self.current_texture.width,
                                       wnd_height / self.current_texture.height) * .99
            self.pic_zoom_null = self.pic_zoom_future

        if reset_mouse:
            self.mouse_move_cumulative = 0
        self.gesture_mode_timeout = self.timer.time + .2
        self.pic_pos_future = mp.mpc(0)

        if full:
            self.unschedule_pop_message(2)
            self.unschedule_pop_message(4)
            self.program_id = 1 - self.program_id
            # self.pic_zoom = self.pic_zoom_future * self.configs[Configs.STARTING_ZOOM_FACTOR]
            self.pic_zoom = self.pic_zoom_future * config.get(Configs.STARTING_ZOOM_FACTOR)
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

            if self.interface_mode == InterfaceMode.MANDELBROT:
                self.pic_angle_future = -30
                self.pic_zoom = .1
                self.pic_zoom_future = .2

    def move_picture_with_key(self, time_interval):
        dx = time_interval * (self.key_picture_movement[1] - self.key_picture_movement[3]) * 100
        dy = time_interval * (self.key_picture_movement[0] - self.key_picture_movement[2]) * 100
        self.pic_zoom_future *= 1 + (time_interval * (self.key_picture_movement[5] - self.key_picture_movement[4])) * 5
        self.mandel_auto_travel_speed *= 1 + .2 * (
                time_interval * (self.key_picture_movement[7] - self.key_picture_movement[6]))
        self.move_image(dx, -dy)

    def unschedule_pop_message(self, pop_id, force=False):
        for item in self.pop_db:
            if pop_id == item['type']:
                item['end'] = self.timer.time + 1
                if force:
                    self.pop_db.remove(item)

    def schedule_pop_message(self, pop_id, duration=4., shortify=False, **kwargs):
        message_text = POP_MESSAGE_TEXT[pop_id].format(**kwargs)

        for item in self.pop_db:
            if pop_id == item['type']:
                item['full_text'] = message_text
                item['text'] = message_text
                time = self.current_frame_start_time
                item['start'] = time - 1
                item['end'] = time + item['duration']
                return

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
        self.gl_program_round['finish_n'] = int(self.mouse_move_cumulative)

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
        else:
            self.autoflip_speed = 0
            self.unschedule_pop_message(9)
            self.schedule_pop_message(10)

    def update_position_mandel(self):
        mandel_complexity = abs(self.pic_angle) / 10 + self.mandel_auto_complexity
        mandel_complexity *= math.log2(self.pic_zoom) * .66 + 10
        complex_splitter = split_complex_32 if self.use_old_gl else split_complex
        vec4_pos_x, vec4_pos_y = complex_splitter(-self.pic_pos_current / 1000)
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
        self.gl_program_pic[self.program_id]['useCurves'] = (self.interface_mode == InterfaceMode.LEVELS) and \
                                                            self.levels_enabled
        self.gl_program_pic[self.program_id]['count_histograms'] = self.interface_mode == InterfaceMode.LEVELS
        self.gl_program_pic[self.program_id]['show_amount'] = self.transition_stage
        self.gl_program_pic[self.program_id]['hide_borders'] = config.get(Configs.HIDE_BORDERS)
        self.gl_program_pic[self.program_id]['inter_blur'] = config.get(Configs.INTER_BLUR)
        self.gl_program_pic[self.program_id]['pixel_size'] = config.get(Configs.PIXEL_SIZE)
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
        if self.interface_mode == InterfaceMode.SETTINGS:
            if list(Configs)[self.setting_active] == Configs.INTER_BLUR:
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

        self.gl_program_round['finish_n'] = int(self.mouse_move_cumulative)

        if self.interface_mode == InterfaceMode.TRANSFORM:
            self.gl_program_pic[self.program_id]['process_type'] = 1
            self.gl_program_crop['active_border_id'] = self.crop_borders_active
            self.gl_program_crop['crop'] = tuple(self.crop_borders)
            self.gl_program_crop['zoom_scale'] = self.pic_zoom
            self.gl_program_crop['displacement'] = displacement
            self.gl_program_crop['resize_xy'] = self.resize_xy - 1
            self.gl_program_crop['resize_x'] = self.resize_x - 1
            self.gl_program_crop['resize_y'] = self.resize_y - 1
            self.gl_program_crop['angle'] = math.radians(self.pic_angle)

        self.gl_program_browse_squares['zoom_scale'] = self.pic_zoom
        self.gl_program_browse_squares['displacement'] = displacement
        self.gl_program_browse_pic['zoom_scale'] = self.pic_zoom
        self.gl_program_browse_pic['displacement'] = displacement

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)
        self.window_size = Point(width, height)
        wnd_size = (width, height)
        half_wnd_size = (width / 2, height / 2)
        self.gl_program_mandel[0]['half_wnd_size'] = half_wnd_size
        self.gl_program_mandel[1]['half_wnd_size'] = half_wnd_size
        self.gl_program_mandel[2]['half_wnd_size'] = half_wnd_size
        self.gl_program_crop['wnd_size'] = wnd_size
        self.gl_program_browse_squares['wnd_size'] = wnd_size
        self.gl_program_browse_pic['wnd_size'] = wnd_size
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
        elif new_mode == SWITCH_MODE_TINDER:
            self.schedule_pop_message(22, 8000000, True)

    def switch_interface_mode(self, new_mode, toggle=True):
        if toggle and self.interface_mode == new_mode:
            new_mode = InterfaceMode.GENERAL

        self.unschedule_pop_message(self.interface_mode.value + 14 - 50)

        self.interface_mode = new_mode
        self.mouse_buffer *= 0
        self.run_reduce_flipping_speed = 0
        if new_mode != InterfaceMode.GENERAL:
            self.schedule_pop_message(self.interface_mode.value + 14 - 50, 8000, True)

        if new_mode == InterfaceMode.MANDELBROT:
            self.prepare_to_mandelbrot()

        if new_mode == InterfaceMode.LEVELS:
            self.reset_pic_position(full=False, reduced_width=True)
            self.levels_edit_band = 3
            self.update_levels()

    def mouse_gesture_tracking(self, dx, dy, speed=1., dynamic=True):
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
            self.mouse_move_cumulative = 0
            self.mouse_unflipping_speed = 1
            self.rearm_gesture_timeout()
        if self.switch_mode == SWITCH_MODE_COMPARE:
            if math.copysign(1, self.mouse_move_cumulative) != math.copysign(1, mouse_cumulative):
                direction = int(math.copysign(1, self.mouse_move_cumulative))
                mouse_cumulative = self.mouse_move_cumulative
                for _ in [1, 2]:
                    self.run_flip_once = direction
                    self.flip_once()
                    self.mouse_move_cumulative = mouse_cumulative

        self.gl_program_round['finish_n'] = int(self.mouse_move_cumulative)

        if abs(self.mouse_buffer[1]) > 100:
            if self.autoflip_speed:
                pass
            else:
                self.mark_image_and_switch(self.mouse_buffer[1] < 0, compare_mode=True)
            self.mouse_buffer *= 0
            self.rearm_gesture_timeout()

    def rearm_gesture_timeout(self, timeout=.3):
        self.gesture_mode_timeout = self.timer.time + timeout

    def mouse_tin_tracking(self, dx, dy, speed=1, dynamic=True):
        if self.gesture_mode_timeout > self.timer.time:
            self.rearm_gesture_timeout(.2)
            self.mouse_buffer[1] = 0
            return
        dy_antiforce = (self.switch_mode != SWITCH_MODE_COMPARE) * math.copysign(dy * speed, self.mouse_move_cumulative)
        self.mouse_move_cumulative += dx * 1.6 * speed - dy_antiforce
        self.mouse_buffer[1] -= math.copysign(dx, self.mouse_buffer[1])
        if dynamic:
            self.run_reduce_flipping_speed = - .45
        if abs(self.mouse_move_cumulative) > 100:
            self.mark_image_and_switch(self.mouse_move_cumulative > 0)
            self.rearm_gesture_timeout()

        self.gl_program_round['finish_n'] = int(self.mouse_move_cumulative)

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
        # self.tinder_stats[old_image_category + 1] -= 1
        # self.tinder_stats[new_image_category + 1] += 1
        self.tinder_stats_d[old_image_category] -= 1
        self.tinder_stats_d[new_image_category] += 1
        self.mouse_buffer *= 0
        self.mouse_move_cumulative = 0
        self.mouse_unflipping_speed = 1
        if not compare_mode:
            if self.tinder_stats_d[0] != 0:
                self.new_image_index = self.next_unmarked_image(self.image_index)
                directory_changed = self.file_to_dir[self.new_image_index] != self.file_to_dir[self.image_index]
                # if self.random_folder_mode and directory_changed:
                #     random_index = random.choice(self.dir_to_file)[0]
                #     self.new_image_index = self.next_unmarked_image(random_index)
                self.load_image()
            else:
                self.run_flip_once = 1

    def next_unmarked_image(self, start_index):
        while self.image_categories[start_index] != 0:
            start_index = (start_index + 1) % self.image_count
        return start_index

    def small_zoom_jump(self):
        self.thumbs_displacement = -self.pic_zoom * self.pic_pos_current.conjugate() \
                                   / self.pic_square / self.thumb_period / 2
        displacement = mp.nint(self.thumbs_displacement)
        abs_jump = displacement.imag * (self.thumb_row_elements * 2 + 1)
        abs_jump += restrict(displacement.real, -self.thumb_row_elements, self.thumb_row_elements)
        self.run_flip_once = int(abs_jump)
        self.small_zoom = False
        self.thumb_row_elements = 0

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

    def process_rapid_menu(self):
        self.show_rapid_menu = False
        self.mouse_buffer *= 0
        if self.selected_rapid_action >= 0:
            self.discrete_actions(rapid_menu_actions[self.selected_rapid_action][1])

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
        self.gl_program_round['finish_n'] = int(self.mouse_move_cumulative)

    def mouse_position_event(self, x, y, dx, dy):
        if self.pressed_mouse:  # on some systems drag registers as positioning
            self.mouse_drag_event(x, y, dx, dy)
            return
        self.mouse_buffer += [dx, dy]

        if self.interface_mode == InterfaceMode.GENERAL:
            if self.small_zoom:
                self.move_image(dx, -dy, accelerate=.02)
                return
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.move_image(dx, -dy)
                self.mouse_buffer *= 0
                return

            if self.switch_mode == SWITCH_MODE_CIRCLES:
                self.mouse_circle_tracking()
            elif self.switch_mode == SWITCH_MODE_COMPARE:
                self.mouse_gesture_tracking(dx, dy, speed=.2, dynamic=False)
            elif self.switch_mode == SWITCH_MODE_TINDER:
                self.mouse_tin_tracking(dx, dy)

        elif self.interface_mode == InterfaceMode.MENU:
            if self.menu_bottom > 1:
                self.mouse_buffer[1] = restrict(self.mouse_buffer[1], self.menu_top, self.menu_bottom)
                self.imgui.mouse_position_event(20, self.mouse_buffer[1], 0, 0)

        elif self.interface_mode == InterfaceMode.SETTINGS:
            if abs(self.mouse_buffer[1]) > 150:
                self.setting_active += 1 if self.mouse_buffer[1] > 0 else -1
                self.setting_active = self.setting_active % (len(config.settings) + 2)
                self.mouse_buffer *= 0

        elif self.interface_mode == InterfaceMode.LEVELS:
            if abs(self.mouse_buffer[0]) > 200:
                self.levels_edit_group = 1 if self.mouse_buffer[0] > 0 else 0
                self.mouse_buffer *= 0
            if abs(self.mouse_buffer[1]) > 150:
                self.levels_edit_band += 1 if self.mouse_buffer[1] > 0 else -1
                self.levels_edit_band = restrict(self.levels_edit_band, 0, 4)
                self.mouse_buffer *= 0

        elif self.interface_mode == InterfaceMode.TRANSFORM:
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

        elif self.interface_mode == InterfaceMode.MANDELBROT:
            pass
            self.move_image(dx, -dy)

    def visual_move(self, dx, dy):
        if self.pressed_mouse == 1:
            pass
        elif self.pressed_mouse == 2:
            self.pic_zoom_future *= 1 / (1 + 1.02 ** (- dx + dy)) + .5
        elif self.pressed_mouse == 3:
            self.pic_angle_future += (- dx + dy) / 15
            self.unschedule_pop_message(2)

    def mouse_drag_event(self, x, y, dx, dy):
        self.mouse_buffer[1] += dy * .5
        amount = (dy * 5 - dx) / 1500
        self.right_click_start -= (abs(dx) + abs(dy)) * .01
        if self.interface_mode == InterfaceMode.GENERAL:
            if self.pressed_mouse == 1:
                if self.switch_mode == SWITCH_MODE_COMPARE:
                    self.move_image(dx, -dy)
                else:
                    self.mouse_buffer += [dx, dy]
            else:
                self.visual_move(dx, dy)
        elif self.interface_mode == InterfaceMode.MENU:
            self.mouse_buffer[1] = restrict(self.mouse_buffer[1], self.menu_top, self.menu_bottom)
            self.imgui.mouse_position_event(20, self.mouse_buffer[1], 0, 0)
            if self.menu_bottom > 1:
                self.imgui.mouse_press_event(20, self.mouse_buffer[1], 1)

        # elif self.interface_mode == InterfaceMode.SETTINGS:
        #     config_current = self.configs[Configs.i[self.setting_active]]
        #     config_min = self.config_formats[Configs.i[self.setting_active]][0]
        #     config_max = self.config_formats[Configs.i[self.setting_active]][1]
        #     config_new = config_current * (1 - amount) - amount / 1000 * config_max
        #     config_new = restrict(config_new, config_min, config_max)
        #     self.configs[Configs.i[self.setting_active]] = config_new

        elif self.interface_mode == InterfaceMode.SETTINGS:
            if self.setting_active > len(config.settings) - 1:
                return
            active_setting = list(Configs)[self.setting_active]
            config_current = config.get(active_setting)
            config_min, config_max = Configs.FORMATS.value[active_setting.value][:2]
            config_new = config_current * (1 - amount) - amount / 1000 * config_max
            config_new = restrict(config_new, config_min, config_max)
            config.set(active_setting, config_new)

        elif self.interface_mode == InterfaceMode.LEVELS:
            self.adjust_levels(amount)

        elif self.interface_mode == InterfaceMode.TRANSFORM:
            if self.transform_mode == 3:
                self.visual_move(dx, dy)
            else:
                self.adjust_transform(amount, (dx, dy))

        elif self.interface_mode == InterfaceMode.MANDELBROT:
            self.visual_move(dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        if self.interface_mode == InterfaceMode.MANDELBROT:
            self.pic_zoom_future *= 1.3 if y_offset < 0 else .7
            return
        elif self.interface_mode == InterfaceMode.LEVELS:
            if y_offset < 0:
                self.discrete_actions(Actions.LEVELS_NEXT_BAND)
            else:
                self.discrete_actions(Actions.LEVELS_PREVIOUS_BAND)
        elif self.interface_mode == InterfaceMode.TRANSFORM:
            self.transform_mode -= 1 if y_offset > 0 else -1
            self.transform_mode = restrict(self.transform_mode, 1, 3)
            self.crop_borders_active = 0
        else:
            self.run_flip_once = 1 if y_offset < 0 else -1

    def mouse_press_event(self, x, y, button):
        button_code = 4 if button == 3 else button
        self.pressed_mouse = self.pressed_mouse | button_code
        self.move_sensitivity = 0.
        if self.pressed_mouse == 4:
            self.wnd.close()
            return

        if self.pressed_mouse == 2:
            self.right_click_start = self.timer.time
        else:
            self.right_click_start = 0

        if self.interface_mode == InterfaceMode.MENU:
            self.imgui.mouse_press_event(20, self.mouse_buffer[1], button)

        elif self.interface_mode == InterfaceMode.GENERAL:
            if button == 1:
                self.left_click_start = self.timer.time
            if self.pressed_mouse == 3:
                self.random_image()
                self.mouse_buffer *= 0

        elif self.interface_mode == InterfaceMode.LEVELS and self.levels_enabled:
            if self.levels_edit_band == 4:
                self.levels_edit_parameter = (self.levels_edit_group * 2 + self.pressed_mouse) % 5
            elif self.pressed_mouse < 4:
                self.levels_edit_parameter = (self.levels_edit_group * 3 + self.pressed_mouse) % 6

        elif self.interface_mode == InterfaceMode.TRANSFORM:
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
        button_code = 4 if button == 3 else button
        if button_code == 2 and self.timer.time - self.right_click_start < .15:
            if self.interface_mode == InterfaceMode.GENERAL:
                self.switch_interface_mode(InterfaceMode.MENU)
            elif self.interface_mode != InterfaceMode.MANDELBROT:
                self.switch_interface_mode(InterfaceMode.GENERAL)
        if button == 1 and self.interface_mode == InterfaceMode.GENERAL:
            if self.timer.time - self.left_click_start < .15:
                if self.small_zoom:
                    self.pic_zoom_future = 10
                else:
                    self.reset_pic_position(full=False, reset_mouse=self.switch_mode != SWITCH_MODE_COMPARE)
                    self.init_end_time = self.timer.time
            if self.switch_mode != SWITCH_MODE_COMPARE:
                self.show_image_info = self.show_image_info % 2
                self.process_rapid_menu()

        if self.interface_mode == InterfaceMode.MENU and button == 1:
            self.imgui.mouse_release_event(20, self.mouse_buffer[1], button)
            self.menu_bottom = -1

        if self.interface_mode == InterfaceMode.SETTINGS:
            if self.pressed_mouse == 1:
                if self.setting_active == len(config.settings):
                    print("Saving")
                    config.save_settings()
                    self.schedule_pop_message(23)
                elif self.setting_active == len(config.settings) + 1:
                    self.switch_interface_mode(InterfaceMode.GENERAL)

        self.pressed_mouse = self.pressed_mouse & ~button_code
        self.right_click_start = 0
        self.levels_edit_parameter = 0
        if self.interface_mode == InterfaceMode.TRANSFORM:
            if self.transform_mode == 1:
                self.crop_borders_active = 0

    def discrete_actions(self, action):

        if action == Actions.IMAGE_NEXT:
            self.run_key_flipping = 1

        elif action == Actions.IMAGE_PREVIOUS:
            self.run_key_flipping = -1

        elif action == Actions.WINDOW_GOTO_NEXT_SCREEN:
            self.use_screen_id = (self.use_screen_id + 1) % len(self.screens)
            self.wnd._window.set_fullscreen(True, screen=self.screens[self.use_screen_id])
            # current_settings = load_settings()
            # current_settings[Configs.FULL_SCREEN_ID] = self.use_screen_id
            config.set(Configs.FULL_SCREEN_ID, self.use_screen_id)
            config.save_settings()
            # print(str(self.average_frame_time))
            # self.save_settings(current_settings)

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

        elif action in random_image_actions:
            self.random_image(action)

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
            self.show_image_info = (self.show_image_info + 1) % 2

        elif action == Actions.APPLY_TRANSFORM:
            self.apply_transform()

        elif action == Actions.APPLY_ROTATION_90:
            self.save_rotation_90()

        elif action == Actions.SWITCH_MODE_CIRCLES:
            self.switch_swithing_mode(SWITCH_MODE_CIRCLES)

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

        elif action in list(InterfaceMode):
            self.switch_interface_mode(action)

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_IN_ON:
            self.key_picture_movement[5] = True

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_ON:
            self.key_picture_movement[4] = True

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_IN_OFF:
            self.key_picture_movement[5] = False

        elif action == Actions.KEYBOARD_MOVEMENT_ZOOM_OUT_OFF:
            self.key_picture_movement[4] = False

        elif action == Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_ON:
            self.key_picture_movement[6] = True

        elif action == Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_ON:
            self.key_picture_movement[7] = True

        elif action == Actions.KEYBOARD_MOVEMENT_LEFT_BRACKET_OFF:
            self.key_picture_movement[6] = False

        elif action == Actions.KEYBOARD_MOVEMENT_RIGHT_BRACKET_OFF:
            self.key_picture_movement[7] = False

        elif action == Actions.ACTION_SPACE_PRESS:
            if self.switch_mode == Actions.SWITCH_MODE_TINDER:
                self.random_image(Actions.IMAGE_RANDOM_UNSEEN_FILE)
            elif self.autoflip_speed:
                self.random_image(Actions.IMAGE_RANDOM_UNSEEN_FILE)
            else:
                self.run_key_flipping = 1

        elif action == Actions.KEYBOARD_UP_PRESS:
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[0] = True
            elif self.switch_mode == Actions.SWITCH_MODE_TINDER:
                self.run_key_flipping = -1
            else:
                self.random_image(Actions.IMAGE_RANDOM_UNSEEN_FILE)

        elif action == Actions.KEYBOARD_DOWN_PRESS:
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[2] = True
            elif self.switch_mode == Actions.SWITCH_MODE_TINDER:
                self.run_key_flipping = 1
            else:
                self.first_directory_image(0)

        elif action == Actions.KEYBOARD_LEFT_PRESS:
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[1] = True
            elif self.switch_mode == Actions.SWITCH_MODE_TINDER:
                self.mark_image_and_switch(0)
            else:
                self.run_key_flipping = -1

        elif action == Actions.KEYBOARD_RIGHT_PRESS:
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[3] = True
            elif self.switch_mode == Actions.SWITCH_MODE_TINDER:
                self.mark_image_and_switch(1)
            else:
                self.run_key_flipping = 1

        elif action == Actions.KEYBOARD_UP_RELEASE:
            self.run_key_flipping = 0
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[0] = False

        elif action == Actions.KEYBOARD_DOWN_RELEASE:
            self.run_key_flipping = 0
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[2] = False

        elif action == Actions.KEYBOARD_LEFT_RELEASE:
            self.run_key_flipping = 0
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[1] = False

        elif action == Actions.KEYBOARD_RIGHT_RELEASE:
            self.run_key_flipping = 0
            if self.big_zoom and self.switch_mode != SWITCH_MODE_COMPARE:
                self.key_picture_movement[3] = False

        elif action == Actions.KEYBOARD_MOVEMENT_MOVE_ALL_RELEASE:
            self.key_picture_movement[0] = False
            self.key_picture_movement[1] = False
            self.key_picture_movement[2] = False
            self.key_picture_movement[3] = False

        elif action == Actions.KEYBOARD_FLIPPING_FAST_NEXT_ON:
            self.run_key_flipping = 4

        elif action == Actions.KEYBOARD_FLIPPING_FAST_PREVIOUS_ON:
            self.run_key_flipping = -4

        elif action == Actions.ACTION_SPACE_RELEASE:
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

        elif Actions.KEYBOARD_PRESS_9 >= action >= Actions.KEYBOARD_PRESS_1:
            self.digit_entered(str(action - 110))

    def key_event(self, key, action, modifiers):
        find_key = (action, self.interface_mode, modifiers.ctrl, modifiers.shift, modifiers.alt, key)
        found_action = KEYBOARD_SHORTCUTS.get(find_key)

        if found_action:
            self.discrete_actions(found_action)

        if action == self.wnd.keys.ACTION_PRESS:
            self.last_key_press_time = self.timer.time

    def digit_entered(self, digit):
        self.entered_digits = self.entered_digits[-3:] + digit
        show_string = self.entered_digits[:2]
        if len(self.entered_digits) > 2:
            show_string += "." + self.entered_digits[2:]
        self.schedule_pop_message(pop_id=25, duration=4, jump_percent=show_string)
        self.digit_flop_time = self.current_frame_start_time + 3

    def jump_to_percent(self):
        self.digit_flop_time = 0
        # percent_to_jump = int(self.entered_digits)
        divider = 100
        if len(self.entered_digits) == 3:
            divider = 1000
        elif len(self.entered_digits) == 4:
            divider = 10000
        self.new_image_index = int(self.image_count * int(self.entered_digits) / divider)
        self.entered_digits = ""
        self.load_next_existing_image()

    def unseen_image_routine(self):
        if self.current_frame_start_time > self.last_image_load_time + IMAGE_UN_UNSEE_TIME:
            if self.seen_images.all():
                self.seen_images = np.zeros(self.image_count, dtype=bool)
                self.current_image_is_unseen = True
                self.all_images_seen_times += 1
                if self.all_images_seen_times > 1:
                    self.schedule_pop_message(21, 8000, many_times=f'{self.all_images_seen_times:d} times')
                elif self.all_images_seen_times == 1:
                    self.schedule_pop_message(21, 8000, many_times='')

            self.seen_images[self.image_index] = True
            self.current_image_is_unseen = False

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
        sum_tex = self.mandel_stat_buffer.read()
        hg = np.frombuffer(sum_tex, dtype=np.uint32).reshape(64, 32).copy(order='F')
        self.mandel_zones_hg = hg / (hg.max() + 1)
        self.mandel_stat_buffer.write(self.mandel_stat_empty)
        # self.mandel_stat_buffer.clear()

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

        if self.pic_zoom > (4e10 if self.use_old_gl else 4e28):
            self.mandel_id = 0
        elif self.pic_zoom > (2e5 if self.use_old_gl else 2e13):
            self.mandel_id = 1
        else:
            self.mandel_id = 2
        self.update_position_mandel()

    def close(self):
        print("Closing program")
        if self.thumb_loader != multiprocessing.Process:
            try:
                self.thumb_queue_tasks.put(None)
            finally:
                pass

            self.thumb_loader.join(timeout=1)
            if self.thumb_loader.exitcode is None:
                self.thumb_loader.terminate()
        if self.image_loader != multiprocessing.Process:
            try:
                self.image_queue_tasks.put(None)
            finally:
                pass

            self.image_loader.join(timeout=1)
            if self.image_loader.exitcode is None:
                self.image_loader.terminate()

    def render_preload_imgui(self):
        self.wnd.swap_buffers()
        self.wnd.clear()

    def render_preload(self):
        self.wnd.swap_buffers()
        self.wnd.clear()

        imgui.new_frame()
        imgui.set_next_window_position(10, self.next_message_top, 1, pivot_x=0, pivot_y=0)
        imgui.begin("Loading", True, SIDE_WND_FLAGS)
        imgui.set_window_font_scale(2)
        imgui.text("Loading image list")
        imgui.text(f"So far images loaded:")
        imgui.text(f"{self.image_count}")
        imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def render(self, time=0, frame_time=0):
        if not self.init_end_time:
            self.render_preload()
            return

        self.current_frame_start_time, self.last_frame_duration = self.timer.next_frame()
        if self.reset_frame_timer:
            self.last_frame_duration = 1 / 60
        # frame_time_chunk = (frame_time / 3 + .02) ** .5 - .14  # Desmos: \ \left(\frac{x}{3}+.02\right)^{.5}-.14
        frame_time_chunk = (self.last_frame_duration / 3 + .02) ** .5 - .14
        if frame_time_chunk > .5:
            frame_time_chunk = .5

        self.wnd.swap_buffers()
        self.wnd.clear()
        if self.wnd.is_closing:
            return

        if self.interface_mode == InterfaceMode.MANDELBROT:
            self.mandelbrot_routine(self.current_frame_start_time, frame_time_chunk)
            self.picture_vao.render(self.gl_program_mandel[self.mandel_id], vertices=1)
        else:
            self.update_position()
            if self.interface_mode == InterfaceMode.LEVELS:
                self.read_and_clear_histo()
            if self.transition_stage < 1:
                if type(self.current_texture_old) is moderngl.Texture:
                    self.current_texture_old.use(5)
                    self.picture_vao.render(self.gl_program_pic[1 - self.program_id], vertices=1)

            if self.small_zoom:
                self.render_small_zoom()
            else:
                self.update_preload_cache()

            self.current_texture.use(5)
            self.picture_vao.render(self.gl_program_pic[self.program_id], vertices=1)

            if self.switch_mode == SWITCH_MODE_COMPARE and type(self.current_texture_old) is moderngl.Texture:
                self.render_compare()

            self.picture_vao.transform(self.gl_program_borders, self.ret_vertex_buffer, vertices=1)
            self.pic_screen_borders = np.frombuffer(self.ret_vertex_buffer.read(), dtype=np.float32)
            self.pic_screen_borders = np.nan_to_num(self.pic_screen_borders, True, 0, 100, 100)
            self.pic_screen_borders = np.clip(self.pic_screen_borders, (-10400, -10300, -10200, -10100),
                                              (10100, 10200, 10300, 10400))
            if (self.pic_screen_borders[2] == self.pic_screen_borders[0]) \
                    or (self.pic_screen_borders[3] == self.pic_screen_borders[1]):
                self.pic_screen_borders = np.array((0, 110, 100, 210))

            self.check_small_big_zoom()

            if self.interface_mode == InterfaceMode.TRANSFORM:
                self.picture_vao.render(self.gl_program_crop, vertices=1)

            self.ctx.enable_only(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
            if self.switch_mode != SWITCH_MODE_COMPARE:
                self.round_vao.render(self.gl_program_round)
            if self.switch_mode != SWITCH_MODE_CIRCLES and self.interface_mode == InterfaceMode.GENERAL and self.pressed_mouse == 0:
                self.gl_program_round['finish_n'] = int(self.mouse_buffer[1])
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
            self.compute_transition()
        if abs(self.run_reduce_flipping_speed):
            self.reduce_flipping_speed()
        if abs(self.run_flip_once):
            self.flip_once()
        if self.run_key_flipping:
            self.key_flipping()
        if True in self.key_picture_movement:
            self.move_picture_with_key(frame_time_chunk)
        if self.current_image_is_unseen and self.interface_mode != InterfaceMode.MANDELBROT:
            self.unseen_image_routine()
        if self.pic_zoom < 1e-6:
            self.discrete_actions(Actions.CLOSE_PROGRAM)
        if self.interface_mode == InterfaceMode.GENERAL:
            if self.autoflip_speed != 0 and self.pressed_mouse == 0:
                self.do_auto_flip()
            if 0 < self.digit_flop_time < self.current_frame_start_time:
                self.jump_to_percent()
            if self.switch_mode != SWITCH_MODE_COMPARE and self.pressed_mouse == 1 and \
                    self.timer.time - self.left_click_start > .15 and not self.show_rapid_menu:
                self.show_image_info = self.show_image_info or 2
                self.show_rapid_menu = True
                self.mouse_buffer *= 0
                self.run_reduce_flipping_speed = 0
                self.mouse_move_cumulative = 0

    def render_compare(self):
        def tanh_adjusted(x, k=6):
            return 0.5 * (1 + math.tanh(k * (x - 0.5)))

        self.program_id = 1 - self.program_id
        self.update_position()
        self.gl_program_compare['line_position'] = 1 - self.split_line
        self.gl_program_pic[self.program_id]['half_picture'] = self.split_line - 1 * (self.mouse_move_cumulative < 0)
        split_curve = (self.mouse_move_cumulative / 100 % 1)
        self.split_line = mix(self.split_line, tanh_adjusted(split_curve), .2)
        self.current_texture_old.use(5)
        self.picture_vao.render(self.gl_program_pic[self.program_id], vertices=1)
        self.program_id = 1 - self.program_id
        self.picture_vao.render(self.gl_program_compare, vertices=1)

    def render_small_zoom(self):
        if not self.thumb_queue_data.empty():
            self.get_loaded_thumbs()
        try:
            if self.thumbs_for_image_requested != self.thumb_central_index and self.thumb_queue_tasks.empty():
                self.request_thumbs()
        except OSError as e:
            pass  # Happens when queue is closed after program started closing
            return
            # print(f"error {e.args}")

        full_row_size = self.thumb_row_elements * 2 + 1
        row_elements = int(self.window_size.x / (self.pic_square * self.thumb_period) / 2 - .4)
        self.thumb_row_elements = restrict(row_elements, max(self.thumb_row_elements, 1), self.THUMBS_ROW_MAX)
        show_rows_up = math.ceil(self.thumbs_backward / full_row_size)
        show_rows_down = math.ceil(self.thumbs_forward / full_row_size)

        self.thumbs_displacement = -self.pic_zoom * self.pic_pos_current.conjugate() / \
                                   self.pic_square / self.thumb_period / 2
        view_shift = int(self.thumbs_displacement.imag) * full_row_size
        central_index = restrict(self.image_index + view_shift, 0, self.image_count)
        first_shown = self.image_index - self.thumbs_backward
        last_shown = self.image_index + self.thumbs_forward
        if first_shown + 5 * full_row_size > central_index and first_shown > 0:
            self.thumbs_backward += 1
            self.thumbs_forward -= 1
            self.thumbs_for_image_requested = -1
        if last_shown < central_index + 5 * full_row_size and last_shown < self.image_count:
            self.thumbs_backward -= 1
            self.thumbs_forward += 1
            self.thumbs_for_image_requested = -1

        self.gl_program_browse_squares['row_elements'] = self.thumb_row_elements
        self.gl_program_browse_pic['row_elements'] = self.thumb_row_elements
        self.gl_program_browse_pic['thumb_offset'] = self.thumb_id
        self.gl_program_browse_pic['thumbs_backward'] = self.thumbs_backward
        self.gl_program_browse_pic['thumbs_forward'] = self.thumbs_forward

        for row in range(-show_rows_up, show_rows_down + 1):
            self.gl_program_browse_squares['row'] = row
            self.gl_program_browse_pic['row'] = row
            self.picture_vao.render(self.gl_program_browse_squares, vertices=1)
            self.picture_vao.render(self.gl_program_browse_pic, vertices=1)

    def check_small_big_zoom(self):
        pic_w = self.pic_screen_borders[2] - self.pic_screen_borders[0]
        pic_h = self.pic_screen_borders[3] - self.pic_screen_borders[1]
        self.pic_square = max(pic_w, pic_h, 1)

        if self.timer.time - self.init_end_time > 1:
            big_zoom_x = pic_w > self.window_size.x * 1.2
            big_zoom_y = pic_h > self.window_size.y * 1.2
            self.big_zoom = (big_zoom_x or big_zoom_y) and (self.pic_zoom_null != self.pic_zoom_future)
            small_zoom_x = self.pic_square < self.window_size.x * .8
            small_zoom_y = self.pic_square < self.window_size.y * .8
            if small_zoom_x and small_zoom_y and self.image_count:
                self.small_zoom = True
            else:
                if self.small_zoom:
                    self.small_zoom_jump()

    def render_ui(self):
        imgui.new_frame()
        self.imgui_style.alpha = 1

        if self.central_message_showing:
            self.imgui_central_message()

        # Settings window
        if self.interface_mode == InterfaceMode.SETTINGS:
            self.imgui_settings()

        # Levels window
        if self.interface_mode == InterfaceMode.LEVELS:
            self.imgui_levels()

        self.next_message_top = 10

        if self.interface_mode == InterfaceMode.MANDELBROT:
            self.imgui_mandelbrot()

        elif self.show_image_info:
            self.imgui_image_info()

        if self.show_rapid_menu:
            self.imgui_rapid_menu()

        if self.interface_mode == InterfaceMode.TRANSFORM:
            self.imgui_transforms()

        if self.switch_mode == SWITCH_MODE_TINDER:
            self.imgui_tinder_stats()
            self.imgui_mpx_info()

        # Menu window
        if self.interface_mode == InterfaceMode.MENU:
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
        red_colors = [.2] * (len(config.settings) + 2)
        red_colors[self.setting_active] = .7
        imgui.set_next_window_position(pos_x, pos_y, 1, pivot_x=.5, pivot_y=0.5)
        imgui.set_next_window_bg_alpha(.9)
        imgui.begin("Settings", False, CENRAL_WND_FLAGS)

        for key_id, key in enumerate(Configs.Values.value):
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, red_colors.pop(0), .2, .2)
            imgui.slider_float(Configs.DESCRIPTIONS.value[key], config.settings[key], *Configs.FORMATS.value[key])
            imgui.pop_style_color()

        imgui.dummy(10, 10)
        imgui.push_style_color(imgui.COLOR_BUTTON, red_colors.pop(0), .2, .2)
        imgui.small_button("Save settings as default")
        imgui.pop_style_color()
        imgui.push_style_color(imgui.COLOR_BUTTON, red_colors.pop(0), .2, .2)
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

            if self.menu_bottom < 0:
                menu_clicked = True
            self.next_message_top += imgui.get_window_height()
            self.menu_bottom = self.next_message_top
            imgui.end_popup()

        if menu_clicked:
            if self.interface_mode == InterfaceMode.MENU:
                self.menu_bottom = 1
                self.show_image_info = 0
                self.switch_interface_mode(InterfaceMode.GENERAL)

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

        def add_cells_with_text(texts, list_of_selected, span_columns=None, do_center=False):
            for n, text in enumerate(texts):
                letters_blue = .5 if n in list_of_selected else 1
                imgui.push_style_color(imgui.STYLE_ALPHA, 1, 1, letters_blue)
                column_w = column_width
                if span_columns:
                    column_w *= span_columns[n]
                    if len(texts) > 1:
                        imgui.set_column_width(n, column_w)

                if do_center:
                    # Calculate the horizontal offset to center-align the text in the column
                    offset_x = (column_w - imgui.calc_text_size(text).x - 1) // 2
                    imgui_cursor = imgui.get_cursor_pos_x()
                    imgui.set_cursor_pos_x(imgui_cursor + offset_x)
                    # imgui.set_column_offset(n, 40 * n)
                    # imgui.align_text_to_frame_padding()
                    # imgui.same_line(offset_x)

                imgui.text(text)
                imgui.pop_style_color()
                imgui.next_column()
            # imgui.set_column_width(table_row, column_width)

        pos_x, pos_y = self.imgui_io.display_size.x, self.imgui_io.display_size.y * .5
        imgui.set_next_window_position(pos_x, pos_y, 1, pivot_x=1, pivot_y=0.5)
        imgui.set_next_window_size(0, 0)

        column_width = self.imgui_io.display_size.x // 30
        hg_size = ((column_width - 2) * 6, (self.imgui_io.display_size.y - 9 * line_height * 2) / 6)

        imgui.set_next_window_bg_alpha(.8)
        imgui.begin("Levels settings", True, SIDE_WND_FLAGS)

        hg_names = ("Gray", "Red", "Green", "Blue", "RGB")
        hg_colors = ((0.55, 0.55, 0.55),
                     (0.7, 0.3, 0.3),
                     (0.3, 0.7, 0.3),
                     (0.3, 0.3, 0.7),
                     (0.8, 0.8, 0.8))

        for hg_num in range(5):
            hg_bg_color = .5 if hg_num == self.levels_edit_band + 1 else .3
            imgui.text(hg_names[hg_num] + " Histogram")
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *(hg_bg_color,) * 4)
            imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, *hg_colors[hg_num])
            imgui.plot_histogram("", self.histogram_array[hg_num], graph_size=hg_size)
            imgui.pop_style_color(2)

        self.imgui_style.alpha = .2 + .6 * self.levels_enabled

        imgui.columns(1)
        add_cells_with_text(["Levels"], [self.levels_edit_band // 4], (6,), do_center=True)  # , (6, ), do_center=True

        imgui.columns(3)
        sel_group = (self.levels_edit_group + 1) * (1 - self.levels_edit_band // 4)
        add_cells_with_text(["", "Input", "Output"], [sel_group], (1, 3, 2), do_center=True)

        imgui.columns(6)
        selected = range(self.levels_edit_group * 3 + 1,
                         self.levels_edit_group * 3 + 4) if self.levels_edit_band < 4 else []
        add_cells_with_text(["", " min", " max", "gamma", " min", " max"], selected, do_center=True)

        color_red = (.7, .2, .2)
        color_blue = (.2, .2, .6)
        color_gray = (.2, .2, .2)

        def add_one_slider(column, row_number, wide, column_is_active, band_active, sell_active):
            bg_color = color_gray
            if band_active:
                if sell_active:
                    bg_color = color_red
                elif column_is_active:
                    bg_color = color_blue
            imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *bg_color)
            imgui.slider_float("", self.levels_borders[column][row_number], 0, (1, 10)[wide], '%.2f', (0, 32)[wide])
            imgui.pop_style_color()
            imgui.next_column()

        active_columns = range(self.levels_edit_group * 3, self.levels_edit_group * 3 + 3)
        for table_row in range(4):
            imgui.text(hg_names[table_row + 1])
            imgui.next_column()
            for col in range(5):
                add_one_slider(col, table_row, col == 2, col in active_columns,
                               self.levels_edit_band == table_row, col == self.levels_edit_parameter - 1)

        imgui.columns(2)
        add_cells_with_text(["Pre-levels Saturation", "Post-levels Saturation"], [self.levels_edit_group], (3, 3),
                            do_center=True)
        imgui.columns(4)
        selected = self.levels_edit_group * 2 * (self.levels_edit_band // 4)
        add_cells_with_text(["Hard", "Soft"] * 2, [selected, selected + 1], do_center=True)

        for sat_column in range(4):
            add_one_slider(5, sat_column, True, sat_column // 2 == self.levels_edit_group,
                           self.levels_edit_band == 4, sat_column == self.levels_edit_parameter - 1)

        imgui.set_window_font_scale(1)
        imgui.end()

    def imgui_show_info_window(self, text_list, win_name, from_bottom=0):
        win_top = from_bottom if from_bottom else self.next_message_top
        imgui.set_next_window_position(10, win_top, 1, pivot_x=0, pivot_y=1 if from_bottom else 0)
        imgui.begin(win_name, True, SIDE_WND_FLAGS)
        for text_ in text_list:
            imgui.text(text_)

        win_height = imgui.get_window_height()
        imgui.end()

        if from_bottom:
            return from_bottom - win_height
        else:
            self.next_message_top += win_height

    def imgui_image_info(self):
        self.imgui_style.alpha = .7
        full_file_path = self.get_file_path(self.image_index)
        full_file_dir, file_name = os.path.split(full_file_path)
        folder_path, folder_name = os.path.split(full_file_dir)
        folder_path_text = "Folder path: "
        string_width = 42
        percent_in_list = (self.image_index + 1) / (1 + self.image_count) * 100
        string_width_minus = string_width - len(folder_path_text)
        folder_path_text += folder_path[:string_width_minus] + "\n"
        for path_chunk_start in range(string_width_minus, len(folder_path), string_width):
            folder_path_text += folder_path[path_chunk_start: path_chunk_start + string_width] + "\n"

        dir_index = self.file_to_dir[self.image_index]
        images_in_folder = self.dir_to_file[dir_index][1]
        index_in_folder = self.image_index + 1 - self.dir_to_file[dir_index][0]

        info_text = [
            "File   name: " + file_name,
            "Folder name: " + folder_name,
            folder_path_text,
            f"Image in all list: {self.image_index + 1:d} of {self.image_count:d} ({percent_in_list:.2f}%)",
        ]
        if self.image_index < len(self.image_ratings):
            image_rating = self.image_ratings[self.image_index]
            if image_rating:
                info_text.append(f"Image rating: {int(image_rating) / 1000000:.3f}%")
        else:
            info_text += [f"Folder #: {dir_index + 1:d} of {self.dir_count:d}",
                          f"Image in folder: {index_in_folder:d} of {images_in_folder:d}",]
        next_bottom = self.imgui_show_info_window(info_text, "Path info", self.imgui_io.display_size.y - 10)

        im_mp = self.image_original_size.x * self.image_original_size.y / 1000000
        info_text = [
            "File size: " + format_bytes_3(self.current_image_file_size),
            f"Image size: {self.image_original_size.x} x {self.image_original_size.y}",
            f"Image size: {im_mp:.2f} megapixels",
            f"Current zoom: {self.pic_zoom * 100:.1f}%",
            f"Visual rotation angle: {self.pic_angle:.2f}°", ]

        self.imgui_show_info_window(info_text, "Image info", next_bottom)

    def imgui_rapid_menu(self):
        grid_element_size = self.imgui_io.display_size.y // 12
        circle_rad = grid_element_size // 2

        self.selected_rapid_action = 4 + round(self.mouse_buffer[0] / grid_element_size) + \
                                     round(self.mouse_buffer[1] / grid_element_size) * 3
        if max(abs(self.mouse_buffer / grid_element_size)) >= 1.5:
            self.selected_rapid_action = -1

        # Define the colors for the highlighted and normal buttons
        highlighted_color = (9.0, 0.2, 0.1, 1.0)  # Red color for the highlighted button
        normal_color = (0.2, 0.3, 0.5, 1.0)  # Gray color for normal buttons

        self.imgui_style.alpha = .8
        imgui.set_next_window_size(grid_element_size * 3.5, grid_element_size * 3.5)
        imgui.set_next_window_position(self.imgui_io.display_size.x * .99,
                                       self.imgui_io.display_size.y // 2, 1,
                                       pivot_x=1, pivot_y=.5)

        imgui.begin("Quick actions", False, CENRAL_WND_FLAGS)

        imgui.columns(3)

        for i in range(9):
            button_color = highlighted_color if i == self.selected_rapid_action else normal_color
            imgui.push_style_color(imgui.COLOR_BUTTON, *button_color)
            imgui.button(rapid_menu_actions[i][0], width=grid_element_size, height=grid_element_size)
            imgui.next_column()
            imgui.pop_style_color()

        imgui.columns(1)

        header_height = 35
        win_start = imgui.get_window_position()
        win_size = np.ones(2) * imgui.get_window_size() - (0, header_height)
        circle_coord = win_start + win_size / 2 + (0, header_height) + self.mouse_buffer
        draw_list = imgui.get_window_draw_list()
        draw_list.add_circle(*circle_coord.tolist(), circle_rad, imgui.get_color_u32_rgba(.1, .8, 3, .9), 4, 3)
        imgui.end()

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
        im_mp_1 = self.image_original_size.x * self.image_original_size.y / 1_000_000
        im_mp_2 = self.current_texture.width * self.current_texture.height / 1_000_000
        # im_mp_3 = int(new_image_width) * int(new_image_height) / 1000000
        im_mp_3 = int(new_texture_size.x) * int(new_texture_size.y) / 1_000_000
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
        imgui.slider_float("Scale whole image", self.resize_xy * 100, 10, 1000, '%.2f', 32)
        imgui.pop_style_color()
        push_bg_color(3, 1)
        imgui.slider_float("Scale width", self.resize_x * 100, 10, 1000, '%.2f', 32)
        imgui.pop_style_color()
        push_bg_color(3, 1)
        imgui.slider_float("Scale height", self.resize_y * 100, 10, 1000, '%.2f', 32)
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

    def imgui_mpx_info(self):
        def text_centered(text):
            offset_x = (column_w - imgui.calc_text_size(text).x - 1) // 2
            imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + offset_x)
            imgui.text(text)

        next_message_top_r = 10
        column_w = 90
        pos_x = self.imgui_io.display_size.x
        im_mp_1 = self.image_original_size.x * self.image_original_size.y / 100000
        size_value = restrict(int(math.log10(im_mp_1) * 30), 0, 100)
        mpix_text = f"{im_mp_1 / 10:.{2 if im_mp_1 < 100 else 1}f} Mpx"

        imgui.set_next_window_position(pos_x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)

        imgui.push_style_color(imgui.COLOR_BORDER, .5, .5, .5, .1)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .1, .2, .2, .6)
        imgui.begin("Img size", True, SIDE_WND_FLAGS)
        imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, 1 - size_value / 100, size_value / 100, .2)
        imgui.set_window_font_scale(1)
        self.imgui_style.alpha = .7

        imgui.progress_bar(size_value / 100, (column_w, 20), mpix_text)

        next_message_top_r += imgui.get_window_height()

        text_centered(f"{self.image_original_size.x} x {self.image_original_size.y}")

        imgui.pop_style_color()
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
        info_text = [f"-: {self.tinder_stats_d[-1]}", f"  {self.tinder_stats_d[0]}  ", f"+: {self.tinder_stats_d[1]}"]
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
            images_in_folder = self.dir_to_file[dir_index][1]
            index_in_folder = image_index + 1 - self.dir_to_file[dir_index][0]
            info_text = ["File name: " + os.path.basename(self.get_file_path(image_index)),
                         "Image # (current folder): " + f"{index_in_folder:d} of {images_in_folder:d}",
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
    # start_fullscreen = False if "-f" in sys.argv else True
    # exclude_plus_minus = "-exclude_plus_minus" in sys.argv
    # random_folder_mode = "-random_folder_mode" in sys.argv

    # enable_vsync = True
    # window = mglw.get_local_window_cls('pyglet')(fullscreen=start_fullscreen, vsync=enable_vsync)
    thumb_queue_tasks, thumb_queue_data, thumb_loader = init_image_loader(do_thumb=True)
    image_queue_tasks, image_queue_data, image_loader = init_image_loader()
    window = mglw.get_local_window_cls('pyglet')(vsync=True)
    display = pyglet.canvas.get_display()
    screens = display.get_screens()
    # load_screen_id = int(config.get(Configs.FULL_SCREEN_ID))
    # load_screen_id = 0 if load_screen_id >= len(screens) else load_screen_id
    load_screen_id = min(int(config.get(Configs.FULL_SCREEN_ID)), len(screens) - 1)

    if program_args.full_screen:
        window.mouse_exclusivity = True
        window._window.set_fullscreen(True, screen=screens[load_screen_id])

    window.print_context_info()
    mglw.activate_context(window=window)

    window.config = ModernSlideShower(ctx=window.ctx, wnd=window, timer=timer)
    # window.config.exclude_plus_minus = "-exclude_plus_minus" in sys.argv
    # window.config.random_folder_mode = "-random_folder_mode" in sys.argv
    # if "-scan_all_files" in sys.argv:
    #     window.config.scan_all_files = True

    timer.next_frame()
    timer.next_frame()
    window.config.post_init()
    window.config.thumb_queue_tasks = thumb_queue_tasks
    window.config.thumb_queue_data = thumb_queue_data
    window.config.thumb_loader = thumb_loader
    window.config.image_queue_tasks = image_queue_tasks
    window.config.image_queue_data = image_queue_data
    window.config.image_loader = image_loader
    window.config.screens = screens
    window.config.use_screen_id = load_screen_id
    print(f"Startup time: {timer.time:.02f} s")

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


def setup_parser():
    parser.add_argument("--base_folder_name", help="Specify folder name of base folder in path")
    parser.add_argument("-f", "--full_screen", action="store_true", help="Start in full screen mode")
    parser.add_argument("-r", "-F5", "--random_image", action="store_true", help="Start with random image")
    parser.add_argument("-rd", "-F7", "--random_dir", action="store_true",
                        help="Start with first image in random subdirectory")
    parser.add_argument("--old_gl", action="store_true", help="Use old opengl for compatibility")
    parser.add_argument("-t", "--tinder_mode", action="store_true", help="Open viewer in accept/reject mode")
    parser.add_argument("--skip_load", type=int, choices=range(0, 20),
                        help="A level of skipping when loading list. At 1 skip every two images, at N skip every 2^N+1 image.")
    parser.add_argument("-e", "--exclude_sorted", action="store_true", help="Do not process directories named ++ or --")
    parser.add_argument("-i", "--ignore_extention", action="store_true",
                        help="Try to open every file regardless of its extension, scan all files.")
    parser.add_argument("path", nargs='*', help="Path(s) to file(s) and/or directory(s) to show")


if __name__ == '__main__':
    timer = mglw.timers.clock.Timer()
    timer.start()
    config = Config()
    parser = argparse.ArgumentParser(description="ModernSlideShower, a smooth image viewer", allow_abbrev=False)
    setup_parser()
    program_args, unknown_args = parser.parse_known_args()
    print("Detected arguments:\n", program_args)
    print(f"Unknown arguments:\n {unknown_args}\n" if unknown_args else "\n")
    main_loop()
