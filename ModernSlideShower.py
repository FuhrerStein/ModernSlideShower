import datetime
import shutil
import imgui
import moderngl
import moderngl_window as mglw
import moderngl_window.context.base
import moderngl_window.meta
import moderngl_window.opengl.vao
import moderngl_window.timers.clock
from moderngl_window.opengl import program
from moderngl_window.integrations.imgui import ModernglWindowRenderer
from PIL import Image
import numpy as np
import math
import os
import sys
import random
import collections

# from scipy.interpolate import BSpline
HIDE_BORDERS = 1
TRANSITION_SPEED = 2
INTER_BLUR = 3
STARTING_ZOOM_FACTOR = 4

BUTTON_STICKING_TIME = 0.3

LIST_FILE_TYPE = 'sldlist'
JPEGTRAN_EXE_PATH = "c:\\Soft\\Programs\\libjpeg-turbo64\\bin"
JPEGTRAN_EXE_FILE = "jpegtran.exe"
JPEGTRAN_OPTIONS = ' -optimize -rotate {0} -trim -copy all -outfile "{1}" "{1}"'

IMAGE_FILE_TYPES = ('jpg', 'png', 'jpeg', 'gif', 'tif', 'tiff')
EMPTY_IMAGE_LIST = "Empty.jpg"
SAVE_FOLDER = ".\\SaveFolder\\"
Point = collections.namedtuple('Point', ['x', 'y'])

#    todo: show compression and decompression times when saving/loading lists
#    todo: lossless jpeg image cropping

#    todo: correlate transition speed with actual flipping rate
#    todo: find memory leaks
#    todo: save settings to file
#    todo: dialog to edit settings
#    todo: program icon
#    todo: shortcuts everywhere <- rewrite this to be more specific
#    todo: filelist position indicator
#    todo: keyboard navigation on picture
#    todo: show full image info
#    todo: explain about jpegtan in case it was not found
#    todo: interface to adjust curves
#    todo: different color curves
#    todo: zooming slideshow mode
#    todo: filtering image with glsl code
#    todo: support for large images by splitting them into smaller subtextures
#    todo: seam resizing
#    todo: set settings change speeds in setting settings
#    todo: add 3-4 user-editable parameters to mandelbrot_mode
#    todo: add other julia sets in mandelbrot_mode
#    todo: tutorial mode to teach user how to use program
#    todo: double-double presision calculation for mandelbrot_mode


#    todone: lossy image cropping in full edit mode
#    todone: make mandelbrot_mode eye candy
#    todone: basic live mandelbrot when no images found
#    todone: decompress list file
#    todone: compress list file
#    todone: generalize settings
#    todone: jump to random folder (not image)
#    todone: jump to random image in same folder
#    todone: simple blur during transition
#    todone: if jpegtan was not found, do not show message to save rotation
#    todone: sticking of enable/disable levels key
#    todone: swap levels adjustments: put gamma min and max, adjust gamma with l+r mouse
#    todone: do not apply borders when apply levels
#    todone: replace simple pop messages with full-fledged messages queue
#    todone: levels edit with left and right mouse
#    todone: show short image info
#    todone: navigate through levels interface with only mouse
#    todone: navigate through settings by mouse
#    todone: adjustable between-image transition
#    todone: do not regenerate empty image if it is already there
#    todone: nice empty image
#    todone: save 90°-rotation when saving edited image
#    todone: autoflip mode
#    todone: smooth image change
#    todone: random image jump from mouse
#    todone: self.pressed_mouse must support middle button with other buttons
#    todone: show friendly message in case no images are found
#    todone: start with random image if playlist had _r in its name
#    todone: pleasant image centering
#    todone: show message when changing folder
#    todone: simple slidelist compression to reduce file size.
#    todone: make image sharpest possible using right mipmap texture
#    todone: show message when ending list
#    todone: update histograms on the fly
#    todone: save edited image
#    todone: interface to adjust levels
#    todone: different color levels
#    todone: mouse action for left+right drag
#    todone: help message on F1 or H
#    todone: free image rotation
#    todone: rotate indicator
#    todone: copy/move indicator
#    todone: work with broken and gray jpegs
#    todone: keep image in visible area
#    todone: mouse flip indicator
#    todone: inertial movement and zooming


round_glsl = '''
#version 330

#if defined VERTEX_SHADER
#define edge 10
#define point_size 25
#define round_alpha .15

layout(location = 0) in vec2 in_position;
layout(location = 1) in int in_index;

uniform vec2 wnd_size;
uniform vec2 displacement;
uniform float round_size;
uniform bool clockwise;
uniform int finish_n;

out float alpha;
out float out_index_alpha;


void main() {
    vec2 new_pos = (in_position * round_size + displacement) / wnd_size - vec2(1, 1);

    float point_n_norm = mod(in_index, 25) * 4;
    float s1 = smoothstep(100 + finish_n * 1.1, 120 + finish_n * 1.2, point_n_norm);
    float s2 = smoothstep(finish_n - 10, finish_n * 1.1, point_n_norm);
    alpha = round_alpha * (s1 + 1 - s2);

    gl_Position = vec4(new_pos, .5, 1.0);
    gl_PointSize = point_size;

    out_index_alpha = alpha;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
in float alpha;
in float out_index_alpha;

void main() {
    float point_alpha = alpha - pow(distance(gl_PointCoord.xy, vec2(0.5, 0.5)), 2.5);
    fragColor = vec4(1, 1, 1, point_alpha);
}
#endif
'''

picture_glsl = '''
#version 430

#if defined VERTEX_SHADER

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;

uniform vec2 displacement;
uniform vec2 pix_size;
uniform vec2 wnd_size;
uniform float zoom_scale;
uniform float angle;
uniform float show_amount;
uniform float transparency;
uniform bool one_by_one;

out vec2 uv0;
out float translucency;
out float f_show_amount;
out float min_edge;
out float max_edge;


void main() {

    mat2 rotate_mat = mat2(cos(angle), - sin(angle), sin(angle), cos(angle));
    mat2 pix_size_mat = mat2(pix_size.x, 0, 0, pix_size.y);
    mat2 wnd_size_mat = mat2(1 / wnd_size.x, 0, 0, 1 / wnd_size.y);
    gl_Position = vec4((in_position * pix_size_mat * rotate_mat + displacement) * zoom_scale * wnd_size_mat, 0, 1);
    uv0 = in_texcoord;
    translucency = 1 - smoothstep(.5, .7, transparency);
    f_show_amount = smoothstep(0, .7, show_amount);
    min_edge = 2.5 * (smoothstep(0, 1, show_amount)) + show_amount;
    max_edge = 2 * (smoothstep(.7, 1, show_amount)) + show_amount;

    if (one_by_one)
    {
        gl_Position = vec4(in_position.x, -in_position.y, 1, 1);
    }

}


#elif defined FRAGMENT_SHADER


layout(binding=5) uniform sampler2D texture0;
layout(binding=6) uniform sampler2D texture_curve;
layout(binding=7, r32ui) uniform uimage2D histogram_texture;
uniform bool useCurves;
uniform bool count_histograms;
uniform vec2 transition_center;
uniform float zoom_scale;

out vec4 fragColor;
in vec2 uv0;
in float translucency;
in float f_show_amount;
in float min_edge;
in float max_edge;

uniform float hide_borders;

void main() {
    vec4 tempColor = textureLod(texture0, uv0, - log(zoom_scale));

    if(useCurves)
    {
        // local by-color curves
        tempColor.r = texelFetch(texture_curve, ivec2((tempColor.r * 255 + .5), 1), 0).r;
        tempColor.g = texelFetch(texture_curve, ivec2((tempColor.g * 255 + .5), 2), 0).r;
        tempColor.b = texelFetch(texture_curve, ivec2((tempColor.b * 255 + .5), 3), 0).r;

        // global curves
        tempColor.r = texelFetch(texture_curve, ivec2((tempColor.r * 255 + .5), 0), 0).r;
        tempColor.g = texelFetch(texture_curve, ivec2((tempColor.g * 255 + .5), 0), 0).r;
        tempColor.b = texelFetch(texture_curve, ivec2((tempColor.b * 255 + .5), 0), 0).r;
    }

    if(count_histograms)
    {
        // gray histogram
        // formula is not perfect and needs to be updated to account for 1.0 - 255 conversion
        int gray_value = int((tempColor.r * 299 + tempColor.g * 587 + tempColor.b * 114) * 51 / 200);
        imageAtomicAdd(histogram_texture, ivec2(gray_value, 0), 1u);

        // red, green and blue histograms
        // todo: use fma()
        imageAtomicAdd(histogram_texture, ivec2(tempColor.r * 255 + .5, 2), 1u);
        imageAtomicAdd(histogram_texture, ivec2(tempColor.g * 255 + .5, 3), 1u);
        imageAtomicAdd(histogram_texture, ivec2(tempColor.b * 255 + .5, 4), 1u);
    }

    // Edges and transitions
    float to_edge_x = pow(smoothstep(-0.5, -.5 + hide_borders, -abs(uv0.x - .5)), .3);
    float to_edge_y = pow(smoothstep(-0.5, -.5 + hide_borders, -abs(uv0.y - .5)), .3);
    float to_edge = smoothstep(0, 1, to_edge_x + to_edge_y) * smoothstep(0, 1,  to_edge_x * to_edge_y);
    float tran_alpha = smoothstep(-min_edge, -max_edge, -length(uv0 - transition_center)) * f_show_amount;

    fragColor = vec4(tempColor.rgb, tran_alpha * translucency * to_edge);
}

#endif
'''


class ModernSlideShower(mglw.WindowConfig):
    gl_version = (4, 3)
    title = "imgui Integration"
    aspect_ratio = None
    clear_color = 0, 0, 0
    wnd = mglw.context.base.BaseWindow

    start_with_random_image = False

    picture_vertices4 = moderngl.VertexArray
    round_vao = moderngl.VertexArray
    crop_vao = moderngl.VertexArray
    ret_vertex_buffer = moderngl.Buffer
    jpegtran_exe = os.path.join(JPEGTRAN_EXE_PATH, JPEGTRAN_EXE_FILE)
    image_count = 0
    dir_count = 0
    image_index = 0
    new_image_index = 0
    dir_list = []
    file_list = []
    dir_to_file = []
    file_to_dir = []

    # mandelbrot_coords = np.array([-1.784, -0.000196, -1.7833, 0.000196])

    image_original_size = Point(0, 0)

    common_path = ""
    im_object = Image
    pic_position = np.array([0., 0.], dtype=np.float64)
    pic_position_future = np.array([0., 0.], dtype=np.float64)
    pic_position_2 = np.array([0., 0.], dtype=np.float64)
    pic_position_2_future = np.array([0., 0.], dtype=np.float64)
    pic_position_speed = np.array([0., 0.], dtype=np.float64)
    # pic_pos_complex = 0j
    # pic_pos_complex_f = 0j
    # pic_speed_complex = 0j
    pic_zoom = .5
    pic_zoom_future = 1.
    pic_angle = 0.
    pic_angle_future = 0.
    gl_program_round = moderngl.program
    gl_program_mandel = moderngl.program
    gl_program_crop = moderngl.program
    gl_program_pic = [moderngl.program] * 2
    gl_program_pic_v = moderngl.program
    program_id = 0
    image_texture = moderngl.Texture
    current_texture = moderngl.Texture
    # image_texture_old = moderngl.Texture
    current_texture_old = moderngl.Texture
    curve_texture = moderngl.Texture
    max_keyboard_flip_speed = .3
    mouse_buffer = np.array([1., 1.])
    mouse_move_atangent = 0.
    mouse_move_atangent_delta = 0.
    mouse_move_cumulative = 0.
    mouse_unflipping_speed = 1.
    round_indicator_cener_pos = 70, 70
    round_indicator_radius = 40
    last_image_folder = None

    transition_center = (.4, .4)

    mandelbrot_mode = False

    configs = {
        HIDE_BORDERS: 0.02,
        TRANSITION_SPEED: .15,
        INTER_BLUR: 10.,
        STARTING_ZOOM_FACTOR: 1.03,
    }

    config_descriptions = {
        HIDE_BORDERS: "Hide image borders",
        TRANSITION_SPEED: "Speed of transition between images",
        INTER_BLUR: "Blur during transition",
        STARTING_ZOOM_FACTOR: "Zoom of newly shown image",
    }

    config_formats = {
        HIDE_BORDERS: (0, 1, '%.3f', 2),
        TRANSITION_SPEED: (0.01, 1, '%.3f', 2),
        INTER_BLUR: (0, 1000, '%.1f', 4),
        STARTING_ZOOM_FACTOR: (0, 5, '%.3f', 4),
    }

    last_key_press_time = 0

    setting_active = 0

    autoflip_speed = 0.

    run_move_image_inertial = False
    run_reduce_flipping_speed = 0.
    run_key_flipping = 0
    key_flipping_next_time = 0.
    run_flip_once = 0
    pressed_mouse = 0

    pop_message_text = ["File {file_name}\nwas moved to folder\n{new_folder}",  # 0
                        "File {file_name}\nwas copied folder\n{new_folder}",  # 1
                        "Image rotated by {angle} degrees. \nPress Enter to save losslessly",  # 2
                        "Rotation saved losslessly",  # 3
                        "Levels correction applied. \nPress F12 to save image with replacement",  # 4
                        "File {file_name}\nsaved with overwrite.",  # 5
                        "File saved with new name\n{file_name}",  # 6
                        "First image in the list",  # 7
                        "Entering folder {current_folder}",  # 8
                        "Autoflipping ON",  # 9
                        "Autoflipping OFF", ]  # 10
    pop_db = []

    histogram_array = np.empty
    histo_texture = moderngl.texture
    histo_texture_empty = moderngl.buffer

    levels_borders = []
    levels_borders_previous = []
    levels_array = np.zeros((4, 256), dtype='uint8')

    levels_open = False
    levels_enabled = True
    levels_edit_band = 0
    levels_edit_parameter = 0
    levels_edit_group = 0

    key_picture_movement = [False] * 6

    transform_mode = 0
    transform_mode_column = 0
    crop_borders = np.array([0.] * 4)

    transition_stage = 1.

    virtual_cursor_position = np.array([0., 0.])

    show_image_info = 0
    current_image_file_size = 0

    central_message_showing = False
    central_message = ["",
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
    [arrow up] : show random image
    [page up] : show first image in previous folder
    [page down] : show first image in next folder
    [M] : move image file out of folder tree into '--' subfolder
    [C] : copy image file out of folder tree into '++' subfolder
    [L] : show/hide levels edit interface
    [I] : show/hide basic image information
    [F12] : save current image as .jpg file with replacement
    [F9] : save current image as *_e.jpg file without replacing original
    [F5] : save current playlist with compression
    [F6] : save current playlist without compression (plain text)
    [P] : show settings window
    [A] : start slideshow (automatic image flipping)
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.init_program()

    def init_program(self):
        self.ret_vertex_buffer = self.ctx.buffer(reserve=32)

        self.picture_vertices4 = mglw.opengl.vao.VAO("main_image")
        # points are flipped to accommodate for inverted texture/image orders
        point_coord = np.array([
            [-1, -1, 1, 1],  # x
            [-1, 1, -1, 1]   # y
        ], dtype=np.float32).T
        texture_coord = np.array([
            [0, 0, 1, 1],    # x
            [1, 0, 1, 0]     # y
        ], dtype=np.float32).T
        self.picture_vertices4.buffer(point_coord, '2f', ['in_position'])
        self.picture_vertices4.buffer(texture_coord, '2f', ['in_texcoord'])
        self.picture_vertices4.index_buffer(np.array([0, 1, 2, 3, 1, 2], dtype=np.int16), 2)

        Image.MAX_IMAGE_PIXELS = 10000 * 10000 * 3
        random.seed()

        picture_program_text = picture_glsl
        picture_program_v_text = ""
        round_program_text = round_glsl
        mandel_program_text = ""
        crop_program_text = ""
        if os.path.isfile('picture.glsl'):
            picture_program_text = open('picture.glsl', 'r').read()
        if os.path.isfile('picture_v.glsl'):
            picture_program_v_text = open('picture_v.glsl', 'r').read()
        if os.path.isfile('round.glsl'):
            round_program_text = open('round.glsl', 'r').read()
        if os.path.isfile('mandelbrot.glsl'):
            mandel_program_text = open('mandelbrot.glsl', 'r').read()
        if os.path.isfile('crop.glsl'):
            crop_program_text = open('crop.glsl', 'r').read()

        dummy_program_description = moderngl_window.meta.ProgramDescription()

        shaders = program.ProgramShaders.from_single(dummy_program_description, picture_program_text)
        self.gl_program_pic = [shaders.create(), shaders.create()]
        shaders = program.ProgramShaders.from_single(dummy_program_description, picture_program_v_text)
        self.gl_program_pic_v = shaders.create()
        shaders = program.ProgramShaders.from_single(dummy_program_description, round_program_text)
        self.gl_program_round = shaders.create()
        shaders = program.ProgramShaders.from_single(dummy_program_description, mandel_program_text)
        self.gl_program_mandel = shaders.create()
        shaders = program.ProgramShaders.from_single(dummy_program_description, crop_program_text)
        self.gl_program_crop = shaders.create()

        self.histo_texture = self.ctx.texture((256, 5), 4)
        self.histo_texture.bind_to_image(7, read=True, write=True)
        self.histo_texture_empty = self.ctx.buffer(reserve=(256 * 5 * 4))
        self.histogram_array = np.zeros((5, 256), dtype=np.float32)
        self.generate_round_geometry()
        self.generate_crop_geometry()
        self.empty_level_borders()
        self.empty_level_borders()
        self.generate_levels_texture()

        self.find_jpegtran()
        self.get_images()

        if self.image_count == 0:
            self.central_message_showing = 2
            self.file_list.append(EMPTY_IMAGE_LIST)
            self.file_to_dir.append(-1)
            self.image_count = 1

        self.find_common_path()
        if self.start_with_random_image:
            self.random_image()
        else:
            self.load_image()
            self.unschedule_pop_message(7)
            self.unschedule_pop_message(8)
        self.current_texture.use(5)
        self.transition_stage = 1

    def previous_level_borders(self):
        self.levels_borders = self.levels_borders_previous
        self.generate_levels_texture()

    def empty_level_borders(self):
        self.levels_borders_previous = self.levels_borders
        self.levels_borders = [[0., 1., 1., 0., 1.].copy() for _ in range(5)]
        self.generate_levels_texture()

    def generate_crop_geometry(self):
        self.crop_vao = mglw.opengl.vao.VAO("crop lines", mode=moderngl.LINES)
        point_coord = np.array([
            [-1, 1.1], [-1, -1.1],
            [1, 1.1], [1, -1.1],
            [-1.1, 1], [1.1, 1],
            [-1.1, -1], [1.1, -1],
        ], dtype=np.float32)
        self.crop_vao.buffer(point_coord, '2f', ['in_position'])

    def generate_round_geometry(self):
        points_count = 50
        points_array_x = np.sin(np.linspace(0., math.pi * 2, points_count, endpoint=False))
        points_array_y = np.cos(np.linspace(0., math.pi * 2, points_count, endpoint=False))
        points_array = np.empty((100,), dtype=points_array_x.dtype)
        points_array[0::2] = points_array_x
        points_array[1::2] = points_array_y
        indices_array_p = np.arange(50, dtype=np.int32)

        self.round_vao = mglw.opengl.vao.VAO("round", mode=moderngl.POINTS)
        self.round_vao.buffer(points_array.astype('f4'), '2f', ['in_position'])
        self.round_vao.buffer(indices_array_p, 'i', ['in_index'])

    def generate_levels_texture(self, band=None):
        if band is None:
            [self.generate_levels_texture(band) for band in range(4)]
            return
        curve_array = np.linspace(0, 1, 256)
        curve_array = (curve_array - self.levels_borders[band][0]) / (
                self.levels_borders[band][1] - self.levels_borders[band][0])
        curve_array = np.clip(curve_array, 0, 1)
        curve_array = curve_array ** self.levels_borders[band][2]
        curve_array = curve_array * (self.levels_borders[band][4] - self.levels_borders[band][3]) + \
                      self.levels_borders[band][3]
        curve_array = np.clip(curve_array * 255, 0, 255)
        self.levels_array[band] = np.around(curve_array).astype('uint8')

        self.curve_texture = self.ctx.texture((256, 4), 1, self.levels_array)
        self.curve_texture.use(location=6)
        self.curve_texture.repeat_x = False

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
                    if file_arguments[0].lower().endswith(IMAGE_FILE_TYPES):
                        self.scan_directory(os.path.dirname(file_arguments[0]), file_arguments[0])
                    else:
                        self.scan_file(file_arguments[0])
                else:
                    [self.scan_file(file) for file in file_arguments]
        else:
            self.scan_directory(os.path.abspath('.\\'))

        print(self.image_count, "total images found")

    def find_common_path(self):
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
                if f.lower().endswith(IMAGE_FILE_TYPES):
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
        if filename.lower().endswith(IMAGE_FILE_TYPES):
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
        dir_index = self.file_to_dir[index]
        if dir_index == -1:
            return EMPTY_IMAGE_LIST
        dir_name = self.dir_list[dir_index]
        file_name = self.file_list[index]
        img_path = os.path.join(dir_name, file_name)
        return img_path

    def reorient_image(self, im):
        try:
            image_exif = im._getexif()
            image_orientation = image_exif[274]
            if image_orientation in (2, '2'):
                return im.transpose(Image.FLIP_LEFT_RIGHT)
            elif image_orientation in (3, '3'):
                return im.transpose(Image.ROTATE_180)
            elif image_orientation in (4, '4'):
                return im.transpose(Image.FLIP_TOP_BOTTOM)
            elif image_orientation in (5, '5'):
                return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
            elif image_orientation in (6, '6'):
                return im.transpose(Image.ROTATE_270)
            elif image_orientation in (7, '7'):
                return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
            elif image_orientation in (8, '8'):
                return im.transpose(Image.ROTATE_90)
            else:
                return im
        except (KeyError, AttributeError, TypeError, IndexError):
            return im

    def release_texture(self, texture):
        if type(texture) is moderngl.texture.Texture:
            try:
                texture.release()
            except Exception:
                pass

    def load_image(self, mandelbrot=False):
        image_path = self.get_file_path()
        if not os.path.isfile(image_path):
            if image_path == EMPTY_IMAGE_LIST:
                mandelbrot = True
            else:
                self.load_next_existing_image()
                return

        if mandelbrot:
            self.image_original_size = Point(self.wnd.width, self.wnd.height)
            image_bytes = np.empty(self.wnd.width * self.wnd.height * 3, dtype=np.uint8)
            self.mandelbrot_mode = True
            self.show_image_info = 1
        else:
            self.mandelbrot_mode = False
            try:
                with Image.open(image_path) as img_buffer:
                    if img_buffer.mode == "RGB":
                        self.im_object = self.reorient_image(img_buffer)
                    else:
                        self.im_object = self.reorient_image(img_buffer).convert(mode="RGB")
                    image_bytes = self.im_object.tobytes()
                    self.image_original_size = Point(self.im_object.width, self.im_object.height)
                    self.current_image_file_size = os.stat(image_path).st_size
            except Exception as e:
                print("Error reading ", self.get_file_path(), e)
                self.load_next_existing_image()
                return

        # self.image_texture_old = self.image_texture
        if self.image_texture != self.current_texture:
            self.release_texture(self.image_texture)
        self.current_texture_old = self.current_texture

        self.image_texture = self.ctx.texture((self.image_original_size.x, self.image_original_size.y), 3, image_bytes)
        self.image_texture.repeat_x = False
        self.image_texture.repeat_y = False
        self.image_texture.build_mipmaps()
        self.image_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.current_texture = self.image_texture
        self.image_index = self.new_image_index

        self.wnd.title = "ModernSlideShower: " + image_path
        self.reset_pic_position()

        current_folder = self.file_to_dir[self.image_index]

        if current_folder != self.last_image_folder:
            self.schedule_pop_message(8, 5, current_folder=self.dir_list[current_folder:current_folder + 1])
        self.last_image_folder = current_folder
        if self.image_index == 0 and not mandelbrot:
            self.schedule_pop_message(7)
            self.pic_zoom = self.pic_zoom_future * (self.configs[STARTING_ZOOM_FACTOR] - .5) ** -1
            self.update_position()
        else:
            self.unschedule_pop_message(7)

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

    def move_file_out(self, do_copy=False):
        full_name = self.get_file_path(self.image_index)
        short_name = os.path.basename(full_name)
        parent_folder = os.path.dirname(full_name)
        own_subfolder = parent_folder[len(self.common_path):]
        prefix_subfolder = ["--", "++"][do_copy]
        if not own_subfolder.startswith("\\"):
            own_subfolder = "\\" + own_subfolder
        new_folder = os.path.join(self.common_path, prefix_subfolder) + own_subfolder
        file_operation = [shutil.move, shutil.copy][do_copy]
        # os.path.join()
        if not os.path.isdir(new_folder):
            try:
                os.makedirs(new_folder)
            except Exception as e:
                print("Could not create folder", e)

        try:
            file_operation(full_name, new_folder)
        except Exception as e:
            # todo good message here
            print("Could not complete file " + ["move", "copy"][do_copy], e)
            return

        self.mouse_move_cumulative = self.mouse_move_cumulative * .05
        if not do_copy:
            dir_index = self.file_to_dir[self.image_index]
            self.dir_to_file[dir_index][1] -= 1
            for fix_dir in range(dir_index + 1, self.dir_count):
                self.dir_to_file[fix_dir][0] -= 1
            self.file_list.pop(self.image_index)
            self.file_to_dir.pop(self.image_index)
            self.image_count -= 1
            self.new_image_index = self.image_index % self.image_count
            # if not self.image_index < self.image_count:
            #     self.new_image_index = 0
            self.schedule_pop_message(0, duration=10, file_name=short_name, new_folder=new_folder)
            self.load_image()
        else:
            self.schedule_pop_message(1, duration=10, file_name=short_name, new_folder=new_folder)

    def save_rotation(self):
        if self.pic_angle_future % 360 and self.jpegtran_exe:
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
                                       self.window_size[0] / self.current_texture.height)
        else:
            self.pic_zoom_future = min(self.window_size[0] / self.current_texture.width,
                                       self.window_size[1] / self.current_texture.height)
        self.move_image()

        if self.pic_angle_future % 360:  # not at original angle
            if not (self.pic_angle_future % 90) and self.jpegtran_exe:  # but at 90°-ish
                self.schedule_pop_message(2, duration=8000000, angle=360 - self.pic_angle_future % 360)
        else:
            self.unschedule_pop_message(2)

    def check_if_image_in_window(self):
        correction_vector = np.array([0., 0.])
        if self.mandelbrot_mode:
            border = max(self.window_size[0], self.window_size[1]) * 1.5
            x_im = self.pic_position[0] - .5 * max(self.window_size[0], self.window_size[1])
            if abs(x_im) > border:
                correction_vector[0] = math.copysign(border, x_im) - x_im

            if abs(self.pic_position[1]) > border:
                correction_vector[1] = math.copysign(border, self.pic_position[1]) - self.pic_position[1]

        else:
            verts = np.frombuffer(self.ret_vertex_buffer.read(), dtype=np.float32).reshape((4, 2))
            image_bottom, image_top = min(verts[:, 1]), max(verts[:, 1])
            image_left, image_right = min(verts[:, 0]), max(verts[:, 0])
            speed = 1000 / self.pic_zoom
            correction_vector[0] -= speed * (image_left * (image_left > 0) + image_right * (image_right < 0))
            correction_vector[1] -= speed * (image_bottom * (image_bottom > 0) + image_top * (image_top < 0))
        return correction_vector, abs(np.linalg.norm(correction_vector))

    def move_image_inertial(self, dt):
        correction_vector, its_size = self.check_if_image_in_window()
        if its_size != 0:
            self.pic_position_future += correction_vector / 10

        if self.transition_stage < 1:  # todo: make so that transition_stage doesn't lag behind mouse_move_cumulative
            transition_step = self.configs[TRANSITION_SPEED] ** 2 * (1.2 - self.transition_stage)
            self.transition_stage += transition_step / (1 - abs(self.mouse_move_cumulative) / 100)
            # print(self.transition_stage)
            if self.transition_stage > .99999:
                self.transition_stage = 1
                # self.release_texture(self.image_texture_old)
                self.release_texture(self.current_texture_old)

        self.pic_position_future += self.pic_position_speed / self.pic_zoom
        # self.pic_position_2_future += self.pic_position_speed
        self.pic_position_speed *= .95
        distance_to_ghost = np.log(1 + np.linalg.norm(self.pic_position - self.pic_position_future))
        abs_speed = np.log(1 + np.linalg.norm(self.pic_position_speed))
        sp_sum = distance_to_ghost * self.pic_zoom_future + abs_speed
        sp_sum2 = .5 + .5 / (math.copysign(1, sp_sum) + 1 / (sp_sum + .01))
        self.pic_position = self.pic_position * (1 - sp_sum2) + self.pic_position_future * sp_sum2
        self.pic_position_2 = self.pic_position_2 * .95 + self.pic_position_future * .05

        scale_disproportion = abs(self.pic_zoom_future / self.pic_zoom - 1) ** .7 * .2
        pic_zoom_new = self.pic_zoom * (1 - scale_disproportion * self.transition_stage ** 2) \
                       + self.pic_zoom_future * scale_disproportion * self.transition_stage ** 2
        if pic_zoom_new / self.pic_zoom < 1:
            centerting_factor = 1 - scale_disproportion / self.pic_zoom / 100
            self.pic_position_future = self.pic_position_future * centerting_factor
            if self.levels_open:
                self.pic_position_future -= np.array((self.window_size[0] / 5, 0)) * (1 - centerting_factor)

        self.pic_zoom = pic_zoom_new

        rotation_disproportion = abs(self.pic_angle_future - self.pic_angle) / 10
        self.pic_angle = self.pic_angle * (1 - .2) + self.pic_angle_future * .2

        self.update_position()

        if sp_sum + scale_disproportion * 20 + rotation_disproportion + (self.transition_stage < 1) < .1:
            self.run_move_image_inertial = False

    def move_image(self, d_coord=np.array([0., 0.])):
        if self.mandelbrot_mode:
            #  todo: make this window_size-relative
            d_coord /= 3
        self.pic_position_speed += d_coord
        self.pic_position_speed *= .97
        self.run_move_image_inertial = True

    def key_flipping(self, time):
        if self.key_flipping_next_time > time:
            return
        self.run_flip_once = 1 if self.run_key_flipping > 0 else -1
        self.key_flipping_next_time = time + .4 / abs(self.run_key_flipping)

    def first_directory_image(self, direction=0):
        dir_index = (self.file_to_dir[self.image_index] + direction) % self.dir_count
        self.new_image_index = self.dir_to_file[dir_index][0]
        self.load_image()

    def next_image(self):
        self.new_image_index = (self.image_index + 1) % self.image_count
        self.load_image()

    def previous_image(self):
        self.new_image_index = (self.image_index - 1) % self.image_count
        self.load_image()

    def random_image(self, jump_type="rand_file"):
        if jump_type == "rand_file":
            self.new_image_index = random.randrange(self.image_count)
        elif jump_type == "current_dir":
            dir_index = self.file_to_dir[self.image_index]
            self.new_image_index = self.dir_to_file[dir_index][0] + random.randrange(self.dir_to_file[dir_index][1])
        elif jump_type == "rand_dir_first_file":
            dir_index = random.randrange(self.dir_count)
            self.new_image_index = self.dir_to_file[dir_index][0]
        elif jump_type == "rand_dir_rand_file":
            dir_index = random.randrange(self.dir_count)
            self.new_image_index = self.dir_to_file[dir_index][0] + random.randrange(self.dir_to_file[dir_index][1])

        self.load_image()
        self.unschedule_pop_message(8)

    def first_image(self):
        self.new_image_index = 0
        self.load_image()

    def apply_transform(self):
        crop_tuple = (int(self.crop_borders[0]), int(self.crop_borders[3]),
                      self.current_texture.size[0] - int(self.crop_borders[1]) - int(self.crop_borders[0]),
                      self.current_texture.size[1] - int(self.crop_borders[2]) - int(self.crop_borders[3]))
        new_texture = self.ctx.texture((crop_tuple[2], crop_tuple[3]), 3)
        temp_buffer = self.ctx.buffer(reserve=crop_tuple[2] * crop_tuple[3] * 3)
        source_framebuffer = self.ctx.framebuffer([self.current_texture])
        source_framebuffer.read_into(temp_buffer, crop_tuple)
        new_texture.write(temp_buffer)

        self.crop_borders *= 0
        if self.current_texture != self.image_texture:
            self.current_texture.release()
        self.current_texture = new_texture

        self.update_position()
        self.transform_mode = 0

    def apply_levels(self):
        new_texture = self.ctx.texture(self.current_texture.size, 3)
        render_framebuffer = self.ctx.framebuffer([new_texture])
        render_framebuffer.clear()
        render_framebuffer.use()
        self.gl_program_pic[self.program_id]['one_by_one'] = True
        self.gl_program_pic[self.program_id]['hide_borders'] = 0
        self.picture_vertices4.render(self.gl_program_pic[self.program_id])
        self.gl_program_pic[self.program_id]['one_by_one'] = False
        self.gl_program_pic[self.program_id]['hide_borders'] = self.configs[HIDE_BORDERS]
        self.ctx.screen.use()
        new_texture.use(5)
        self.current_texture = new_texture
        self.levels_open = False
        self.levels_edit_band = 0
        self.empty_level_borders()
        self.generate_levels_texture()
        self.update_position()
        self.schedule_pop_message(4, 8000000)

    def save_current_texture(self, replace):
        texture_data = self.current_texture.read()
        new_image = Image.frombuffer("RGB", self.current_texture.size, texture_data)
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

    def show_curves_interface(self):
        self.levels_open = not self.levels_open
        self.update_position()

    def reset_pic_position(self, full=True):
        wnd_width, wnd_height = self.wnd.size
        self.pic_zoom_future = min(wnd_width / self.current_texture.width, wnd_height / self.current_texture.height)
        self.pic_position_future = np.array([-100., 0.], dtype=np.float64)
        self.run_move_image_inertial = True

        if full:
            self.unschedule_pop_message(2)
            self.unschedule_pop_message(4)
            self.program_id = 1 - self.program_id
            self.pic_zoom = self.pic_zoom_future * self.configs[STARTING_ZOOM_FACTOR]
            self.pic_position = np.array([0., 0.], dtype=np.float64)
            self.pic_position_future = np.array([0., 0.], dtype=np.float64)
            self.pic_position_speed = np.array([0., 0.], dtype=np.float64)
            self.pic_angle = 0.
            self.pic_angle_future = 0.
            self.transition_stage = 0.
            self.transition_center = (.3 + .4 * random.random(), .3 + .4 * random.random())

            if self.mandelbrot_mode:
                self.pic_angle_future = -30

        self.update_position()

    def move_picture_with_key(self, time_interval):
        dx = time_interval * (self.key_picture_movement[1] - self.key_picture_movement[3])
        dy = time_interval * (self.key_picture_movement[0] - self.key_picture_movement[2])
        d_coord = np.array([dx, -dy], dtype=np.float64) * 100
        self.pic_zoom_future *= 1 + (time_interval * (self.key_picture_movement[5] - self.key_picture_movement[4]))
        self.move_image(d_coord)

    def unschedule_pop_message(self, pop_id):
        for item in self.pop_db:
            if pop_id == item['type']:
                self.pop_db.remove(item)

    def schedule_pop_message(self, pop_id, duration=4., **kwargs):
        self.unschedule_pop_message(pop_id)
        message_text = self.pop_message_text[pop_id].format(**kwargs)

        new_line = dict.fromkeys(['type', 'alpha', 'duration', 'start', 'end'], 0.)
        new_line['text'] = message_text
        new_line['duration'] = duration
        new_line['type'] = pop_id

        self.pop_db.append(new_line)
        # print(self.pop_db)

    def pop_message_dispatcher(self, time):
        for item in self.pop_db:
            if item['end'] == 0:
                item['start'] = time
                item['end'] = time + item['duration']
            else:
                item['alpha'] = self.restrict((time - item['start']) * 2, 0, 1) * self.restrict(item['end'] - time, 0,
                                                                                                1)
            if time > item['end']:
                self.pop_db.remove(item)

    def do_auto_flip(self):
        self.mouse_move_cumulative += self.autoflip_speed
        self.run_reduce_flipping_speed = - .15
        if abs(self.mouse_move_cumulative) > 100:
            self.run_flip_once = 1 if self.mouse_move_cumulative > 0 else -1
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

    def virtual_mouse_cursor(self, dx, dy):
        self.virtual_cursor_position += [dx, dy]
        if self.setting_active:
            if abs(self.virtual_cursor_position[1]) > 150:
                self.setting_active += 1 if self.virtual_cursor_position[1] > 0 else -1
                self.setting_active = self.restrict(self.setting_active, 1, len(self.configs))
                self.virtual_cursor_position *= 0
        elif self.levels_open:
            if abs(self.virtual_cursor_position[0]) > 200:
                self.levels_edit_group = 1 if self.virtual_cursor_position[0] > 0 else 0
                self.virtual_cursor_position *= 0
            if abs(self.virtual_cursor_position[1]) > 150:
                self.levels_edit_band += 1 if self.virtual_cursor_position[1] > 0 else -1
                self.levels_edit_band = self.restrict(self.levels_edit_band, 0, 3)
                self.virtual_cursor_position *= 0
        elif self.transform_mode:
            if abs(self.virtual_cursor_position[0]) > 200:
                self.transform_mode_column = 1 if self.virtual_cursor_position[0] > 0 else 0
                self.virtual_cursor_position *= 0
            if abs(self.virtual_cursor_position[1]) > 150:
                self.transform_mode += 1 if self.virtual_cursor_position[1] > 0 else -1
                self.transform_mode = self.restrict(self.transform_mode, 1, 2)
                self.virtual_cursor_position *= 0

    def mouse_circle_tracking(self, dx, dy):
        self.mouse_buffer *= .9
        self.mouse_buffer += (dx, dy)
        mouse_speed = np.linalg.norm(self.mouse_buffer) ** .6
        move_atangent_new = math.atan(self.mouse_buffer[0] / self.mouse_buffer[1])
        move_atangent_delta_new = self.mouse_move_atangent - move_atangent_new
        if abs(move_atangent_delta_new) > .2:
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

    def flip_once(self):
        self.mouse_move_cumulative *= .05

        if self.run_flip_once == 1:
            self.next_image()
        else:
            self.previous_image()
        self.run_flip_once = 0

    def reduce_flipping_speed(self, time_frame):
        if self.run_reduce_flipping_speed < 0:
            self.run_reduce_flipping_speed = time_frame + abs(self.run_reduce_flipping_speed)
        if self.run_reduce_flipping_speed > time_frame:
            return
        self.mouse_move_cumulative -= math.copysign(self.mouse_unflipping_speed, self.mouse_move_cumulative)
        self.mouse_unflipping_speed = self.mouse_unflipping_speed * .8 + .3
        if abs(self.mouse_move_cumulative) < 5:
            self.run_reduce_flipping_speed = 0
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

    def autoflip_toggle(self):
        if self.autoflip_speed == 0:
            self.autoflip_speed = .5
            self.schedule_pop_message(9)
        else:
            self.autoflip_speed = 0
            self.schedule_pop_message(10)

    def update_position(self):
        # for name in self.gl_program_pic[self.program_id]:
        #     member = self.gl_program_pic[self.program_id][name]
        #     print(name, type(member), member)

        texture_size = self.current_texture.size
        self.gl_program_pic[self.program_id]['displacement'] = tuple(self.pic_position)
        self.gl_program_pic[self.program_id]['pix_size'] = texture_size
        self.gl_program_pic[self.program_id]['wnd_size'] = self.wnd.size
        self.gl_program_pic[self.program_id]['zoom_scale'] = self.pic_zoom
        self.gl_program_pic[self.program_id]['angle'] = math.radians(self.pic_angle)
        self.gl_program_pic[self.program_id]['useCurves'] = self.levels_open and self.levels_enabled
        self.gl_program_pic[self.program_id]['count_histograms'] = self.levels_open
        self.gl_program_pic[self.program_id]['show_amount'] = self.transition_stage
        self.gl_program_pic[self.program_id]['transparency'] = 0
        self.gl_program_pic[self.program_id]['hide_borders'] = self.configs[HIDE_BORDERS]
        self.gl_program_pic[self.program_id]['inter_blur'] = self.configs[INTER_BLUR]
        self.gl_program_pic[self.program_id]['transition_center'] = self.transition_center
        self.gl_program_pic[1 - self.program_id]['transparency'] = self.transition_stage

        self.gl_program_pic_v['displacement'] = tuple(self.pic_position)
        self.gl_program_pic_v['pix_size'] = texture_size
        self.gl_program_pic_v['wnd_size'] = self.wnd.size
        self.gl_program_pic_v['zoom_scale'] = self.pic_zoom
        self.gl_program_pic_v['angle'] = math.radians(self.pic_angle)

        if self.transform_mode:
            self.gl_program_pic[self.program_id]['hide_borders'] = 0
            self.gl_program_pic[self.program_id]['crop'] = tuple(self.crop_borders)
            self.gl_program_crop['pix_size'] = texture_size
            self.gl_program_crop['zoom_scale'] = self.pic_zoom
            self.gl_program_crop['displacement'] = tuple(self.pic_position)
            self.gl_program_crop['wnd_size'] = self.wnd.size
            self.gl_program_crop['crop'] = tuple(self.crop_borders)

        self.gl_program_round['wnd_size'] = self.wnd.size
        self.gl_program_round['displacement'] = tuple(self.round_indicator_cener_pos)
        self.gl_program_round['round_size'] = self.round_indicator_radius
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

        self.gl_program_mandel['wnd_size'] = self.wnd.size
        self.gl_program_mandel['pic_position'] = tuple(self.pic_position)
        self.gl_program_mandel['pic_position_precise'] = tuple(self.pic_position_2)

        self.gl_program_mandel['zoom'] = self.pic_zoom
        self.gl_program_mandel['complexity'] = abs(self.pic_angle) / 10

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)
        self.update_position()

    def restrict(self, val, minval, maxval):
        if val < minval: return minval
        if val > maxval: return maxval
        return val

    def change_settings(self, amount):
        if self.setting_active:
            self.configs[self.setting_active] = self.restrict(self.configs[self.setting_active] *
                                                              (1 - amount) - amount / 1000 *
                                                              self.config_formats[self.setting_active][1],
                                                              self.config_formats[self.setting_active][0],
                                                              self.config_formats[self.setting_active][1])
            self.update_position()
            if self.setting_active == INTER_BLUR:
                self.gl_program_pic[self.program_id]['transparency'] = .5
                self.run_move_image_inertial = False

        elif self.levels_open:
            edit_parameter = self.levels_edit_parameter - 1
            if edit_parameter > 2:
                amount = - amount
            if edit_parameter == 2:
                self.levels_borders[self.levels_edit_band][edit_parameter] = self.restrict(
                    self.levels_borders[self.levels_edit_band][edit_parameter] * (1 + amount), 0.01, 10)
            else:
                new_value = self.levels_borders[self.levels_edit_band][edit_parameter] + amount
                self.levels_borders[self.levels_edit_band][edit_parameter] = self.restrict(new_value, 0, 1)
            self.generate_levels_texture(self.levels_edit_band)
            self.update_position()
        elif self.transform_mode == 2:  # cropping
            border_id = self.transform_mode_column * 2 - 1 + self.pressed_mouse & 3
            opposite_border_id = 2 * self.transform_mode_column - 1 + ((self.pressed_mouse & 3) ^ 3)
            # print((self.pressed_mouse & 3), (self.pressed_mouse & 3) ^ 3)
            # print(border_id, opposite_border_id)
            self.crop_borders[border_id] += -amount * 100 * (1 - 2 * (border_id % 2))
            if self.crop_borders[border_id] + self.crop_borders[opposite_border_id] + 1 > self.current_texture.size[
                self.transform_mode_column]:
                self.crop_borders[opposite_border_id] = self.current_texture.size[self.transform_mode_column] - \
                                                        self.crop_borders[border_id] - 1
            self.update_position()

    def mouse_position_event(self, x, y, dx, dy):
        if self.setting_active or (self.levels_open and self.levels_enabled) or self.transform_mode:
            self.virtual_mouse_cursor(dx, dy)
            return
        elif self.autoflip_speed != 0:
            d_coord = dx - dy
            self.autoflip_speed += d_coord / 500 * (1.5 - math.copysign(.5, self.autoflip_speed * d_coord))
        else:
            self.imgui.mouse_position_event(x, y, dx, dy)
            self.mouse_circle_tracking(dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        if self.setting_active or (self.levels_open and self.levels_enabled) or (self.transform_mode > 1):
            self.change_settings((dy * 5 - dx) / 1500)
            return

        # self.imgui.mouse_drag_event(x, y, dx, dy)
        if self.pressed_mouse == 1:
            self.move_image(np.array([dx, -dy], dtype=np.float64))
        elif self.pressed_mouse == 2:
            self.pic_zoom_future *= 1 / (1 + 1.02 ** (- dx + dy)) + .5
            self.move_image()
        elif self.pressed_mouse == 3:
            self.pic_angle_future += (- dx + dy) / 15
            self.move_image()
            self.unschedule_pop_message(2)

    def mouse_scroll_event(self, x_offset, y_offset):
        if self.mandelbrot_mode:
            self.pic_zoom_future *= 1.3 if y_offset < 0 else .7
            self.move_image()
            return
        self.run_flip_once = 1 if y_offset < 0 else -1

    def mouse_press_event(self, x, y, button):
        # self.imgui.mouse_press_event(x, y, button)
        button_code = 4 if button == 3 else button
        self.pressed_mouse = self.pressed_mouse | button_code
        if self.pressed_mouse == 4:
            self.wnd.close()
            return

        if self.levels_open and self.levels_enabled:
            if self.pressed_mouse < 4:
                self.levels_edit_parameter = (self.levels_edit_group * 3 + self.pressed_mouse) % 6

        if self.pressed_mouse == 5:
            self.random_image()
        if self.pressed_mouse == 6:
            self.autoflip_toggle()

    def mouse_release_event(self, x: int, y: int, button: int):
        # self.imgui.mouse_release_event(x, y, button)
        button_code = 4 if button == 3 else button
        self.pressed_mouse = self.pressed_mouse & ~button_code
        if self.levels_edit_parameter > 0:
            self.levels_edit_parameter = 0
        self.run_move_image_inertial = True

    def unicode_char_entered(self, char):
        pass
        # self.imgui.unicode_char_entered(char)

    def key_event(self, key, action, modifiers):
        # self.imgui.key_event(key, action, modifiers)
        if action == self.wnd.keys.ACTION_PRESS:
            self.last_key_press_time = self.timer.time

            if self.setting_active:
                if key == self.wnd.keys.TAB:
                    self.setting_active = (self.setting_active + 1) % 4

            elif self.mandelbrot_mode:
                if key == self.wnd.keys.RIGHT:
                    self.key_picture_movement[3] = True
                    return
                elif key == self.wnd.keys.LEFT:
                    self.key_picture_movement[1] = True
                    return
                elif key == self.wnd.keys.UP:
                    self.key_picture_movement[0] = True
                    return
                elif key == self.wnd.keys.DOWN:
                    self.key_picture_movement[2] = True
                    return
                elif key in [self.wnd.keys.MINUS, 65453]:
                    self.key_picture_movement[4] = True
                    return
                elif key in [61, 65451]:  # +
                    self.key_picture_movement[5] = True
                    return
                # print(key)

            elif self.levels_open:
                if key == 59:  # ;
                    self.levels_enabled = not self.levels_enabled
                    self.update_position()
                if key == self.wnd.keys.P:
                    self.previous_level_borders()
                    return
                if key == self.wnd.keys.O:
                    self.empty_level_borders()
                if key == self.wnd.keys.TAB:
                    self.levels_edit_band = (self.levels_edit_band + 1) % 4
                if key == self.wnd.keys.R:
                    if self.levels_edit_band == 1:
                        self.levels_edit_band = 0
                    else:
                        self.levels_edit_band = 1
                if key == self.wnd.keys.G:
                    if self.levels_edit_band == 2:
                        self.levels_edit_band = 0
                    else:
                        self.levels_edit_band = 2
                if key == self.wnd.keys.B:
                    if self.levels_edit_band == 3:
                        self.levels_edit_band = 0
                    else:
                        self.levels_edit_band = 3
                if key == self.wnd.keys.ENTER:
                    self.apply_levels()
                    return
            elif self.transform_mode:
                if key == self.wnd.keys.ENTER:
                    self.apply_transform()
                    return

            if modifiers.shift:
                if key == self.wnd.keys.RIGHT:
                    self.key_picture_movement[3] = True
                    return
                elif key == self.wnd.keys.LEFT:
                    self.key_picture_movement[1] = True
                    return
                elif key == self.wnd.keys.UP:
                    self.key_picture_movement[0] = True
                    return
                elif key == self.wnd.keys.DOWN:
                    self.key_picture_movement[2] = True
                    return

            if modifiers.ctrl:
                if key == self.wnd.keys.SPACE:
                    self.run_key_flipping = 4
                elif key == self.wnd.keys.RIGHT:
                    self.run_key_flipping = 4
                elif key == self.wnd.keys.LEFT:
                    self.run_key_flipping = -8
                elif key == self.wnd.keys.R:
                    self.current_texture = self.image_texture
                    self.current_texture.use(5)
            else:
                if key == self.wnd.keys.A:
                    self.autoflip_toggle()
                if key == self.wnd.keys.T:
                    self.transform_mode = (self.transform_mode + 1) % 2
                    if self.transform_mode:
                        self.reset_pic_position(False)
                        self.update_position()
                if key == self.wnd.keys.ENTER:
                    self.save_rotation()
                if key == self.wnd.keys.SPACE:
                    self.run_key_flipping = 1
                elif key == self.wnd.keys.RIGHT:
                    self.run_key_flipping = 1
                elif key == self.wnd.keys.LEFT:
                    self.run_key_flipping = -1
                elif key == self.wnd.keys.L:
                    self.show_curves_interface()
                elif key == self.wnd.keys.I:
                    self.show_image_info = (self.show_image_info + 1) % 2
                elif key == self.wnd.keys.C:
                    self.move_file_out(do_copy=True)
                elif key == self.wnd.keys.U:
                    self.load_image(mandelbrot=True)
                elif key in [self.wnd.keys.M, self.wnd.keys.BACKSLASH]:
                    self.move_file_out()
                elif key == self.wnd.keys.PAGE_UP:
                    self.first_directory_image(-1)
                elif key == self.wnd.keys.DOWN:
                    self.first_directory_image(0)
                elif key == self.wnd.keys.PAGE_DOWN:
                    self.first_directory_image(1)
                elif key == self.wnd.keys.UP:
                    self.random_image()
                elif key == self.wnd.keys.COMMA:
                    self.rotate_image_90(True)
                elif key == self.wnd.keys.PERIOD:
                    self.rotate_image_90()
                elif key == self.wnd.keys.H:
                    self.central_message_showing = 0 if self.central_message_showing else 1
                elif key == self.wnd.keys.F1:
                    self.central_message_showing = 0 if self.central_message_showing else 1
                elif key == self.wnd.keys.F4:
                    self.save_list_file()
                elif key == self.wnd.keys.F3:
                    self.save_list_file(compress=False)
                elif key == self.wnd.keys.F5:
                    self.random_image("rand_file")
                elif key == self.wnd.keys.F6:
                    self.random_image("current_dir")
                elif key == self.wnd.keys.F7:
                    self.random_image("rand_dir_first_file")
                elif key == self.wnd.keys.F8:
                    self.random_image("rand_dir_rand_file")
                elif key == self.wnd.keys.F12:
                    self.save_current_texture(True)
                elif key == self.wnd.keys.F9:
                    self.save_current_texture(False)
                elif key == self.wnd.keys.P:
                    self.setting_active = 0 if self.setting_active else 1
                elif key in [self.wnd.keys.MINUS, 65453]:
                    self.key_picture_movement[4] = True
                elif key in [61, 65451]:  # +
                    self.key_picture_movement[5] = True

        elif action == self.wnd.keys.ACTION_RELEASE:
            if self.mandelbrot_mode or modifiers.shift:
                if key == self.wnd.keys.RIGHT:
                    self.key_picture_movement[3] = False
                elif key == self.wnd.keys.LEFT:
                    self.key_picture_movement[1] = False
                elif key == self.wnd.keys.UP:
                    self.key_picture_movement[0] = False
                elif key == self.wnd.keys.DOWN:
                    self.key_picture_movement[2] = False
                return

            if key == self.wnd.keys.SPACE:
                self.run_key_flipping = 0
            elif key == self.wnd.keys.RIGHT:
                self.run_key_flipping = 0
            elif key == self.wnd.keys.LEFT:
                self.run_key_flipping = 0
            elif key in [self.wnd.keys.MINUS, 65453]:
                self.key_picture_movement[4] = False
            elif key in [61, 65451]:  # +
                self.key_picture_movement[5] = False
            elif key == 59:  # [;]
                if self.timer.time - self.last_key_press_time > BUTTON_STICKING_TIME:
                    self.levels_enabled = not self.levels_enabled
                    self.update_position()

    def read_and_clear_histo(self):
        hg = np.frombuffer(self.histo_texture.read(), dtype=np.uint32).reshape(5, 256).copy()
        hg[1] = hg[2] + hg[3] + hg[4]
        self.histogram_array = hg.astype(np.float32)
        self.histo_texture.write(self.histo_texture_empty)

    def render(self, time: float, frame_time: float):
        if self.run_move_image_inertial:
            self.move_image_inertial(time)
        if not (self.run_reduce_flipping_speed == 0):
            self.reduce_flipping_speed(time)
        if not (self.run_flip_once == 0):
            self.flip_once()
        if not (self.run_key_flipping == 0):
            self.key_flipping(time)
        if True in self.key_picture_movement:
            self.move_picture_with_key(frame_time)
        self.pop_message_dispatcher(time)
        if self.autoflip_speed != 0 and self.pressed_mouse == 0:
            self.do_auto_flip()

        self.wnd.clear()
        self.read_and_clear_histo()

        if self.mandelbrot_mode:
            self.picture_vertices4.render(self.gl_program_mandel)
        else:
            if self.transition_stage < 1:
                if type(self.current_texture_old) is moderngl.texture.Texture:
                    self.current_texture_old.use(5)
                    self.picture_vertices4.render(self.gl_program_pic[1 - self.program_id])

            self.current_texture.use(5)
            self.picture_vertices4.render(self.gl_program_pic[self.program_id])
            if self.transform_mode:
                self.crop_vao.render(self.gl_program_crop)
            self.picture_vertices4.transform(self.gl_program_pic_v, self.ret_vertex_buffer,
                                             mode=moderngl.POINTS, vertices=4)

        self.ctx.enable_only(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.round_vao.render(self.gl_program_round)
        self.render_ui(time)

    def format_bytes(self, size):
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

    def render_ui(self, time):
        # io = imgui.get_io()
        # io.config_resize_windows_from_edges = True
        # new_font = io.fonts.add_font_default()
        # io.fonts.build()
        # new_font = io.fonts.add_font_from_file_ttf(
        #     "c:/Soft/PyCharm Community Edition 2020.1.2/jbr/lib/fonts/DroidSans.ttf", 20,
        # )
        # imgui.refresh_font_texture()

        # with imgui.font(new_font):

        # display_size = io.display_size

        # line_height = imgui.get_text_line_height_with_spacing()
        # imgui.set_next_window_position(io.display_size[0] / 4, io.display_size[1] / 4)
        # imgui.set_next_window_position(0, 0)
        # imgui.set_next_window_content_size(io.display_size[0] / 4, line_height * 99)
        # imgui.set_next_window_size(0, line_height * 15 * 0)

        io = imgui.get_io()
        io.ini_file_name = np.empty(0).tobytes()
        imgui.new_frame()
        style = imgui.get_style()

        style.alpha = 1
        line_height = imgui.get_text_line_height_with_spacing()
        in_cenral_wnd_flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | \
                              imgui.WINDOW_NO_INPUTS | imgui.WINDOW_NO_COLLAPSE
        im_gui_window_flags = in_cenral_wnd_flags | imgui.WINDOW_NO_TITLE_BAR

        # Settings window
        if self.setting_active:
            imgui.set_next_window_position(io.display_size.x * .5, io.display_size.y * 0.5, 1, pivot_x=.5, pivot_y=0.5)
            imgui.set_next_window_bg_alpha(.9)
            imgui.begin("Settings", False, in_cenral_wnd_flags)

            for key in self.configs.keys():
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, .2 + .5 * (self.setting_active == key), .2, .2)
                imgui.slider_float(self.config_descriptions[key], self.configs[key], self.config_formats[key][0],
                                   self.config_formats[key][1],
                                   self.config_formats[key][2],
                                   self.config_formats[key][3])
                imgui.pop_style_color()

            imgui.end()

        def add_cells_with_text(texts, list_of_selected):
            for n, text in enumerate(texts):
                letters_blue = .5 if int(n / 2) in list_of_selected else 1
                imgui.push_style_color(imgui.STYLE_ALPHA, 1, 1, letters_blue)
                imgui.text(text)
                imgui.pop_style_color()
                imgui.next_column()

        # Levels window
        if self.levels_open:
            imgui.set_next_window_position(io.display_size.x, io.display_size.y * 0.5, 1, pivot_x=1, pivot_y=0.5)
            imgui.set_next_window_size(0, 0)

            hg_size = (io.display_size.x / 5, (io.display_size.y - 9 * line_height * 2) / 6)

            imgui.set_next_window_bg_alpha(.8)
            imgui.begin("Levels settings", True, im_gui_window_flags)

            hg_names = ["Gray", "RGB", "Red", "Green", "Blue"]
            hg_colors = [[0.8, 0.8, 0.8],
                         [0.7, 0.7, 0.7],
                         [0.7, 0.3, 0.3],
                         [0.3, 0.7, 0.3],
                         [0.3, 0.3, 0.7]]

            for hg_num in range(5):
                bg_color = .3
                if hg_num == self.levels_edit_band + 1:
                    bg_color = .5
                imgui.text(hg_names[hg_num] + " Histogram")
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, bg_color, bg_color, bg_color, bg_color)
                imgui.push_style_color(imgui.COLOR_PLOT_HISTOGRAM, hg_colors[hg_num][0], hg_colors[hg_num][1],
                                       hg_colors[hg_num][2])
                imgui.plot_histogram("", self.histogram_array[hg_num], graph_size=hg_size)
                imgui.pop_style_color(2)

            style.alpha = .2 + .6 * self.levels_enabled

            imgui.columns(3)
            add_cells_with_text(["", "Input", "   Output"], [self.levels_edit_group])

            imgui.columns(6)
            add_cells_with_text(["", " min", " max", "gamma", " min", " max"], [self.levels_edit_group * 2,
                                                                                self.levels_edit_group * 2 + 1])

            def add_grid_elements(row_name, row_number):
                active_columns = [self.levels_edit_group * 3, self.levels_edit_group * 3 + 1,
                                  self.levels_edit_group * 3 + 2]
                letters_blue = 1
                if self.levels_edit_band == row_number:
                    letters_blue = .5
                imgui.push_style_color(imgui.STYLE_ALPHA, 1, 1, letters_blue)
                imgui.text(row_name)
                imgui.next_column()
                for column in range(5):
                    if column == self.levels_edit_parameter - 1 and self.levels_edit_band == row_number:
                        bg_color = (.7, .2, .2)
                    elif column in active_columns and self.levels_edit_band == row_number:
                        bg_color = (.2, .2, .6)
                    else:
                        bg_color = (.2, .2, .2)
                    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, bg_color[0], bg_color[1], bg_color[2])
                    imgui.slider_float("", self.levels_borders[row_number][column], 0, 1, '%.2f', [1, .10][column == 4])
                    imgui.pop_style_color()
                    imgui.next_column()
                imgui.pop_style_color()

            add_grid_elements("RGB", 0)
            add_grid_elements("Red", 1)
            add_grid_elements("Green", 2)
            add_grid_elements("Blue", 3)

            imgui.set_window_font_scale(1)
            imgui.end()

        # imgui.push_font(new_font)
        # imgui.begin("Example: my style editor")
        # imgui.show_style_editor()
        # imgui.end()
        # imgui.pop_font()

        if self.central_message_showing:
            imgui.set_next_window_position(io.display_size.x * 0.5, io.display_size.y * 0.5, 1, pivot_x=0.5,
                                           pivot_y=0.5)
            _, caption, *message = self.central_message[self.central_message_showing].splitlines()
            imgui.set_next_window_size(0, 0)
            imgui.set_next_window_bg_alpha(.8)
            imgui.begin(caption, False, in_cenral_wnd_flags)
            imgui.set_window_font_scale(1.5)
            for text in message:
                imgui.text(text)
            imgui.end()

        next_message_top = 10
        if self.show_image_info > 0:
            style.alpha = .7

            def show_info_window(message_top, text_list, win_name):
                imgui.set_next_window_position(10, message_top, 1, pivot_x=0, pivot_y=0)
                imgui.begin(win_name, True, im_gui_window_flags)
                for text in text_list:
                    imgui.text(text)
                message_top += imgui.get_window_height()
                imgui.end()
                return message_top

            if self.show_image_info == 1:
                if self.mandelbrot_mode:
                    info_text = [
                        "Mandelbrot mode",
                        "Position:",
                        f"  x: {self.gl_program_mandel['pic_position'].value[0] / self.window_size[0]:.17f}",
                        f"  y: {self.gl_program_mandel['pic_position'].value[1] / self.window_size[1]:.17f}",
                        f"Log Zoom: {math.log(self.gl_program_mandel['zoom'].value, 2):.1f}",
                        f"Actual Zoom: {self.gl_program_mandel['zoom'].value:,.1f}",
                        f"Complexity: {self.gl_program_mandel['complexity'].value:.2f}"
                    ]
                else:
                    info_text = ["Folder: " + os.path.dirname(self.get_file_path(self.image_index))]
                    next_message_top = show_info_window(next_message_top, info_text, "Directory")
                    im_mp = self.image_original_size.x * self.image_original_size.y / 1000000
                    dir_index = self.file_to_dir[self.image_index]
                    dirs_in_folder = self.dir_to_file[dir_index][1]
                    index_in_folder = self.image_index + 1 - self.dir_to_file[dir_index][0]
                    info_text = ["File name: " + os.path.basename(self.get_file_path(self.image_index)),
                                 "File size: " + self.format_bytes(self.current_image_file_size),
                                 "Image size: " + f"{self.image_original_size.x} x {self.image_original_size.y}",
                                 "Image size: " + f"{im_mp:.2f} megapixels",
                                 "Image # (current folder): " + f"{index_in_folder:d} of {dirs_in_folder:d}",
                                 "Image # (all list): " + f"{self.image_index + 1:d} of {self.image_count:d}",
                                 "Folder #: " + f"{dir_index + 1:d} of {self.dir_count:d}"]
                next_message_top = show_info_window(next_message_top, info_text, "File props")
            next_message_top += 10

        if self.transform_mode:
            next_message_top_r = 100
            imgui.set_next_window_position(io.display_size.x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)
            imgui.set_next_window_size(0, 0)

            imgui.begin("Image tranformations", True, im_gui_window_flags)
            imgui.text("Image tranformations")
            style.alpha = .8
            imgui.set_window_font_scale(1.6)
            next_message_top_r += imgui.get_window_height()
            imgui.end()

            imgui.set_next_window_position(io.display_size.x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)

            bg_color = 1 if self.transform_mode == 1 else 0.5
            imgui.push_style_color(imgui.COLOR_BORDER, bg_color, bg_color, bg_color, bg_color)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .5, .5, bg_color, bg_color)

            imgui.begin("Navigation", True, im_gui_window_flags)
            # style.alpha = .5
            imgui.text("Navigation")
            imgui.set_window_font_scale(1.2)
            next_message_top_r += imgui.get_window_height()
            imgui.pop_style_color()
            imgui.pop_style_color()
            imgui.end()

            bg_color = 1 if self.transform_mode == 2 else 0.5
            imgui.push_style_color(imgui.COLOR_BORDER, bg_color, bg_color, bg_color, bg_color)
            imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, .2, .2, .2 + .1 * bg_color, bg_color)

            imgui.set_next_window_position(io.display_size.x, next_message_top_r, 1, pivot_x=1, pivot_y=0.0)
            imgui.set_next_window_bg_alpha(.8)
            imgui.begin("Crop image", True, im_gui_window_flags)
            imgui.text("Image cropping")

            style.alpha = .2 + .6 * self.levels_enabled * (self.transform_mode == 2)

            new_image_width = self.current_texture.width - int(self.crop_borders[0]) - int(self.crop_borders[1])
            new_image_height = self.current_texture.width - int(self.crop_borders[2]) - int(self.crop_borders[3])
            imgui.set_window_font_scale(1)
            imgui.text("Original image size: " + f"{self.image_original_size.x} x {self.image_original_size.y}")
            imgui.text("Current image size: " + f"{self.current_texture.width} x {self.current_texture.height}")
            imgui.text("New image size: " + f"{new_image_width} x {new_image_height}")

            imgui.text("")
            imgui.text("Crop from borders")
            imgui.columns(4)
            add_cells_with_text([" Left", " Right", "Bottom", "  Top"], [self.transform_mode_column])
            imgui.pop_style_color()
            imgui.pop_style_color()

            for column in range(4):
                if column // 2 == self.transform_mode_column and self.pressed_mouse & 3 == column % 2 + 1:
                    bg_color = (.7, .2, .2)
                elif column // 2 == self.transform_mode_column:
                    bg_color = (.2, .2, .6)
                else:
                    bg_color = (.2, .2, .2)
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, bg_color[0], bg_color[1], bg_color[2])
                imgui.slider_int("", self.crop_borders[column], 0, self.current_texture.size[column // 2], '%d')
                imgui.pop_style_color()
                imgui.next_column()

            imgui.end()

        if len(self.pop_db) > 0:
            for item in self.pop_db:
                style.alpha = item['alpha'] * .8
                imgui.set_next_window_position(10, next_message_top)
                imgui.begin(str(item['type']), True, im_gui_window_flags)
                imgui.set_window_font_scale(1.2)
                imgui.text(item['text'])
                next_message_top += imgui.get_window_height() * max(item['alpha'] ** .3, item['alpha'])
                imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())


def main_loop() -> None:
    # mglw.setup_basic_logging(20)  # logging.INFO
    window_cls = mglw.get_local_window_cls()

    start_fullscreen = True
    if "-f" in sys.argv:
        start_fullscreen = not start_fullscreen

    window = window_cls(fullscreen=start_fullscreen)
    if start_fullscreen:
        window.mouse_exclusivity = True
    window.print_context_info()
    mglw.activate_context(window=window)
    timer = mglw.timers.clock.Timer()
    window.config = ModernSlideShower(ctx=window.ctx, wnd=window, timer=timer)

    timer.start()

    while not window.is_closing:
        current_time, delta = timer.next_frame()
        window.render(current_time, delta)
        window.swap_buffers()

    _, duration = timer.stop()
    window.destroy()
    if duration > 0:
        mglw.logger.info(
            "Duration: {0:.2f}s @ {1:.2f} FPS".format(
                duration, window.frames / duration
            )
        )


if __name__ == '__main__':
    # main_loop(ModernSlideShower)
    main_loop()
