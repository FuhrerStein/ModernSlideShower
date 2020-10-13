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

# from scipy.interpolate import BSpline


#    todo: smooth image change
#    todo: autoflip mode: repeat flipping with a speed shown by the user
#    todo: save rotation when saving edited image
#    todo: shortcuts written everywhere
#    todo: all operations supported by messages
#    todo: lossless jpeg image cropping
#    todo: filelist position indicator
#    todo: lossy image cropping in full edit mode
#    todo: doubleclick event
#    todo: keyboard navigation on picture
#    todo: show short image info
#    todo: show full image info
#    todo: interface to adjust curves
#    todo: different color curves
#    todo: explain about jpegtan in case it was not found


#    +todo: random image jump from mouse
#    +todo: self.pressed_mouse must support middle button with other buttons
#    +todo: show friendly message in case no images are found
#    +todo: start with random image if playlist had _r in its name
#    +todo: pleasant image centering
#    +todo: show message when changing folder
#    +todo: simple slidelist compression to reduce file size.
#    +todo: make image sharpest possible using right mipmap texture
#    +todo: show message when ending list
#    +todo: update histograms on the fly
#    +todo: save edited image
#    +todo: interface to adjust levels
#    +todo: different color levels
#    +todo: mouse action for left+right drag
#    +todo: help message on F1 or H
#    +todo: free image rotation
#    +todo: rotate indicator
#    +todo: copy/move indicator
#    +todo: work with broken and gray jpegs
#    +todo: keep image in visible area
#    +todo: mouse flip indicator
#    +todo: inertial movement and zooming



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
uniform bool one_by_one;

out vec2 uv0;
out float zoom_frag;

void main() {

    float M_PI = 3.14159265358;
    float pi_angle = angle * M_PI / 180;
    mat2 rotate_mat = mat2(cos(pi_angle), - sin(pi_angle), sin(pi_angle), cos(pi_angle));
    mat2 pix_size_mat = mat2(pix_size.x, 0, 0, pix_size.y);
    mat2 wnd_size_mat = mat2(1 / wnd_size.x, 0, 0, 1 / wnd_size.y);
    gl_Position = vec4((in_position * pix_size_mat * rotate_mat + displacement) * zoom_scale * wnd_size_mat, 0, 1);
//    gl_Position = vec4((in_position * pix_size_mat * rotate_mat + displacement) * zoom_scale * wnd_size_mat, in_position.y, 1.5 - in_position.y / 2);
    uv0 = in_texcoord;
    zoom_frag = zoom_scale;

    if (one_by_one)
    {
        gl_Position = vec4(in_position.x, -in_position.y, 1, 1);
    }

}


#elif defined FRAGMENT_SHADER


layout(binding=5) uniform sampler2D texture0;
layout(binding=6) uniform sampler2D texture_curve;
//layout(binding=6) uniform mat(256, 5) texture_curve;
layout(binding=7, r32ui) uniform uimage2D histogram_texture;
uniform bool useCurves;
uniform bool count_histograms;

out vec4 fragColor;
in vec2 uv0;
in float zoom_frag;

void main() {
//    vec4 tempColor = texture(texture0, uv0);
    vec4 tempColor = textureLod(texture0, uv0, - log(zoom_frag));

    if(useCurves)
    {
        // local by-color curves
        tempColor.r = texelFetch(texture_curve, ivec2((tempColor.r * 255 + .5), 1), 0).r;
        tempColor.g = texelFetch(texture_curve, ivec2((tempColor.g * 255 + .5), 2), 0).r;
        tempColor.b = texelFetch(texture_curve, ivec2((tempColor.b * 255 + .5), 3), 0).r;

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
        imageAtomicAdd(histogram_texture, ivec2(tempColor.r * 255 + .5, 2), 1u);
        imageAtomicAdd(histogram_texture, ivec2(tempColor.g * 255 + .5, 3), 1u);
        imageAtomicAdd(histogram_texture, ivec2(tempColor.b * 255 + .5, 4), 1u);
    }

    fragColor = tempColor;
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

    picture_vertices = moderngl.VertexArray
    round_vao = moderngl.VertexArray
    save_folder = ".\\SaveFolder\\"
    image_file_types = ('jpg', 'png', 'jpeg', 'gif', 'tif', 'tiff')
    list_file_type = 'sldlist'
    jpegtran_exe = "c:\\Soft\\Programs\\libjpeg-turbo64\\bin\\jpegtran.exe"
    jpegtran_options = ' -optimize -rotate {0} -trim -copy all -outfile "{1}" "{1}"'
    image_list = []
    image_count = 0
    image_index = 0
    new_image_index = 0
    common_path = ""
    im_object = Image
    # clock = pyglet.clock
    # flip_clock = pyglet.clock
    # inertial_clock = pyglet.clock
    halt_mouse_processing = False
    pic_position = np.array([0., 0.])
    pic_position_future = np.array([0., 0.])
    pic_position_speed = np.array([0., 0.])
    pic_zoom = 1.
    pic_zoom_future = 1.
    pic_angle = 0.
    pic_angle_future = 0.
    gl_program = moderngl.program
    gl_program_round = moderngl.program
    gl_program_histo = moderngl.program
    image_texture = moderngl.Texture
    current_texture = moderngl.Texture
    curve_texture = moderngl.TextureArray
    use_curves = False
    levels_enabled = True
    max_keyboard_flip_speed = .3
    mouse_buffer = np.array([0., 0.])
    mouse_move_atangent = 0.
    mouse_move_atangent_delta = 0.
    mouse_move_cumulative = 0.
    mouse_unflipping_speed = 1.
    round_indicator_cener_pos = 70, 70
    round_indicator_radius = 40
    last_image_folder = None

    resource_dir = ('.')
    autoflip_speed = 0.

    run_move_image_inertial = False
    run_reduce_flipping_speed = 0.
    run_key_flipping = 0
    key_flipping_next_time = 0.
    run_flip_once = 0
    pressed_mouse = 0

    pop_message_timeout = 3.
    pop_message_type = 0
    pop_message_text = ["File moved",
                        "File copied",
                        "Image rotated. \nPress Enter to save",
                        "Rotation saved losslessly",
                        "Levels correction applied.",
                        "File saved with overwrite.",
                        "File saved with suffix.",
                        "Image list start",
                        "Next folder",
                        "Autoflipping ON",
                        "Autoflipping OFF",]
    pop_message_deadline = 0.

    histogram_array = np.empty
    histo_texture = moderngl.texture
    histo_texture_empty = moderngl.buffer

    levels_borders = []
    levels_borders_previous = []

    levels_edit_band = 0
    levels_array = np.zeros((4, 256), dtype='uint8')

    mouse_direct_edit = 0
    empty_image_list = "Empty.jpg"


    central_message_showing = False
    central_message = ["",
                       '''
    Usage shortcuts
    
    [F1], [H]: show/hide this help window
    [,] [.] : rotate image left and right by 90°
    [M] : move image file out of folder tree into '--' subfolder
    [C] : copy image file out of folder tree into '++' subfolder
    [Space], [right arrow] : show next image
    [left arrow] : show previous image
    [move mouse in circles clockwise] : show next image
    [move mouse in circles counterclockwise] : show previous image   
    [arrow up] : show random image
    ''',
                       '''
    No images found
    
    No loadable images found
    You may pass a directory as a first argument to this 
    script, or put this script in a directory with images.
    Press H or F1 to close this window and to show help message.
    '''
                       ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)
        self.init_program()

    def init_program(self):
        self.picture_vertices = mglw.opengl.vao.VAO("main_image")
        point_coord = np.array([-1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, 1], dtype=np.float32)
        self.picture_vertices.buffer(point_coord, '2f', ['in_position'])
        texture_coord = np.array([0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0], dtype=np.float32)
        self.picture_vertices.buffer(texture_coord, '2f', ['in_texcoord'])
        # self.gl_program = self.load_program('picture.glsl')
        # self.gl_program_round = self.load_program('round.glsl')

        picture_program_text = picture_glsl
        round_program_text = round_glsl
        if os.path.isfile('picture.glsl'):
            picture_program_text = open('picture.glsl', 'r').read()
        if os.path.isfile('round.glsl'):
            round_program_text = open('round.glsl', 'r').read()

        dummy_program_description = moderngl_window.meta.ProgramDescription()
        shaders = program.ProgramShaders.from_single(dummy_program_description, picture_program_text)
        self.gl_program = shaders.create()
        shaders = program.ProgramShaders.from_single(dummy_program_description, round_program_text)
        self.gl_program_round = shaders.create()

        self.histo_texture = self.ctx.texture((256, 5), 4)
        self.histo_texture.bind_to_image(7, read=True, write=True)
        self.histo_texture_empty = self.ctx.buffer(reserve=(256 * 5 * 4))
        self.histogram_array = np.zeros((5, 256), dtype=np.float32)
        self.generate_interface_geometry()
        self.empty_level_borders()
        self.empty_level_borders()
        self.generate_levels_texture()

        self.find_jpegtran()
        self.get_images()

        if self.image_count == 0:
            self.central_message_showing = 2
            self.image_list.append(self.empty_image_list)
            self.image_count = 1

        self.find_common_path()
        if self.start_with_random_image:
            self.random_image()
        else:
            self.load_image()
            self.schedule_pop_message(0)

    def previous_level_borders(self):
        self.levels_borders = self.levels_borders_previous
        self.generate_levels_texture()

    def empty_level_borders(self):
        self.levels_borders_previous = self.levels_borders
        self.levels_borders = [[1., 0., 1., 0., 1.].copy() for _ in range(5)]
        self.generate_levels_texture()

    def generate_interface_geometry(self):
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
        curve_array = (curve_array - self.levels_borders[band][1]) / (
                self.levels_borders[band][2] - self.levels_borders[band][1])
        curve_array = np.clip(curve_array, 0, 1)
        curve_array = curve_array ** self.levels_borders[band][0]
        curve_array = curve_array * (self.levels_borders[band][4] - self.levels_borders[band][3]) + \
                      self.levels_borders[band][3]
        curve_array = np.clip(curve_array * 255, 0, 255)
        self.levels_array[band] = np.around(curve_array).astype('uint8')

        self.curve_texture = self.ctx.texture((256, 4), 1, self.levels_array)
        self.curve_texture.use(location=6)
        self.curve_texture.repeat_x = False

    def find_jpegtran(self):
        if not os.path.isfile(self.jpegtran_exe):
            if os.path.isfile(".\\jpegtran.exe"):
                self.jpegtran_exe = os.path.abspath(".\\jpegtran.exe")
            else:
                self.jpegtran_exe = None

    def get_images(self):
        file_arguments = []
        dir_arguments = []
        if len(sys.argv) > 1:
            for argument in sys.argv[1:]:
                if os.path.isdir(argument):
                    dir_arguments.append(os.path.abspath(argument))
                if os.path.isfile(argument):
                    file_arguments.append(os.path.abspath(argument))

            if len(dir_arguments):
                [self.scan_directory(directory) for directory in dir_arguments]
            if len(file_arguments):
                if len(dir_arguments) == 0 and len(file_arguments) == 1:
                    if file_arguments[0].lower().endswith(self.image_file_types):
                        self.scan_directory(os.path.dirname(file_arguments[0]), file_arguments[0])
                    else:
                        self.read_list_file(file_arguments[0])
                else:
                    [self.read_list_file(file) for file in file_arguments]
        else:
            self.scan_directory(os.path.abspath('.\\'))

        image_count = len(self.image_list)
        print(image_count, "total images found")
        self.image_count = image_count

    def find_common_path(self):
        if self.image_count > 10000:
            self.common_path = os.path.commonpath(self.image_list[::self.image_count // 1000])
        else:
            self.common_path = os.path.commonpath(self.image_list)
        parent_path = os.path.dirname(self.image_list[0])
        if self.common_path == parent_path:
            self.common_path = os.path.dirname(self.common_path)

    def scan_directory(self, dirname, look_for_file=None):
        print("Searching for images in", dirname)
        for root, dirs, files in os.walk(dirname):
            for f in files:
                if f.lower().endswith(self.image_file_types):
                    img_path = os.path.join(root, f)
                    self.image_count += 1
                    if not self.image_count % 1000:
                        print(self.image_count, "images found", end="\r")
                    self.image_list.append(img_path)
                    if look_for_file:
                        if self.image_list[-1] == look_for_file:
                            self.new_image_index = len(self.image_list) - 1

    def read_list_file(self, filename):
        if filename.lower().endswith(self.image_file_types):
            self.image_count += 1
            if not self.image_count % 1000:
                print(self.image_count, "images found", end="\r")
            self.image_list.append(os.path.abspath(filename))
        elif filename.lower().endswith(self.list_file_type):
            print("Opening list", filename)
            with open(filename, 'r', encoding='utf-8') as file_handle:
                loaded_list = [current_place.rstrip() for current_place in file_handle.readlines()]

            print("Image list decompression")
            decomressed_list = []
            last_entry = ""
            # common_symbols = 0
            for line in loaded_list:
                if line[0] == ":":
                    common_symbols = int(line[1:5])
                    new_line = last_entry[:common_symbols] + line[6:]
                else:
                    new_line = line
                last_entry = new_line
                decomressed_list.append(new_line)

            self.image_list += decomressed_list
            if "_r" in os.path.basename(filename):
                self.start_with_random_image = True

    def save_list_file(self, compress=False):
        if not os.path.isdir(self.save_folder):
            try:
                os.makedirs(self.save_folder)
            except Exception as e:
                print("Could not create folder ", e)

        suffix = ""
        if compress:
            compressed_list = []
            last_entry = ""
            suffix = "_c"
            for line in self.image_list:
                common_symbols = 0
                for i in range(len(line)):
                    if line[:i] != last_entry[:i]:
                        break
                    common_symbols = i
                if common_symbols > 5:
                    new_line = f":{common_symbols:04d} " + line[common_symbols:]
                else:
                    new_line = line
                compressed_list.append(new_line)
                last_entry = line
        else:
            compressed_list = self.image_list

        new_list_file_name = self.save_folder + 'list_' + datetime.datetime.now().strftime(
            "%Y-%m-%d_%H.%M.%S") + suffix + '.' + self.list_file_type
        with open(new_list_file_name, 'w', encoding='utf-8') as file_handle:
            file_handle.writelines("{}\n".format(file_path) for file_path in compressed_list)

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

    def generate_dummy_image(self):
        bands = []
        for band in range(3):
            bands.append(Image.effect_mandelbrot((1920, 1670), (-1.8, -1, .5, 1), 30 + band * 10))
        dummy_image = Image.merge("RGB", bands)
        return dummy_image

    def load_image(self):
        def release_texture(texture):
            if type(texture) is moderngl.texture.Texture:
                try:
                    texture.release()
                except Exception as ex:
                    pass

        if not os.path.isfile(self.image_list[self.new_image_index]):
            if self.image_list[self.new_image_index] != self.empty_image_list:
                self.find_next_existing_image()
        image_path = self.image_list[self.new_image_index]
        image_texture_old = self.image_texture
        current_texture_old = self.current_texture
        try:
            with Image.open(image_path) as img_buffer:
                if img_buffer.mode == "RGB":
                    self.im_object = self.reorient_image(img_buffer)
                else:
                    self.im_object = self.reorient_image(img_buffer).convert(mode="RGB")
                image_bytes = self.im_object.tobytes()
        except Exception as e:
            if self.image_list[self.new_image_index] == self.empty_image_list:
                self.im_object = self.generate_dummy_image()
                image_bytes = self.im_object.tobytes()
            else:
                print("Error reading ", self.image_list[self.new_image_index], e)
                return

        self.image_texture = self.ctx.texture((self.im_object.width, self.im_object.height), 3, image_bytes)
        self.image_texture.build_mipmaps()
        self.image_texture.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self.image_texture.use(5)
        self.current_texture = self.image_texture
        self.image_index = self.new_image_index

        release_texture(image_texture_old)
        release_texture(current_texture_old)

        self.wnd.title = "ModernSlideShower: " + image_path
        self.reset_pic_position()
        if self.pop_message_type == 3:
            self.schedule_pop_message(0)
        if self.last_image_folder is not None:
            if os.path.dirname(self.image_list[self.image_index]) != self.last_image_folder:
                self.schedule_pop_message(9)
            self.last_image_folder = None
        if self.image_index == 0:
            self.schedule_pop_message(8)

    def find_next_existing_image(self):
        start_number = self.new_image_index
        increment = 1
        if self.image_index - self.new_image_index == 1:
            increment = -1
        while True:
            self.new_image_index = (self.new_image_index + increment) % self.image_count
            if start_number == self.new_image_index:
                # todo good message here
                print("No valid images found. Exiting.")
                self.wnd.close()
                break
            if os.path.isfile(self.image_list[self.new_image_index]):
                return

    def move_file_out(self, do_copy=False):
        parent_folder = os.path.dirname(self.image_list[self.image_index])
        own_subfolder = parent_folder[len(self.common_path):]
        new_folder = self.common_path + ["\\--", "\\++"][do_copy] + own_subfolder
        file_operation = [shutil.move, shutil.copy][do_copy]

        if not os.path.isdir(new_folder):
            try:
                os.makedirs(new_folder)
            except Exception as e:
                print("Could not create folder", e)

        try:
            file_operation(self.image_list[self.image_index], new_folder)
        except Exception as e:
            # todo good message here
            print("Could not complete file " + ["move", "copy"][do_copy], e)
            return

        if not do_copy:
            self.image_list.pop(self.image_index)
            self.image_count = len(self.image_list)
            if not self.image_index < self.image_count:
                self.new_image_index = 0
            self.schedule_pop_message(1)
            self.load_image()
        else:
            self.schedule_pop_message(2)

    def save_rotation(self):
        if self.pic_angle_future % 360 and self.jpegtran_exe:
            rotate_command = self.jpegtran_exe + self.jpegtran_options.format(round(360 - self.pic_angle_future % 360),
                                                                              self.image_list[self.image_index])
            # print(rotate_command)

            os.system(rotate_command)

            self.load_image()
            self.schedule_pop_message(4)

    def rotate_image_90(self, left=False):
        remainder = self.pic_angle_future % 90
        self.pic_angle_future = round(self.pic_angle_future - remainder + 90 * (left - (not left) * (remainder == 0)))
        if self.pic_angle_future % 180:
            self.pic_zoom_future = min(self.wnd.height / self.current_texture.width,
                                       self.wnd.width / self.current_texture.height)
        else:
            self.pic_zoom_future = min(self.wnd.width / self.current_texture.width,
                                       self.wnd.height / self.current_texture.height)
        self.move_image()

        if self.pic_angle_future % 360:  # not at original angle
            if not (self.pic_angle_future % 90):  # but at 90°-ish
                self.schedule_pop_message(3)
        else:
            self.schedule_pop_message(0)

    def check_if_image_in_window(self):
        rad_angle = math.radians(self.pic_angle)
        rotation_matrix = np.array(((np.cos(rad_angle), -np.sin(rad_angle)),
                                    (np.sin(rad_angle), np.cos(rad_angle))))

        rotated_displacement = self.pic_position.dot(rotation_matrix)
        correction_vector = np.array([0., 0.])
        if rotated_displacement[0] > self.current_texture.width:
            correction_vector[0] -= rotated_displacement[0] - self.current_texture.width
        if rotated_displacement[0] + self.current_texture.width < 0:
            correction_vector[0] -= rotated_displacement[0] + self.current_texture.width
        if rotated_displacement[1] > self.current_texture.height:
            correction_vector[1] -= rotated_displacement[1] - self.current_texture.height
        if rotated_displacement[1] + self.current_texture.height < 0:
            correction_vector[1] -= rotated_displacement[1] + self.current_texture.height
        correction_vector = correction_vector.dot(rotation_matrix.T)
        return correction_vector, abs(np.linalg.norm(correction_vector))

    def move_image_inertial(self, dt):
        correction_vector, its_size = self.check_if_image_in_window()
        if its_size != 0:
            self.pic_position_future += correction_vector / 10

        self.pic_position_future += self.pic_position_speed
        self.pic_position_speed *= .95
        distance_to_ghost = np.log(1 + np.linalg.norm(self.pic_position - self.pic_position_future))
        abs_speed = np.log(1 + np.linalg.norm(self.pic_position_speed))
        sp_sum = distance_to_ghost + abs_speed
        sp_sum2 = .5 + .5 / (math.copysign(1, sp_sum) + 1 / (sp_sum + .01))
        self.pic_position = self.pic_position * (1 - sp_sum2) + self.pic_position_future * sp_sum2

        scale_disproportion = abs(self.pic_zoom_future / self.pic_zoom - 1) ** .7 * 2
        pic_zoom_new = self.pic_zoom * (1 - .1 * scale_disproportion) + self.pic_zoom_future * .1 * scale_disproportion
        if pic_zoom_new / self.pic_zoom < 1:
            centerting_factor = 1 - scale_disproportion / self.pic_zoom / 100
            self.pic_position_future = self.pic_position_future * centerting_factor
            if self.use_curves:
                pass
                self.pic_position_future -= np.array((self.wnd.width / 5, 0)) * (1 - centerting_factor)

        self.pic_zoom = pic_zoom_new

        rotation_disproportion = abs(self.pic_angle_future - self.pic_angle) / 10
        self.pic_angle = self.pic_angle * (1 - .2) + self.pic_angle_future * .2

        self.update_position()

        if sp_sum + scale_disproportion * 5 + rotation_disproportion < .1:
            self.run_move_image_inertial = False

    def move_image(self, d_coord=(0., 0.)):
        self.pic_position_future += d_coord
        self.pic_position_speed += d_coord
        self.pic_position_speed *= .97
        self.run_move_image_inertial = True

    def key_flipping(self, time):
        if self.key_flipping_next_time > time:
            return
        if self.run_key_flipping > 0:
            self.next_image()
        else:
            self.previous_image()

        self.key_flipping_next_time = time + .4 / abs(self.run_key_flipping)

    def first_directory_image(self, direction=0):
        current_pix_dir = os.path.dirname(self.image_list[self.new_image_index])
        previous_pix_dir = current_pix_dir
        previous_image_index = self.new_image_index
        increment = 1 if direction == 1 else -1
        # if direction == 1:
        #     increment = 1
        while previous_pix_dir == current_pix_dir:
            self.new_image_index = previous_image_index
            if (self.new_image_index + direction) % self.image_count == 0:
                break
            previous_image_index = (self.new_image_index + increment) % self.image_count
            previous_pix_dir = os.path.dirname(self.image_list[previous_image_index])

        if direction == -1:
            self.new_image_index = (self.new_image_index - 1) % self.image_count
            self.first_directory_image()
            return
        elif direction == 1:
            self.new_image_index = (self.new_image_index + 1) % self.image_count
        self.load_image()

    def next_image(self):
        self.new_image_index = (self.image_index + 1) % self.image_count
        self.last_image_folder = os.path.dirname(self.image_list[self.image_index])
        self.load_image()

    def previous_image(self):
        self.new_image_index = (self.image_index - 1) % self.image_count
        self.load_image()

    def random_image(self):
        self.new_image_index = int(self.image_count * random.random())
        self.load_image()

    def first_image(self):
        self.new_image_index = 0
        self.load_image()

    def apply_levels(self):
        new_texture = self.ctx.texture(self.current_texture.size, 3)
        render_framebuffer = self.ctx.framebuffer([new_texture])
        render_framebuffer.clear()
        render_framebuffer.use()
        self.gl_program['one_by_one'] = True
        self.picture_vertices.render(self.gl_program)
        self.gl_program['one_by_one'] = False
        self.ctx.screen.use()
        new_texture.use(5)
        self.current_texture = new_texture
        self.use_curves = False
        self.levels_edit_band = 0
        self.empty_level_borders()
        self.generate_levels_texture()
        self.update_position()
        self.schedule_pop_message(5)

    def save_current_texture(self, replace):
        texture_data = self.current_texture.read()
        new_image = Image.frombuffer("RGB", self.current_texture.size, texture_data)
        new_file_name = self.image_list[self.image_index]
        if not replace:
            stripped_file_name = os.path.splitext(new_file_name)[0]
            new_file_name = stripped_file_name + "_e" + ".jpg"
        orig_exif = self.im_object.getexif()
        print("Saving under name", new_file_name)
        new_image.save(new_file_name, quality=90, exif=orig_exif, optimize=True)
        if replace:
            self.schedule_pop_message(6)
        else:
            self.image_list.insert(self.image_index + 1, new_file_name)
            self.image_count = len(self.image_list)
            self.schedule_pop_message(7)

    def show_curves_interface(self):
        self.use_curves = not self.use_curves
        self.update_position()

    def reset_pic_position(self):
        wnd_width, wnd_height = self.wnd.size
        self.pic_zoom_future = min(wnd_width / self.current_texture.width, wnd_height / self.current_texture.height)
        self.pic_zoom = self.pic_zoom_future
        self.pic_position = np.array([0., 0.])
        self.pic_position_future = np.array([0., 0.])
        self.pic_position_speed = np.array([0., 0.])
        self.pic_angle = 0.
        self.pic_angle_future = 0.

        self.update_position()

    def schedule_pop_message(self, id):
        self.pop_message_type = id
        self.pop_message_deadline = 0

    def pop_message_dispatcher(self, time_frame):
        if self.pop_message_deadline == 0:
            self.pop_message_deadline = time_frame + self.pop_message_timeout

        if self.pop_message_type == 3:
            self.pop_message_deadline = time_frame + self.pop_message_timeout

        if self.pop_message_deadline < time_frame:
            self.pop_message_deadline = 0
            self.pop_message_type = 0

    def do_auto_flip(self):
        self.mouse_move_cumulative += self.autoflip_speed
        # self.mouse_move_cumulative *= .99
        self.run_reduce_flipping_speed = - .15
        if abs(self.mouse_move_cumulative) > 100:
            self.mouse_move_cumulative *= .05
            self.run_flip_once = 1 if self.mouse_move_cumulative > 0 else -1
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

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
        if self.halt_mouse_processing:
            return
        mouse_move_delta = self.mouse_move_atangent_delta * (4 - math.copysign(2, self.mouse_move_atangent_delta * self.mouse_move_cumulative)) * mouse_speed

        if self.autoflip_speed != 0:
            self.autoflip_speed += mouse_move_delta * .01
        else:
            self.mouse_move_cumulative += mouse_move_delta
            self.mouse_move_cumulative *= .999
        self.run_reduce_flipping_speed = - .15
        if abs(self.mouse_move_cumulative) > 100:
            self.mouse_move_cumulative *= .05
            self.run_flip_once = 1 if self.mouse_move_cumulative > 0 else -1
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

    def flip_once(self, forward):
        if forward == 1:
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
            self.schedule_pop_message(10)
        else:
            self.autoflip_speed = 0
            self.schedule_pop_message(11)

    def update_position(self):
        texture_size = self.current_texture.size
        self.gl_program['pix_size'] = texture_size
        self.gl_program['angle'] = self.pic_angle
        self.gl_program['zoom_scale'] = self.pic_zoom
        self.gl_program['useCurves'] = self.use_curves and self.levels_enabled
        self.gl_program['count_histograms'] = self.use_curves
        self.gl_program['displacement'] = tuple(self.pic_position)
        self.gl_program['wnd_size'] = self.wnd.size
        self.gl_program_round['wnd_size'] = self.wnd.size
        self.gl_program_round['displacement'] = tuple(self.round_indicator_cener_pos)
        self.gl_program_round['round_size'] = self.round_indicator_radius
        self.gl_program_round['finish_n'] = self.mouse_move_cumulative

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)
        self.update_position()

    def restrict(self, val, minval, maxval):
        if val < minval: return minval
        if val > maxval: return maxval
        return val

    def change_levels(self, amount):
        if self.mouse_direct_edit == 3 and self.pressed_mouse == 0:
            return
        if self.pressed_mouse > 2:
            return

        edit_parameter = self.mouse_direct_edit + self.pressed_mouse - 1
        if edit_parameter > 2:
            amount = - amount

        if edit_parameter == 0:
            self.levels_borders[self.levels_edit_band][edit_parameter] = self.restrict(
                self.levels_borders[self.levels_edit_band][edit_parameter] * (1 + amount), 0.01, 10)
        else:
            new_value = self.levels_borders[self.levels_edit_band][edit_parameter] + amount
            self.levels_borders[self.levels_edit_band][edit_parameter] = self.restrict(new_value, 0, 1)
        # print(self.levels_edit_band)

        self.generate_levels_texture(self.levels_edit_band)
        self.update_position()

    def mouse_position_event(self, x, y, dx, dy):
        if self.mouse_direct_edit:
            self.change_levels((dy - dx) / 1500)
        elif self.autoflip_speed != 0:
            d_coord = dx - dy
            self.autoflip_speed += d_coord / 500 * (2 - math.copysign(1, self.autoflip_speed * d_coord))
        else:
            self.imgui.mouse_position_event(x, y, dx, dy)
            self.mouse_circle_tracking(dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        if self.mouse_direct_edit:
            self.change_levels((dy - dx) / 1500)
            return
        # self.imgui.mouse_drag_event(x, y, dx, dy)
        if self.pressed_mouse == 1:
            self.move_image(np.array([dx, -dy]) / self.pic_zoom)
        elif self.pressed_mouse == 2:
            self.pic_zoom_future *= 1 / (1 + 1.02 ** (- dx + dy)) + .5
            self.move_image()
        elif self.pressed_mouse == 3:
            self.pic_angle_future += (- dx + dy) / 15
            self.move_image()
            if self.pop_message_type == 3:
                self.schedule_pop_message(0)

    def mouse_scroll_event(self, x_offset, y_offset):
        if y_offset > 0:
            self.previous_image()
        elif y_offset < 0:
            self.next_image()
        # self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        # self.imgui.mouse_press_event(x, y, button)
        button_code = 4 if button == 3 else button
        self.pressed_mouse = self.pressed_mouse | button_code
        if self.pressed_mouse == 4:
            self.wnd.close()
        if self.pressed_mouse == 5:
            self.random_image()
        if self.pressed_mouse == 6:
            self.autoflip_toggle()

    def mouse_release_event(self, x: int, y: int, button: int):
        # self.imgui.mouse_release_event(x, y, button)
        button_code = 4 if button == 3 else button
        self.pressed_mouse = self.pressed_mouse & ~button_code
        # print(self.pressed_mouse)

    def unicode_char_entered(self, char):
        pass
        # self.imgui.unicode_char_entered(char)

    def key_event(self, key, action, modifiers):
        # self.imgui.key_event(key, action, modifiers)
        if action == self.wnd.keys.ACTION_PRESS:
            if modifiers.ctrl:
                if key == self.wnd.keys.SPACE:
                    self.run_key_flipping = 4
                elif key == self.wnd.keys.RIGHT:
                    self.run_key_flipping = 4
                elif key == self.wnd.keys.LEFT:
                    self.run_key_flipping = -8
                elif key == self.wnd.keys.S:
                    self.save_list_file(True)
                elif key == self.wnd.keys.R:
                    self.current_texture = self.image_texture
                    self.current_texture.use(5)
            else:
                if key == self.wnd.keys.A:
                    self.autoflip_toggle()
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
                elif key == self.wnd.keys.C:
                    self.move_file_out(do_copy=True)
                elif key == self.wnd.keys.M:
                    self.move_file_out()
                elif key == self.wnd.keys.PAGE_UP:
                    self.first_directory_image(-1)
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
                elif key == self.wnd.keys.F12:
                    self.save_current_texture(True)
                elif key == self.wnd.keys.F1:
                    self.central_message_showing = 0 if self.central_message_showing else 1
                elif key == self.wnd.keys.S:
                    self.save_current_texture(modifiers.shift)

            if self.use_curves:
                # print(key)
                if modifiers.shift:
                    pass
                else:
                    if key == 59:  # ;
                        self.levels_enabled = not self.levels_enabled
                        self.update_position()
                    if key == self.wnd.keys.BACKSLASH:  # '
                        self.mouse_direct_edit = 3
                    if key == 39:  # '
                        self.mouse_direct_edit = 1
                    if key == self.wnd.keys.P:
                        self.previous_level_borders()
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
                    # if key == self.wnd.keys.Y:
                    #     # print(self.histo_texture.read())
                    #     hg = np.frombuffer(self.histo_texture.read(), dtype=np.uint32)
                    #     hg1 = np.array(hg / hg.max(), dtype=np.float32)
                    #     # hg /= hg.max
                    #     self.histogram_array[0] = hg1
                    #     with np.printoptions(precision=3, suppress=True):
                    #         print(hg)
                    if key == self.wnd.keys.ENTER:
                        self.apply_levels()

            if key == self.wnd.keys.Z and modifiers.shift:
                pass
        elif action == self.wnd.keys.ACTION_RELEASE:
            if key == self.wnd.keys.SPACE:
                self.run_key_flipping = 0
            elif key == self.wnd.keys.RIGHT:
                self.run_key_flipping = 0
            elif key == self.wnd.keys.LEFT:
                self.run_key_flipping = 0
            elif key in [39, self.wnd.keys.BACKSLASH]:
                self.mouse_direct_edit = 0

    def read_and_clear_histo(self):
        hg = np.frombuffer(self.histo_texture.read(), dtype=np.uint32).reshape(5, 256).copy()
        hg[1] = hg[2] + hg[3] + hg[4]
        self.histogram_array = hg.astype(np.float32)
        self.histo_texture.write(self.histo_texture_empty)

    def render(self, time: float, frame_time: float):
        self.wnd.clear()
        self.read_and_clear_histo()

        self.gl_program['angle'] = self.pic_angle
        self.picture_vertices.render(self.gl_program)

        # self.gl_program['angle'] = 0
        # self.picture_vertices.render(self.gl_program)

        self.ctx.enable_only(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.round_vao.render(self.gl_program_round)
        self.render_ui(time)

        if self.run_move_image_inertial:
            self.move_image_inertial(time)
        if not (self.run_reduce_flipping_speed == 0):
            self.reduce_flipping_speed(time)
        if not (self.run_flip_once == 0):
            self.flip_once(self.run_flip_once)
        if not (self.run_key_flipping == 0):
            self.key_flipping(time)
        if self.pop_message_type > 0:
            self.pop_message_dispatcher(time)
        if self.autoflip_speed != 0 and self.pressed_mouse == 0:
            self.do_auto_flip()

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
        im_gui_window_flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_MOVE
        in_cenral_wnd_flags = imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_ALWAYS_AUTO_RESIZE | \
                              imgui.WINDOW_NO_INPUTS | imgui.WINDOW_NO_COLLAPSE

        # print(6)
        if self.use_curves:
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

            def add_cells_with_text(texts):
                for text in texts:
                    imgui.text(text)
                    imgui.next_column()

            imgui.columns(3)
            add_cells_with_text(["    Gamma", "   Input", "   Output"])

            imgui.columns(6)
            add_cells_with_text(["", "", " min", " max", " min", " max"])

            def add_grid_elements(row_name, row_number):
                active_column = self.mouse_direct_edit + self.pressed_mouse - 1
                if (self.mouse_direct_edit == 3 and self.pressed_mouse == 0) or self.mouse_direct_edit == 0:
                    active_column = - 1
                bg_color = (.2, .2, .2)
                letters_blue = 1
                if self.levels_edit_band == row_number:
                    bg_color = (.5, .3, .3)
                    letters_blue = .5
                imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, bg_color[0], bg_color[1], bg_color[2])
                imgui.push_style_color(imgui.STYLE_ALPHA, 1, 1, letters_blue)
                imgui.text(row_name)
                imgui.next_column()
                for column in range(5):
                    if column == active_column and self.levels_edit_band == row_number:
                        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, .7, .2, .2)
                    imgui.slider_float("", self.levels_borders[row_number][column], 0, 1, '%.2f', [1, .10][column == 4])
                    if column == active_column and self.levels_edit_band == row_number:
                        imgui.pop_style_color(1)
                    imgui.next_column()
                imgui.pop_style_color(2)

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

        if self.pop_message_type > 0:
            style.alpha = np.clip(self.pop_message_deadline - time, 0, 1.)
            imgui.set_next_window_position(30, 30)

            imgui.begin("C" * len(self.pop_message_text[self.pop_message_type - 1]), True, im_gui_window_flags)
            imgui.set_window_font_scale(2)

            imgui.text(self.pop_message_text[self.pop_message_type - 1])
            imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())


def main_loop() -> None:
    # mglw.setup_basic_logging(20)  # logging.INFO
    window_cls = mglw.get_local_window_cls()

    start_fullscreen = True
    if "-f" in sys.argv:
        start_fullscreen = not start_fullscreen

    window = window_cls(
        fullscreen=start_fullscreen,
        # resizable=config_cls.resizable,
        # gl_version=config_cls.gl_version,
        # vsync=config_cls.vsync,
        # samples=config_cls.samples,
    )
    if start_fullscreen:
        window.mouse_exclusivity = True
    window.print_context_info()
    mglw.activate_context(window=window)
    timer = moderngl_window.timers.clock.Timer()
    window.config = ModernSlideShower(ctx=window.ctx, wnd=window, timer=timer)

    timer.start()

    while not window.is_closing:
        window.swap_buffers()
        current_time, delta = timer.next_frame()
        window.render(current_time, delta)
        # if not window.is_closing:
        #     window.swap_buffers()

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
