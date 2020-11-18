#version 430

int is_eq(int a, int b){
    return 1 - sign(abs(a - b));
}

int is_neq(int a, int b){
    return sign(abs(a - b));
}

//
//float smootherstep(float edge0, float edge1, float x) {
//    // Scale, and clamp x to 0..1 range
//    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
//    // Evaluate polynomial
//    return x * x * x * (x * (x * 6 - 15) + 10);
//}
//
//double smootherstep(double edge0, double edge1, double x) {
//    // Scale, and clamp x to 0..1 range
//    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
//    // Evaluate polynomial
//    return x * x * x * (x * (x * 6 - 15) + 10);
//}
//
//dvec2 smootherstep(double edge0, double edge1, dvec2 x) {
//    // Scale, and clamp x to 0..1 range
//    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
//    // Evaluate polynomial
//    return x * x * x * (x * (x * 6 - 15) + 10);
//}
//
//dvec2 smootherstep01(dvec2 x1) {
//    // Scale, and clamp x to 0..1 range
//    dvec2 x = clamp(x1, 0.0, 1.0);
//    // Evaluate polynomial
//    return x * x * x * (x * (x * 6 - 15) + 10);
//}


    #if defined VERTEX_SHADER

layout(location = 0) in vec2 in_position;

layout(binding=5) uniform sampler2D texture0;

uniform vec2 displacement;
uniform vec2 wnd_size;
uniform float zoom_scale;
uniform float angle;
uniform float show_amount;
uniform float transparency;
uniform float resize_xy;
uniform float resize_x;
uniform float resize_y;
uniform bool one_by_one;
uniform int active_border_id;
uniform vec4 crop;
uniform int process_type;
// 0: picture
// 1: border
// 2: crop interface
uniform int render_mode;
// 0: normal mode
// 1: transform preview mode
// 2: render transform mode

out vec2 uv0;
out float translucency;
out float f_show_amount;
out float min_edge;
out float max_edge;
out float tran_blur;
out float border_color;
out int work_axis;
out vec4 crop_borders;
out vec2 real_pic_size;


void compute_positions(){
    vec2 pix_size = textureSize(texture0, 0);
    vec2 whole_resize = vec2((resize_xy + 1.) * (resize_x + 1.), (resize_xy + 1.) * (resize_y + 1.));
    vec2 crop_amount = vec2(crop.x + crop.y, crop.z + crop.w);
    float active_angle = is_neq(process_type, 2) * angle;
    mat2 rotate_mat = mat2(cos(active_angle), - sin(active_angle), sin(active_angle), cos(active_angle));
    vec2 displacement_m = displacement * is_neq(render_mode, 2);
    vec2 crop_displacement = crop.xz - crop.yw;

    real_pic_size = 2 * zoom_scale * (pix_size * whole_resize - crop_amount);

    vec2 pic_resized = pix_size * whole_resize;

    vec2 position_part1 = in_position * (pic_resized - crop_amount * is_neq(process_type, 0)) * rotate_mat;
    vec2 position_part2 = position_part1 + displacement_m + crop_displacement * is_neq(process_type, 0);
    gl_Position = vec4(position_part2 * zoom_scale / wnd_size, 0, 1);

    vec2 cropped_pix_size = pic_resized - crop_amount;
    vec2 crop_average_displacement = displacement_m + crop_displacement;
    vec2 crop_border_bl = (crop_average_displacement - cropped_pix_size) * zoom_scale + wnd_size;
    vec2 crop_border_tr = (crop_average_displacement + cropped_pix_size) * zoom_scale + wnd_size;
    crop_borders = vec4(crop_border_bl, crop_border_tr) / 2;

    vec2 crop_scaler = mix(vec2(1), (pic_resized - crop_amount) * zoom_scale / wnd_size, is_eq(render_mode, 2));
    gl_Position.xy -= crop_average_displacement * zoom_scale / wnd_size * is_eq(render_mode, 2);
    gl_Position.xy /= crop_scaler;
    crop_borders += 50000 * vec4(-1, -1, 1, 1) * is_neq(render_mode, 1);
}

void main() {
    compute_positions();

    uv0 = (in_position + 1) / 2;
    translucency = 1 - smoothstep(.5, .7, transparency);
    f_show_amount = smoothstep(0, .7, show_amount);
    min_edge = 2.5 * (smoothstep(0, 1, show_amount)) + show_amount;
    max_edge = 2 * (smoothstep(.7, 1, show_amount)) + show_amount;
    tran_blur = smoothstep(-.5, 0, -show_amount) + (smoothstep(0, .5, transparency));

    border_color = 1 - sign(abs(gl_VertexID / 2 - active_border_id + 1));
    work_axis = 1 - int(gl_VertexID / 4);

    if (one_by_one)
    {
        gl_Position = vec4(in_position, 1, 1);
    }
}

    #elif defined FRAGMENT_SHADER

    #define sub_pixel_distance .45
    #define PI 3.1415926538
    #define inter_pixel_gap .001925// origin unknown, chosen experimentaly


//layout(binding=10) uniform sampler2D texture_hd_r;
//layout(binding=11) uniform sampler2D texture_hd_g;
//layout(binding=12) uniform sampler2D texture_hd_b;
//layout(binding=15) uniform sampler2D levels_borders;
//layout(binding=6) uniform sampler2D texture_curve;
layout(binding=5) uniform sampler2D texture0;
layout(binding=7, r32ui) uniform uimage2D histogram_texture;

uniform bool useCurves;
uniform bool count_histograms;
uniform vec2 transition_center;
uniform float zoom_scale;
uniform float inter_blur;
uniform float spd;
uniform float pixel_size;
uniform vec4 crop;
uniform vec4 lvl_i_min;
uniform vec4 lvl_i_max;
uniform vec4 lvl_o_min;
uniform vec4 lvl_o_max;
uniform vec4 lvl_gamma;
uniform int render_mode;
uniform vec2 wnd_size;
uniform float transparency;
uniform bool one_by_one;

out vec4 fragColor;
in vec2 uv0;
in float translucency;
in float f_show_amount;
in float min_edge;
in float max_edge;
in float tran_blur;
in vec4 crop_borders;

uniform float hide_borders;

vec2 pix_size = textureSize(texture0, 0);
float to_edge_a;
vec2 to_edge;
dvec4 pixel_color_hd;
vec4 pixel_color;


float get_gray(vec4 pixel_color){
    return pixel_color.r * .299 + pixel_color.g * .587 + pixel_color.b * .114;
}

double get_gray(dvec4 pixel_color){
    return pixel_color.r * .299 + pixel_color.g * .587 + pixel_color.b * .114;
}

// unclamped average between smoothstep and smootherstep
dvec2 smootherstep_ease(dvec2 x) {
    return x * x * (x * (x * (x * 6 - 15) + 8) + 3) / 2;
}

double pixel_bands_mixed(vec4 four_pix, dvec2 in_pixel_coords)
{
    dvec2 half_pixel;
    half_pixel.x = mix(four_pix.x, four_pix.y, in_pixel_coords.x);
    half_pixel.y = mix(four_pix.w, four_pix.z, in_pixel_coords.x);
    return mix(half_pixel.y, half_pixel.x, in_pixel_coords.y);
}

dvec4 pixel_color_mixed(sampler2D tex, vec2 uv, float pixel_size)
{
    dvec4 result_pixel;
    vec2 image_size = textureSize(tex, 0);
    dvec2 in_pixel_coords;
    in_pixel_coords = fract((uv) * image_size - .5 + inter_pixel_gap);
    in_pixel_coords = fma(in_pixel_coords, dvec2(1. + pixel_size), dvec2(-pixel_size / 2));
    in_pixel_coords = clamp(in_pixel_coords, 0, 1);
    in_pixel_coords = smootherstep_ease(in_pixel_coords);

    result_pixel.r = pixel_bands_mixed(textureGather(tex, uv, 0), in_pixel_coords);
    result_pixel.g = pixel_bands_mixed(textureGather(tex, uv, 1), in_pixel_coords);
    result_pixel.b = pixel_bands_mixed(textureGather(tex, uv, 2), in_pixel_coords);
    result_pixel.a = get_gray(result_pixel);
    return result_pixel;
}


void main() {
    float actual_blur = 1 + inter_blur * tran_blur;
    vec2 dx = dFdx(uv0) * actual_blur;
    vec2 dy = dFdy(uv0) * actual_blur;
    pixel_color = textureGrad(texture0, uv0, dx, dy);

    if ((zoom_scale > 1) && (transparency == 0) && (one_by_one == false))
    {
        pixel_color_hd = pixel_color_mixed(texture0, uv0, pixel_size);
        { pixel_color = mix(pixel_color, vec4(pixel_color_hd), smoothstep(1, 2, log10(zoom_scale))); }
    }

    if (actual_blur > 1.1)
    {
        pixel_color *= .2;
        pixel_color += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * (dx + dy), dx, dy);
        pixel_color += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * (-dx + dy), dx, dy);
        pixel_color += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * (-dx - dy), dx, dy);
        pixel_color += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * (dx - dy), dx, dy);
    }

    if (useCurves)
    {
        pixel_color = (pixel_color - lvl_i_min) / (lvl_i_max - lvl_i_min);
        pixel_color = clamp(pixel_color, 0, 1);
        pixel_color = pow(pixel_color, lvl_gamma);
        pixel_color = pixel_color * (lvl_o_max - lvl_o_min) + lvl_o_min;
        pixel_color = clamp(pixel_color, 0, 1);

        pixel_color = (pixel_color - lvl_i_min.a) / (lvl_i_max.a - lvl_i_min.a);
        pixel_color = clamp(pixel_color, 0, 1);
        pixel_color = pow(pixel_color, lvl_gamma.aaaa);
        pixel_color = pixel_color * (lvl_o_max.a - lvl_o_min.a) + lvl_o_min.a;
        pixel_color = clamp(pixel_color, 0, 1);
    }

    if (count_histograms)
    {
        // gray value
        pixel_color.a = get_gray(pixel_color);
        //        set_gray(pixel_color);
        ivec4 color_coord = ivec4(fma(pixel_color, vec4(255), vec4(.5)));
        imageAtomicAdd(histogram_texture, ivec2(color_coord.a, 0), 1u);
        imageAtomicAdd(histogram_texture, ivec2(color_coord.r, 1), 1u);
        imageAtomicAdd(histogram_texture, ivec2(color_coord.g, 2), 1u);
        imageAtomicAdd(histogram_texture, ivec2(color_coord.b, 3), 1u);
    }

    if ((zoom_scale > 1) && (transparency == 0) && (true))
    {
        float past_layer_scale;
        float next_layer_scale;
        float layer_alpha;
        dvec4 result_multiplyer = dvec4(1);
        int current_layer = 0;

        do {
            past_layer_scale = next_layer_scale;
            next_layer_scale = pow(2, current_layer + 1);
            layer_alpha = smoothstep(past_layer_scale, next_layer_scale * 2, zoom_scale);
            layer_alpha = .04 * layer_alpha * smoothstep(-1, 5, current_layer);

            result_multiplyer *= 1 + layer_alpha * sin(vec4(pixel_color_hd * PI * 50 * past_layer_scale));
            current_layer += 1;
        } while (layer_alpha > 1e-4);
        pixel_color *= vec4(result_multiplyer.a);
    }

    // Edges and transitions
    float hide_borders_m = hide_borders * is_eq(0, render_mode);
    to_edge = (1 - abs(2 * uv0 - 1)) * pix_size * zoom_scale;
    to_edge = smoothstep(0, hide_borders_m + inter_blur * tran_blur * 5, to_edge);
    to_edge = pow(to_edge, vec2(.3));
    to_edge_a = smoothstep(0, 1, to_edge.x + to_edge.y) * smoothstep(0, 1, to_edge.x * to_edge.y);

    float tran_alpha = smoothstep(-min_edge, -max_edge, -length(uv0 - transition_center)) * f_show_amount;

    float crop_alpha = .8;
    crop_alpha *= step(crop_borders.x, gl_FragCoord.x) * step(gl_FragCoord.x, crop_borders.z);
    crop_alpha *= step(crop_borders.y, gl_FragCoord.y) * step(gl_FragCoord.y, crop_borders.w);
    crop_alpha += .2;
    fragColor = vec4(pixel_color.rgb, tran_alpha * translucency * to_edge_a * crop_alpha);
}

    #elif defined CROP_FRAGMENT

uniform vec2 displacement;
uniform vec2 wnd_size;
uniform float zoom_scale;
out vec4 fragColor;
in float border_color;
in flat int work_axis;
in vec2 real_pic_size;

void main() {
    float alpha = 2 * abs(((displacement * zoom_scale + wnd_size - 2 * gl_FragCoord.xy) / real_pic_size)[work_axis]);
    fragColor = vec4(vec3(.8) + border_color * vec3(.2, -.3, -.3), alpha);
}

#elif defined ROUND_VERTEX
#define point_size 25
#define round_alpha .15

layout(location = 0) in vec2 in_position;

uniform vec2 wnd_size;
uniform int finish_n;

out float alpha;

void main() {
    float point_n_norm = mod(gl_VertexID, 25) * 4;
    float s1 = smoothstep(100 + finish_n * 1.1, 120 + finish_n * 1.2, point_n_norm);
    float s2 = smoothstep(finish_n - 10, finish_n * 1.1, point_n_norm);
    alpha = round_alpha * (s1 + 1 - s2);

    gl_Position = vec4(in_position / wnd_size - 1, .5, 1.0);
    gl_PointSize = point_size;
}

#elif defined ROUND_FRAGMENT

in float alpha;
out vec4 fragColor;

void main() {
    fragColor = vec4(1);
    fragColor.a = alpha - pow(distance(gl_PointCoord.xy, vec2(0.5)), 2.5);
}

    #endif
