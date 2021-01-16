#version 430


    #if defined PICTURE_VETREX
layout(binding=5) uniform sampler2D image_texture;
uniform vec2 displacement;
uniform vec2 wnd_size;
uniform float zoom_scale;
uniform float angle;
uniform float resize_xy;
uniform float resize_x;
uniform float resize_y;
uniform vec4 crop;
uniform int process_type;
// 0: draw picture
// 1: draw picture in transform preview mode
// 2: transform render mode
// 3:
// 4: apply levels

out global_data{
    vec2 point_coords[4];
    vec2 uv_coords[4];
    vec4 crop_borders;

    float translucency;
    float f_show_amount;
    float min_edge;
    float max_edge;
    float tran_blur;
};

// todo: refactor this
uniform float show_amount;
uniform float transparency;

vec4 points_x = vec4(-1, 1, -1, 1);
vec4 points_y = vec4(-1, -1, 1, 1);

vec2 tex_size;
vec2 whole_resize;
mat2 rotate_mat;
vec2 tex_resized;
vec4 pix_extremes;

void transform_point(int point_id) {
    vec2 point_coord_0 = vec2(points_x[point_id], points_y[point_id]);
    uv_coords[point_id] = (point_coord_0 + 1) / 2;
    vec2 point_coord = (point_coord_0 * tex_resized * rotate_mat + displacement) * zoom_scale / wnd_size;
    point_coords[point_id] = ((process_type == 4) ? point_coord_0 : point_coord);
    pix_extremes.xy = min(pix_extremes.xy, point_coord);
    pix_extremes.zw = max(pix_extremes.zw, point_coord);
}

void main() {

    tex_size = textureSize(image_texture, 0);
    whole_resize = vec2((resize_xy + 1.) * (resize_x + 1.), (resize_xy + 1.) * (resize_y + 1.));
    rotate_mat = mat2(cos(angle), - sin(angle), sin(angle), cos(angle));
    tex_resized = tex_size * whole_resize;

    transform_point(0);
    pix_extremes = vec4(point_coords[0], point_coords[0]);
    transform_point(1);
    transform_point(2);
    transform_point(3);

    crop_borders = (pix_extremes + 1) / 2 * wnd_size.xyxy - vec4(-crop.x, -crop.y, crop.z, crop.w) * zoom_scale;

    if (process_type == 2){
        vec2 transform_displacement = (1 - (crop_borders.xy + crop_borders.zw) / wnd_size) ;
        vec2 transform_scale = wnd_size / (crop_borders.zw - crop_borders.xy);
        for (int i = 0; i < 4; i++) {
            point_coords[i] = (point_coords[i] + transform_displacement) * transform_scale;
        }
    }

    // todo: refactor this
    translucency = 1 - smoothstep(.5, .7, transparency);
    f_show_amount = smoothstep(0, .7, show_amount);
    min_edge = 2.5 * (smoothstep(0, 1, show_amount)) + show_amount;
    max_edge = 2 * (smoothstep(.7, 1, show_amount)) + show_amount;
    tran_blur = smoothstep(-.5, 0, -show_amount) + (smoothstep(0, .5, transparency));
}


#elif defined PICTURE_GEOMETRY

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;


in global_data{
vec2 point_coords[4];
vec2 uv_coords[4];
vec4 crop_borders;

float translucency;
float f_show_amount;
float min_edge;
float max_edge;
float tran_blur;
} in_data[];

out vec2 uv0;
out float translucency;
out float f_show_amount;
out float min_edge;
out float max_edge;
out float tran_blur;
out vec4 crop_borders;



void emit_point(int point_id){
    gl_Position = vec4(in_data[0].point_coords[point_id], 0.0, 1.0);
    uv0 = in_data[0].uv_coords[point_id];
    EmitVertex();
}

void main() {
    translucency = in_data[0].translucency;
    f_show_amount =  in_data[0].f_show_amount;
    min_edge = in_data[0].min_edge;
    max_edge = in_data[0].max_edge;
    tran_blur = in_data[0].tran_blur;
    crop_borders = in_data[0].crop_borders;

    emit_point(0);
    emit_point(1);
    emit_point(2);
    emit_point(3);
    EndPrimitive();
}

    #elif defined PICTURE_FRAGMENT

    #define sub_pixel_distance .45
    #define PI 3.1415926538
    #define inter_pixel_gap .001925// origin unknown, chosen experimentaly
    #define  Pr  .299
    #define  Pg  .587
    #define  Pb  .114



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
uniform vec4 saturation;
uniform int render_mode;
uniform int process_type;
uniform vec2 wnd_size;
uniform float transparency;
uniform bool one_by_one;
uniform float half_picture;

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
    return pixel_color.r * Pr + pixel_color.g * Pg + pixel_color.b * Pb;
}

double get_gray(dvec4 pixel_color){
    return pixel_color.r * Pr + pixel_color.g * Pg + pixel_color.b * Pb;
}

// unclamped average between smoothstep and smootherstep
dvec2 smootherstep_ease(dvec2 x) {
    return x * x * (x * (x * (x * 6 - 15) + 8) + 3) / 2;
}

double pixel_bands_mixed(vec4 four_pix, dvec2 in_pixel_coords)
{
    dvec2 half_pixel;
    half_pixel = mix(four_pix.xw, four_pix.yz, in_pixel_coords.x);
    return mix(half_pixel.y, half_pixel.x, in_pixel_coords.y);
}

vec4 change_saturation_hard(vec4 pix, float change) {

    pix.a = sqrt(
        pix.r * pix.r * Pr +
        pix.g * pix.g * Pg +
        pix.b * pix.b * Pb ) ;

    pix = mix(pix.aaaa, pix, change);

    return pix;
}

vec4 change_saturation_soft(vec4 pix, float change) {

    pix.a = sqrt(
        pix.r * pix.r * Pr +
        pix.g * pix.g * Pg +
        pix.b * pix.b * Pb ) ;

    float new_sat, vibrance, change_1, desaturator;
    change_1 = change - 1;
    vec3 pix_diff;
    pix_diff = pix.rgb - pix.aaa;
    pix_diff *= pix_diff;
    vibrance = pix_diff.r + pix_diff.g + pix_diff.b;
    desaturator = 1 + vibrance * change_1 * ((change_1 > 0) ? 50 : 0);

    new_sat = 1 + change_1 / desaturator;
    pix = mix(pix.aaaa, pix, new_sat);
    return pix;
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

    if ((zoom_scale > 1) && (transparency == 0) && (process_type == 0))
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
        pixel_color = change_saturation_hard(pixel_color, saturation.x);
        pixel_color = change_saturation_soft(pixel_color, saturation.y);

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


//        if ((gl_FragCoord.x / wnd_size.x) < .3){
////            pixel_color = changeSaturation(pixel_color.r, pixel_color.g, pixel_color.b, saturation.x);
//            pixel_color = change_saturation(pixel_color, saturation.x);
//        }
//        else{
//            pixel_color = change_saturation_soft2(pixel_color, saturation.x);
//        }

        pixel_color = change_saturation_hard(pixel_color, saturation.z);
        pixel_color = change_saturation_soft(pixel_color, saturation.w);

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

    if ((zoom_scale > 1) && (transparency == 0) && (process_type == 0))
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
    float hide_borders_m = hide_borders * (process_type == 0 ? 1 : 0);
    to_edge = (1 - abs(2 * uv0 - 1)) * pix_size * zoom_scale;
    to_edge = smoothstep(0, hide_borders_m + inter_blur * tran_blur * 5, to_edge);
    to_edge = pow(to_edge, vec2(.3));
    to_edge_a = smoothstep(0, 1, to_edge.x + to_edge.y) * smoothstep(0, 1, to_edge.x * to_edge.y);

    float tran_alpha = smoothstep(-min_edge, -max_edge, -length(uv0 - transition_center)) * f_show_amount;

    float crop_alpha = .8;
    crop_alpha *= step(crop_borders.x, gl_FragCoord.x) * step(gl_FragCoord.x, crop_borders.z);
    crop_alpha *= step(crop_borders.y, gl_FragCoord.y) * step(gl_FragCoord.y, crop_borders.w);
    crop_alpha = (process_type == 1 ? crop_alpha + .2 : 1);
    float half_picture_alpha = 1;
    half_picture_alpha *= step(gl_FragCoord.x, wnd_size.x * (1 - half_picture));
    half_picture_alpha *= step(- wnd_size.x * half_picture, gl_FragCoord.x);

    float irregular_alpha = 0;
    irregular_alpha += cos(uv0.x * 78 + uv0.y * 20) + cos(uv0.y * 78 + uv0.x * 20);
    irregular_alpha += sin(uv0.x * 43 - uv0.y * 120) + cos(uv0.y * 43 - uv0.x * 120);

    float final_alpha = pow(to_edge_a, 1 + irregular_alpha / 20 ) * crop_alpha * tran_alpha * translucency * half_picture_alpha;
    fragColor = vec4(pixel_color.rgb, final_alpha);
}


#elif defined COMPARE_VETREX

void main() {
}


#elif defined COMPARE_GEOMETRY

layout (points) in;
layout (line_strip, max_vertices = 2) out;
uniform float line_position;
//uniform vec2 wnd_size;

void main() {
//    borders_rel = crop_borders / wnd_size.xyxy * 2 - 1;
    float x, y;

    x = line_position * 2 - 1;
    gl_Position = vec4(x, -1, 0.0, 1.0);
    EmitVertex();

    gl_Position = vec4(x, 1, 0.0, 1.0);
    EmitVertex();

    EndPrimitive();
}

    #elif defined COMPARE_FRAGMENT

//uniform vec2 wnd_size;
//in float border_color;
//in flat int work_axis;
//in vec4 crop_borders;

out vec4 fragColor;

void main() {
//    vec2 invert_size = (crop_borders.zw - crop_borders.xy) / 15;
//    vec2 alpha = invert_size * (1 / (gl_FragCoord.xy - crop_borders.xy) + 1 / (gl_FragCoord.xy - crop_borders.zw));
//
//    fragColor = vec4(vec3(.8) + border_color * vec3(.2, -.3, -.3), .3 + abs(alpha[work_axis]));
    fragColor = vec4(1);
}



#elif defined CROP_GEOMETRY

layout (points) in;
layout (line_strip, max_vertices = 8) out;
uniform int active_border_id;
uniform vec2 wnd_size;

in global_data{
vec2 point_coords[4];
vec2 uv_coords[4];
vec4 crop_borders;

float translucency;
float f_show_amount;
float min_edge;
float max_edge;
float tran_blur;
} in_data[];

out vec4 crop_borders;
out flat int work_axis;
out float border_color;
const float point_rel_coords_x[8] = {  0,   0, -.1, 1.1,   1,   1, -.1, 1.1};
const float point_rel_coords_y[8] = {-.1, 1.1,   0,   0, -.1, 1.1,   1,   1};
vec4 borders_rel;

void emit_point(int point_id){
    float x, y;
    x = mix(borders_rel.x, borders_rel.z, point_rel_coords_x[point_id]);
    y = mix(borders_rel.y, borders_rel.w, point_rel_coords_y[point_id]);
    gl_Position = vec4(x, y, 0.0, 1.0);
    EmitVertex();
}


void emit_line(int line_id){
    border_color = (((line_id + 1) == active_border_id) ?  1 : 0 );
    work_axis = line_id >> 1;
    emit_point(line_id * 2);
    emit_point(line_id * 2 + 1);
    EndPrimitive();
}

void main() {
    crop_borders = in_data[0].crop_borders;
    borders_rel = crop_borders / wnd_size.xyxy * 2 - 1;

    emit_line(0);
    emit_line(1);
    emit_line(2);
    emit_line(3);
}

    #elif defined CROP_FRAGMENT

uniform vec2 wnd_size;
in float border_color;
in flat int work_axis;
in vec4 crop_borders;

out vec4 fragColor;

void main() {
    vec2 invert_size = (crop_borders.zw - crop_borders.xy) / 15;
    vec2 alpha = invert_size * (1 / (gl_FragCoord.xy - crop_borders.xy) + 1 / (gl_FragCoord.xy - crop_borders.zw));

    fragColor = vec4(vec3(.8) + border_color * vec3(.2, -.3, -.3), .3 + abs(alpha[work_axis]));
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
