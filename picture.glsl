#version 430

#if defined VERTEX_SHADER

layout(location = 0) in vec2 in_position;

uniform vec2 displacement;
uniform vec2 pix_size;
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
// 0: transform preview mode
// 1: render transform mode



out vec2 uv0;
out float translucency;
out float f_show_amount;
out float min_edge;
out float max_edge;
out float tran_blur;
out float border_color;
out int work_axis;
out vec4 crop_borders;


int is_eq(int a, int b){
    return 1 - sign(abs(a - b));
}

int is_neq(int a, int b){
    return sign(abs(a - b));
}

void main() {
    float x_scale = (resize_xy + 1.) * (resize_x + 1.);
    float y_scale = (resize_xy + 1.) * (resize_y + 1.);
    float active_angle = is_neq(process_type, 2) * angle;
    vec2 displacement_m = displacement * is_neq(render_mode, 2);

    mat2 rotate_mat = mat2(cos(active_angle), - sin(active_angle), sin(active_angle), cos(active_angle));
    mat2 pix_size_mat = mat2(pix_size.x * x_scale, 0, 0, pix_size.y * y_scale);
    mat2 pix_size_mat_crop = mat2(crop.x + crop.y, 0, 0, crop.z + crop.w);
    mat2 wnd_size_mat = mat2(1 / wnd_size.x, 0, 0, 1 / wnd_size.y);
    vec2 position_part1 = in_position * (pix_size_mat - pix_size_mat_crop * is_neq(process_type, 0)) * rotate_mat;
    vec2 position_part2 = position_part1 + displacement_m + (crop.xz - crop.yw) * is_neq(process_type, 0);
    gl_Position = vec4(position_part2 * zoom_scale * wnd_size_mat, 0, 1);

    vec2 crop_border_bl = vec2(-1) * (pix_size_mat - pix_size_mat_crop) + displacement_m + (crop.xz - crop.yw);
    crop_border_bl = crop_border_bl * zoom_scale;
    vec2 crop_border_tr = vec2(1) * (pix_size_mat - pix_size_mat_crop) + displacement_m + (crop.xz - crop.yw);
    crop_border_tr = crop_border_tr * zoom_scale;
    crop_borders = vec4(crop_border_bl + wnd_size, crop_border_tr + wnd_size) / 2;

    vec2 crop_scaler = mix(vec2(1), (crop_border_tr - crop_border_bl) * wnd_size_mat / 2, is_eq(render_mode, 2));
    gl_Position.xy -= (crop_border_tr + crop_border_bl) * wnd_size_mat * is_eq(render_mode, 2) / 2 ;
    gl_Position.xy /= crop_scaler;
    crop_borders += vec4(vec2(-50000), vec2(50000)) * is_neq(render_mode, 1);

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

layout(binding=5) uniform sampler2D texture0;
layout(binding=6) uniform sampler2D texture_curve;
layout(binding=7, r32ui) uniform uimage2D histogram_texture;
uniform bool useCurves;
uniform bool count_histograms;
uniform vec2 transition_center;
uniform float zoom_scale;
uniform float inter_blur;
uniform float spd;
uniform vec4 crop;
uniform vec2 pix_size;
uniform int render_mode;

out vec4 fragColor;
in vec2 uv0;
in float translucency;
in float f_show_amount;
in float min_edge;
in float max_edge;
in float tran_blur;
//in flat float crop_borders;
in vec4 crop_borders;

uniform float hide_borders;


int is_eq(int a, int b){
    return 1 - sign(abs(a - b));
}

int is_neq(int a, int b){
    return sign(abs(a - b));
}

void main() {
    float actual_blur = 1 + inter_blur * tran_blur;
    vec2 dx = dFdx(uv0) * actual_blur;
    vec2 dy = dFdy(uv0) * actual_blur;
    vec4 tempColor = vec4(0);
    vec2 zero = vec2(0);
    tempColor = textureGrad(texture0, uv0, dx, dy);
    if (actual_blur > 1.1)
    {
        tempColor *= .2;
        tempColor += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * ( dx + dy), dx, dy);
        tempColor += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * (-dx + dy), dx, dy);
        tempColor += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * (-dx - dy), dx, dy);
        tempColor += .2 * textureGrad(texture0, uv0 + sub_pixel_distance * ( dx - dy), dx, dy);
    }

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
    float hide_borders_m = hide_borders * is_eq(0, render_mode);
    float to_edge_x = pow(smoothstep(-0.5, -.5 + hide_borders_m, -abs(uv0.x - .5)), .3);
    float to_edge_y = pow(smoothstep(-0.5, -.5 + hide_borders_m, -abs(uv0.y - .5)), .3);
    float to_edge = smoothstep(0, 1, to_edge_x + to_edge_y) * smoothstep(0, 1,  to_edge_x * to_edge_y);
    float tran_alpha = smoothstep(-min_edge, -max_edge, -length(uv0 - transition_center)) * f_show_amount;

    float crop_alpha = 1;
//    crop_alpha *= step(crop.x / pix_size.x, uv0.x) * step(crop.y / pix_size.x - 1, -uv0.x);
//    crop_alpha *= step(crop.w / pix_size.y, uv0.y) * step(crop.z / pix_size.y - 1, -uv0.y);
    crop_alpha *= step(crop_borders.x, gl_FragCoord.x) * step(gl_FragCoord.x, crop_borders.z);
    crop_alpha *= step(crop_borders.y, gl_FragCoord.y) * step(gl_FragCoord.y, crop_borders.w);
    crop_alpha += .2;
    fragColor = vec4(tempColor.rgb, tran_alpha * translucency * to_edge * crop_alpha);
}

#endif
