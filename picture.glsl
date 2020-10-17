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
