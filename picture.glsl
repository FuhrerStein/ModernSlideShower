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
uniform bool one_by_one;

out vec2 uv0;
out float zoom_frag;
out float show_frag;
out float show_amount_frag;

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
//    show_frag = clamp(abs(show_amount) * 8 + in_position.y * (sign(show_amount) + 1) - 1, -1, 2);
//    show_frag = abs(show_amount) * 8 + in_position.y * (sign(show_amount) + 1) - 4.5;
//    show_frag = clamp(abs(show_amount) * 8 + in_position.y * (sign(show_amount) + 1) - 4.5, 0., 1.2);
    show_frag = smoothstep(0, 4, abs(show_amount * (1.5 - sign(show_amount) / 2)) * 8 + (in_position.y - 1.5) * (sign(show_amount) + 1));
    show_amount_frag = 1 - abs(show_amount);

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
in float show_frag;
in float show_amount_frag;

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

    float closeness_to_edge = length(uv0.x - .5) * length(uv0.x - .5);
    closeness_to_edge = smoothstep(0, .2, show_amount_frag) * smoothstep(0, .5, closeness_to_edge) * smoothstep(0, .1, closeness_to_edge);
    fragColor = vec4(tempColor.rgb, show_frag * (1 - closeness_to_edge));
//    fragColor = vec4(tempColor.rgb, show_frag );
}

#endif
