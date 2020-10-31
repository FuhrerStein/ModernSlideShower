#version 430

#if defined VERTEX_SHADER

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;

uniform vec2 displacement;
uniform vec2 pix_size;
uniform vec2 wnd_size;
uniform float zoom_scale;
uniform float angle;

out vec2 new_pos;

void main() {

    mat2 rotate_mat = mat2(cos(angle), - sin(angle), sin(angle), cos(angle));
    mat2 pix_size_mat = mat2(pix_size.x, 0, 0, pix_size.y);
    mat2 wnd_size_mat = mat2(1 / wnd_size.x, 0, 0, 1 / wnd_size.y);

    new_pos = (in_position * pix_size_mat * rotate_mat + displacement) * zoom_scale * wnd_size_mat;

}

#endif
