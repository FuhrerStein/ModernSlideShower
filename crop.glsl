#version 430

#if defined VERTEX_SHADER

layout(location = 0) in vec2 in_position;

uniform vec2 displacement;
uniform vec2 pix_size;
uniform vec2 wnd_size;
uniform float zoom_scale;
uniform float angle;
uniform vec4 crop;
uniform int active_border_id;

out float border_color;

void main() {
    mat2 rotate_mat = mat2(cos(angle), - sin(angle), sin(angle), cos(angle));
    mat2 pix_size_mat = mat2(pix_size.x - crop.x - crop.y, 0, 0, pix_size.y - crop.z - crop.w);
    mat2 wnd_size_mat = mat2(1 / wnd_size.x, 0, 0, 1 / wnd_size.y);
    gl_Position = vec4((in_position * pix_size_mat * rotate_mat + displacement + crop.xz - crop.yw ) * zoom_scale * wnd_size_mat, 0, 1);
    border_color = .5;
    if (gl_VertexID / 2 == active_border_id - 1){
        border_color = 1;
    }

}
#elif defined FRAGMENT_SHADER

out vec4 fragColor;
in float border_color;

void main() {
    fragColor = vec4(border_color);
}

#endif

