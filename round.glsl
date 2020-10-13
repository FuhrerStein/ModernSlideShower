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
