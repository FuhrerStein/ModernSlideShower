#version 430

#if defined VERTEX_SHADER

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;

out vec2 uv0;

void main() {
    uv0 = in_texcoord;
    gl_Position = vec4(in_position.x, -in_position.y, 1, 1);
}


#elif defined FRAGMENT_SHADER

layout(binding=5) uniform sampler2D texture0;
layout(binding=6) uniform sampler2D texture_curve;
layout(binding=7, r32ui) uniform uimage2D histogram_texture;

uniform bool useCurves;
uniform bool count_histograms;
in vec2 uv0;
//uniform vec2 transition_center;
//uniform float zoom_scale;
//uniform float inter_blur;
//uniform float spd;
//uniform float hide_borders;
//in float translucency;
//in float f_show_amount;
//in float min_edge;
//in float max_edge;
//in float tran_blur;


//uniform double X0;
//uniform double Y0;
uniform float zoom;
uniform float complexity;
uniform dvec2 wnd_size;
uniform dvec2 pic_position;

int n;

double a;
double b;
double b_next;

double coord_x;
double coord_y;

float maxIteration;

float final_step = 0;
float pre_final_step = 0;

float red;
float green;
float blue;

float a1;
float a2;
float a3;

void main() {

    maxIteration = - (log2(zoom) * .66 + 10) * complexity;

    coord_x = double(uv0.x - .5) * wnd_size.x / 1000 / zoom - pic_position.x;
    coord_y = double(uv0.y - .5) * wnd_size.y / 1000 / zoom - pic_position.y;

    n = 0;

    while (n + final_step < maxIteration)
    {
//        a2 *= .9;
//        final_step2 += double(final_step / (n + 50));
//        a2 += float((final_step2) * (maxIteration - final_step) * .1);
//        a3 += float((final_step) * (smoothstep(0, 20, float(pow(float(maxIteration), .4)/.0005000  - final_step2 + 30)))) * 0.01;
//        a3 += float((final_step2) * (sin((float(maxIteration / 50 - final_step2 + 10))))) * 1;
//        final_step2 = a + b;
//        step_sum += final_step / n;
//        a3 = float(n / 40 + a * 0);
//        if (n > maxIteration) break;
//        if (n + final_step > maxIteration) break;
//        if (final_step > 5) break;
//        if (maxIteration - final_step2 - 10 > 0) break;
        pre_final_step = final_step;
        final_step = length(vec2(a, b));

        b_next = 2 * a * b + coord_y;
        a = a * a - b * b + coord_x;
        b = b_next;
        n++;
    }
//        a1 = float((final_step2) * (maxIteration - final_step) * 1);
    a1 = smoothstep(0, 40, pre_final_step * (20 - pre_final_step));
    a2 = (pre_final_step - 10) * (maxIteration - n - pre_final_step);
    a2 = smoothstep(-50, (maxIteration - n) * (maxIteration - n - 10), a2) * 2;
    a3 = smoothstep(0, (maxIteration - n), pre_final_step);

    gl_FragColor  = vec4(a2 * 0.4, a2 * .4, a2*1, 1);

//    a1 = smoothstep(-1, 5, float(final_step));
//    a2 = float(final_step);
//    a3 = smoothstep(0, 5, float(final_step2));

//    a2 = smoothstep(0, 10, float(sqrt(a2)));
//    a3 = smoothstep(1, 50, float(a2));



//    red = sin(3.14159 * smoothstep(-1, 1, float(step_sum) + float(final_step)));
//    green = sin(3.14159 * smoothstep(-1, 5, float(step_sum)));
//    blue = .7 * cos(3.14159 * smoothstep(-1, 5, float(step_sum))) + .3 * smoothstep(-1, 5, float(final_step));




    //    float red = pow(smoothstep(-1, 1, sin(length(vec2(a, b)))), 1.7);
    //    float green = sin(length(vec2(a, b)) / 200 + float(sqrt(n / maxIteration))/5);
    //    float blue = 1 - smoothstep(-1, 1, cos(length(vec2(a, b))));
//    float blue = 1 - sin(length(vec2(a, b))) / 1.2;
//    float
//    if (n < maxIteration) gl_FragColor = vec4(length(vec2(a, b)) / 3 , sqrt(n / maxIteration), 0, 1);

//    if (n < maxIteration)
//    {
//        gl_FragColor = vec4(red, green, blue, 1);
//    }
////    else gl_FragColor = vec4(uv0.x, uv0.y, uv0.x + uv0.y, 1);
//    else gl_FragColor = vec4(red, green, blue, 1);// * vec4(uv0.x, uv0.y, uv0.x + uv0.y, 1);
//    gl_FragColor = vec4(red * .7 + green * .3 + blue * -.1, green * .4 + blue * .3 , blue * .5 + green * .1 + red * - .1, 1);// * vec4(uv0.x, uv0.y, uv0.x + uv0.y, 1);


    //    float actual_blur = 1 + inter_blur * tran_blur;
    //    vec2 dx = dFdx(uv0) * actual_blur;
    //    vec2 dy = dFdy(uv0) * actual_blur;
    //    vec2 zero = vec2(0);
    //    tempColor = textureGrad(texture0, uv0, dx, dy);
    //
    //
    //    if(useCurves)
    //    {
    //        // local by-color curves
    //        tempColor.r = texelFetch(texture_curve, ivec2((tempColor.r * 255 + .5), 1), 0).r;
    //        tempColor.g = texelFetch(texture_curve, ivec2((tempColor.g * 255 + .5), 2), 0).r;
    //        tempColor.b = texelFetch(texture_curve, ivec2((tempColor.b * 255 + .5), 3), 0).r;
    //
    //        // global curves
    //        tempColor.r = texelFetch(texture_curve, ivec2((tempColor.r * 255 + .5), 0), 0).r;
    //        tempColor.g = texelFetch(texture_curve, ivec2((tempColor.g * 255 + .5), 0), 0).r;
    //        tempColor.b = texelFetch(texture_curve, ivec2((tempColor.b * 255 + .5), 0), 0).r;
    //    }
    //
    //    if(count_histograms)
    //    {
    //        // gray histogram
    //        // formula is not perfect and needs to be updated to account for 1.0 - 255 conversion
    //        int gray_value = int((tempColor.r * 299 + tempColor.g * 587 + tempColor.b * 114) * 51 / 200);
    //        imageAtomicAdd(histogram_texture, ivec2(gray_value, 0), 1u);
    //
    //        // red, green and blue histograms
    //        // todo: use fma()
    //        imageAtomicAdd(histogram_texture, ivec2(tempColor.r * 255 + .5, 2), 1u);
    //        imageAtomicAdd(histogram_texture, ivec2(tempColor.g * 255 + .5, 3), 1u);
    //        imageAtomicAdd(histogram_texture, ivec2(tempColor.b * 255 + .5, 4), 1u);
    //    }

    // Edges and transitions
    //    float to_edge_x = pow(smoothstep(-0.5, -.5 + hide_borders, -abs(uv0.x - .5)), .3);
    //    float to_edge_y = pow(smoothstep(-0.5, -.5 + hide_borders, -abs(uv0.y - .5)), .3);
    //    float to_edge = smoothstep(0, 1, to_edge_x + to_edge_y) * smoothstep(0, 1,  to_edge_x * to_edge_y);
    //    float tran_alpha = smoothstep(-min_edge, -max_edge, -length(uv0 - transition_center)) * f_show_amount;

    //    fragColor = vec4(tempColor.rgb, tran_alpha * translucency * to_edge);
    //    fragColor = vec4(tempColor, 1);
}

    #endif
