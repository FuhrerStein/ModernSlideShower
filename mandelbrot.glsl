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

#define complexMul(a, b) vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)

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
uniform dvec2 pic_position_precise;

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

    maxIteration = (log2(zoom) * .66 + 10) * complexity;

    coord_x = double(uv0.x - .5) * wnd_size.x / 1000 / zoom - pic_position.x;
    coord_y = double(uv0.y - .5) * wnd_size.y / 1000 / zoom - pic_position.y;
    double max_win = max(wnd_size.x, wnd_size.y);
    double coord_x_high = - pic_position.x / max_win;
    double coord_y_high = - pic_position.y / max_win;
    double coord_x_low  = double(uv0.x - .5) * wnd_size.x * .001 / zoom - pic_position_precise.x / max_win / 1e26*0;
    double coord_y_low  = double(uv0.y - .5) * wnd_size.y * .001 / zoom - pic_position_precise.y / max_win / 1e26*0;

    n = 0;

//    vec2 zlo = vec2(0.);
//    vec2 zhi = vec2(0.);
//    vec2 add = vec2(0.);
//    vec2 z   = vec2(0.);

//
//    while (n + final_step < maxIteration)
//    {
//        pre_final_step = final_step;
//        final_step = length(vec2(a, b));
//
//        b_next = 2 * a * b + coord_y;
//        a = a * a - b * b + coord_x;
//        b = b_next;
//        n++;
//    }

    double a_hi = 0;
    double b_hi = 0;
//    double b_hi_next;


    while (n + final_step < maxIteration)
    {
//        add = 2.0 * complexMul(zhi, zlo);
//        zhi = complexMul(zhi, zhi)       + pc.xy;
//        zlo = complexMul(zlo, zlo) + add + pc.zw;
//        z = zhi + zlo;
//
//

        pre_final_step = final_step;
        final_step = length(vec2((a + a_hi), (b + b_hi)));

//        double a_hi2 = a_hi * a_hi;
//        double a_lo2 = a * a;

//        b_next = 2 * (a + a_hi) * b + coord_y;


//        b_next = 2 * (a + a_hi) * (b_hi + b) + coord_y_high;

        double b_all2 = (b_hi + b) * (b_hi + b);
        double b_hi_o = b_hi;
        double b_o = b;


        b = 2 * (a * b + a_hi * b + a * b_hi) + coord_y_low;
        b_hi = 2 * a_hi * b_hi + coord_y_high;

        a = a * a + coord_x_low + 2 * a_hi * a - b_o * b_o - 2 * b_hi_o * b_o;
//        a_hi = a_hi * a_hi + coord_x_high - b_all2 ;
//        a_hi = a_hi * a_hi + coord_x_high - (b_hi + b) * (b_hi + b) ;
        a_hi = a_hi * a_hi + coord_x_high - b_hi_o * b_hi_o;


        n++;
//
//        if (abs(a_hi) / abs(a) < 10000000) {
//            a_hi += a * .9;
//            a -= a * .9;
//        }
//
//        if (abs(b_hi) / abs(b) < 10000000) {
//            b_hi += b * .9;
//            b -= b * .9;
//        }

        if (abs(a_hi) / abs(a) < 100) {
            a_hi += a;
            a = 0;
        }

        if (abs(b_hi) / abs(b) < 100) {
            b_hi += b;
            b = 0;
        }


//
//        if (abs(a_hi) / abs(a) > 1000000) {
//            a += a_hi * .0001;
//            a_hi -= a_hi * .0001;
//        }
//
//        if (abs(b_hi) / abs(b) > 1000000) {
//            b += b_hi * .0001;
//            b_hi -= b_hi * .0001;
//        }

    }


    a1 = smoothstep(0, 40, pre_final_step * (20 - pre_final_step));
    a2 = (pre_final_step - 10) * (maxIteration - n - pre_final_step);
    a2 = smoothstep(-50, (maxIteration - n) * (maxIteration - n - 10), a2) * 2;
    a3 = smoothstep(0, (maxIteration - n), pre_final_step);

    gl_FragColor  = vec4(a2 * 0.4, a2 * .4, a2 * 1, 1);

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
