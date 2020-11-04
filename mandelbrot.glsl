#version 430

#if defined VERTEX_SHADER

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_texcoord;

out vec2 uv0;

void main() {
    uv0 = in_texcoord - .5;
    gl_Position = vec4(in_position.x, -in_position.y, 1, 1);
}


#elif defined FRAGMENT_SHADER

#define split_constant 67108865  // 2^26+1
#define prezoom 1e-3
#define highdef 0

layout(binding=5) uniform sampler2D texture0;
layout(binding=6) uniform sampler2D texture_curve;
layout(binding=8, r32ui) uniform uimage2D histogram_texture;

uniform bool useCurves;
uniform bool count_histograms;
in vec2 uv0;
uniform float zoom;
uniform float complexity;
uniform dvec2 wnd_size;
uniform dvec2 pic_position_1;
uniform dvec2 pic_position_2;
uniform dvec2 pic_position_3;

uniform dvec4 pic_positiondd_x;
uniform dvec4 pic_positiondd_y;

int n;

//double a;
//double b;
//double b_next;

float red;
float green;
float blue;

float a1;
float a2;
float a3;

dvec2 two_sum(double a, double b){
    double s = a + b;
    double v = s - a;
    double e = (a - (s - v)) + (b - v);
    return dvec2(s, e);
}

dvec2 quick_two_sum(double a, double b){
    double s = a + b;
    double e = b - (s - a);
    return dvec2(s, e);
}

dvec4 two_sum_comp(dvec2 a, dvec2 b){
    dvec2 s = a + b;
    dvec2 v = s - a;
    dvec2 e = (a - (s - v)) + (b - v);
    return dvec4(s.x, e.x, s.y, e.y);
}

dvec2 two_vec_add(dvec2 a, dvec2 b){
    dvec2 s, t;
    s = two_sum(a.x, b.x);
    t = two_sum(a.y, b.y);
    s.y += t.x;
    s = quick_two_sum(s.x, s.y);
    s.y += t.y;
    s = quick_two_sum(s.x, s.y);
    return s;
}

dvec2 two_vec_add_vec(dvec2 a, dvec2 b){
    dvec4 st = two_sum_comp(a, b);
    st.y += st.z;
    st.xy = quick_two_sum(st.x, st.y);
    st.y += st.w;
    st.xy = quick_two_sum(st.x, st.y);
    return st.xy;
}

dvec2 two_vec_add_vec4(dvec2 a, dvec2 b){
    double ss, ee;
    dvec2 s = a + b;
    dvec2 v = s - a;
    dvec2 e = (a - (s - v)) + (b - v);
    e.x += s.y;
    ss = s.x + e.x;
    e.x = e.x - (ss - s.x);
    s.x = ss;
    e.x += e.y;
    ss += e.x;
    ee = e.x - (ss - s.x);
    return dvec2(ss, ee);
}

dvec2 two_vec_add_vec2(dvec2 a, dvec2 b){
    double s, e;
    dvec4 st = two_sum_comp(a, b);
    st.y += st.z;
    s = st.x + st.y;
    st.y = st.y - (s - st.x);
    st.x = s;
    st.y += st.w;
    s = st.x + st.y;
    e = st.y - (s - st.x);
    return dvec2(s, e);

//    return st.xy;
}

dvec2 split(double a){
    double t = a * split_constant;
    double a_hi = t - (t - a);
    double a_lo = a - a_hi;
    return dvec2(a_hi, a_lo);
}

dvec4 split_comp(dvec2 c){
    dvec2 t = c * split_constant;
    dvec2 c_hi = t - (t - c);
    dvec2 c_lo = c - c_hi;
    return dvec4(c_hi.x, c_lo.x, c_hi.y, c_lo.y);
}

dvec2 two_prod(double a, double b){
    double p = a * b;
    dvec2 a_s = split(a);
    dvec2 b_s = split(b);
    double err = ((a_s.x * b_s.x - p) + a_s.x * b_s.y + a_s.y * b_s.x) + a_s.y * b_s.y;
    return dvec2(p, err);
}

//dvec2 two_prod(double a, double b){
//    double p = a * b;
//    dvec2 a_s = split(a);
//    dvec2 b_s = split(b);
//    double err1 = a_s.x * b_s.x - p;
//    double err2 = err1 + a_s.y * b_s.x + a_s.x * b_s.y;
//    double err = a_s.y * b_s.y + err2;
//    return dvec2 (p, err);
//}


dvec2 two_prod_fma(double a, double b){
    double x = a * b;
    double y = fma(a, b, -x);
    return dvec2 (x, y);
}

dvec2 two_vec_prod(dvec2 a, dvec2 b){
    dvec2 p = two_prod(a.x, b.x);
    p.y += a.x * b.y;
    p.y += a.y * b.x;
    p = quick_two_sum(p.x, p.y);
    return p;
}

dvec2 two_vec_prod_fma(dvec2 a, dvec2 b){
    dvec2 p = two_prod_fma(a.x, b.x);
    p.y += a.x * b.y;
    p.y += a.y * b.x;
    p = quick_two_sum(p.x, p.y);
    return p;
}

dvec2 two_vec_prod_fma_s(dvec2 a, dvec2 b){
    double x = a.x * b.x;
    double y = fma(a.x, b.x, -x);
    y += a.x * b.y;
    y += a.y * b.x;
    double s = x + y;
    double e = y - (s - x);
    return dvec2(s, e);
}

dvec2 single_sqr(double a){
    double p = a * a;
    dvec2 a_s = split(a);
    double err = ((a_s.x * a_s.x - p) + 2 * a_s.x * a_s.y) + a_s.y * a_s.y;
    return dvec2(p, err);
}

dvec2 single_sqr_fma(double a){
    double x = a * a;
    double y = fma(a, a, -x);
    return dvec2 (x, y);
}


dvec2 vec_sqr(dvec2 a){
    dvec2 p = single_sqr(a.x);
    p.y += 2 * a.x * a.y;
    p = quick_two_sum(p.x, p.y);
    return p;
}

dvec2 vec_sqr_fma(dvec2 a){
    double x, y, s, e;
    x = a.x * a.x;
    y = fma(a.x, a.x, -x);
    y = fma(a.x, 2 * a.y, y);
    s = x + y;
    e = y - (s - x);
    return dvec2(s, e);
}

dvec2 vec_sqr_simple_1(dvec2 a){
    double p_x, p_y, s, e, t, a_hi_hi, a_hi_lo;
    t = a.x * split_constant;
    a_hi_hi = t - (t - a.x);
    a_hi_lo = a.x - a_hi_hi;
    p_x = a.x * a.x;
    p_y = ((a_hi_hi * a_hi_hi - p_x) + 2 * a_hi_hi * a_hi_lo) + a_hi_lo * a_hi_lo;
    p_y += 2 * a.x * a.y;
    s = p_x + p_y;
    e = p_y - (s - p_x);
    return dvec2(s, e);
}


void main() {
    n = 0;

    float final_step2 = 0;
    float final_step_sum = 0;
    float pre_final_step = 0;
    double final_step = 0;

    dvec2 va = dvec2(0);
    dvec2 vb = dvec2(0);
//    dvec2 va_sqr = dvec2(0);
    dvec2 vb_sqr = dvec2(0);
//    dvec2 vb_n = dvec2(0);

    dvec2 coord_x_var = two_prod(wnd_size.x * uv0.x, prezoom / zoom);
    dvec2 coord_y_var = two_prod(wnd_size.y * uv0.y, prezoom / zoom);

    #if (1 - highdef)
        coord_x_var = two_vec_add_vec4(coord_x_var, pic_positiondd_x.xy);
        coord_x_var = two_vec_add_vec4(coord_x_var, pic_positiondd_x.zw);
        coord_y_var = two_vec_add_vec4(coord_y_var, pic_positiondd_y.xy);
        coord_y_var = two_vec_add_vec4(coord_y_var, pic_positiondd_y.zw);
    #endif


    while (n + final_step < complexity)
    {
        pre_final_step = float(final_step);
        final_step = length(dvec2(va.x, vb.x));

        vb_sqr = -vec_sqr_simple_1(vb);
        vb = two_vec_prod_fma_s(2 * va, vb);
        #if highdef
            vb = two_vec_add_vec4(vb, pic_positiondd_y.xy);
            vb = two_vec_add_vec4(vb, pic_positiondd_y.zw);
        #endif
        vb = two_vec_add_vec4(vb, coord_y_var);

        va = two_vec_add_vec4(vec_sqr_simple_1(va), vb_sqr);
        #if highdef
            va = two_vec_add_vec4(va, pic_positiondd_x.xy);
            va = two_vec_add_vec4(va, pic_positiondd_x.zw);
        #endif
        va = two_vec_add_vec4(va, coord_x_var);
        n++;
    }

    int n_was_bigger = int(step(n, complexity * .95) * 32);
    ivec2 cluster_coord = ivec2((uv0 + .5) * 32) + ivec2(0, n_was_bigger);

    imageAtomicAdd(histogram_texture, cluster_coord, 1u);
    float floatmaxIt = float(complexity);
    a1 = smoothstep(0, 40, pre_final_step * (20 - pre_final_step));
    a1 = smoothstep(0, 40, final_step2 * (20 - final_step2));
//    a1 = smoothstep(0, 10000, final_step2 );
//    a1 = smoothstep(0, 10, final_step2 );
//    a1 = smoothstep(0, 1, sin(final_step2 * 1000) );
//    a1 = smoothstep(0, 1, sin(final_step2 * 100000) );
//    a1 = smoothstep(0, 1, cos(final_step2 * 100000) );
//    a1 = smoothstep(-1, 1, cos(final_step2 * 100000) );
//    a1 = smoothstep(-1, 1, tanh(final_step2 * 1) );
//    a1 = smoothstep(-10000, 10000, (final_step_sum * 1) ) * (pre_final_step - 10) * (float(maxIteration) - n - pre_final_step);
//    a1 = smoothstep(-10000, 10000, (final_step_sum * 1) ) * smoothstep(0, (floatmaxIt - n), pre_final_step);
    a2 = (pre_final_step - 10) * (float(complexity) - n - pre_final_step);
    a2 = smoothstep(-50, (floatmaxIt - n) * (floatmaxIt - n - 10), a2) * 2;
    a3 = smoothstep(0, (floatmaxIt - n), pre_final_step);

    gl_FragColor  = vec4(a2 * 0.4, a2 * .4, a2 * 1, 1);
    gl_FragColor  = vec4(-a3 * 0.1 + a1 * .1, a2 * .7 - a3*.2, a2 * 1, 1);
//    gl_FragColor  = vec4(a3 * 0.1 + a1 * .1, a2 * .7 + a3 *.2, a3 * 1, 1);

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

    //    fragColor = vec4(tempColor.rgb, tran_alpha * translucency * to_edge);
    //    fragColor = vec4(tempColor, 1);
}

    #endif
