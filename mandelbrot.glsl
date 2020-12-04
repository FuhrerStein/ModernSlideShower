#version 430

#define definition 0

#if defined VERTEX_SHADER

void main() {
}

#elif defined GEOMETRY_SHADER
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

void emit_point(float x, float y) {
    gl_Position = vec4(x, y, 0.0, 1.0);
    EmitVertex();
}

void main() {
    emit_point(-1, -1);
    emit_point( 1, -1);
    emit_point(-1,  1);
    emit_point( 1,  1);
    EndPrimitive();
}


#elif defined FRAGMENT_SHADER

#define split_constant 67108865  // 2^26+1

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
#if (definition < 1)
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
#else
dvec2 two_vec_add_vec4(dvec2 a, dvec2 b){
    double ss, ee;
    dvec2 s, v, e;
    s = a + b;
    v = s - a;
    e = (a - (s - v)) + (b - v);
    ee = e.x + s.y;
    ss = s.x + ee;
    ee -= (ss - s.x);
    return dvec2(ss, ee);
}
#endif

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

dvec2 two_vec_prod_fma_s0(dvec2 a, dvec2 b){
    double x, y, s, e;
    x = a.x * b.x;
    y = fma(a.x, b.x, -x);
    y = fma(a.x, b.y, y);
    y = fma(a.y, b.x, y);
    y = fma(a.y, b.y, y);
    s = x + y;
    e = y - (s - x);
    return dvec2(s, e);
}

dvec2 two_vec_prod_fma_s1(dvec2 a, dvec2 b){
    double x, y, s, e;
    x = a.x * b.x;
    y = fma(a.x, b.x, -x);
    y = fma(a.x, b.y, y);
    y = fma(a.y, b.x, y);
    s = x + y;
    e = y - (s - x);
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

dvec2 vec_sqr_simple_2(dvec2 a){
    double p_x, p_y, s, e;
    p_x = a.x * a.x;
    p_y = fma(a.x, a.x, -p_x);
    p_y = fma(2 * a.x, a.y, p_y);
    p_y = fma(a.y, a.y, p_y);
    s = p_x + p_y;
    e = p_y - (s - p_x);
    return dvec2(s, e);
}

dvec2 vec_sqr_simple_3a(dvec2 a){
    double p_x, p_y, s, e;
    p_x = a.x * a.x;
    p_y = fma(a.x, a.x, -p_x);
    p_y = fma(2 * a.x, a.y, p_y);
    p_y = fma(a.y, a.y, p_y);
    s = p_x + p_y;
//    e = p_y - (s - p_x);
    return dvec2(s, e);
}

dvec2 vec_sqr_simple_4(dvec2 a){
    double p_x, p_y;
    p_x = a.x * a.x;
    p_y = fma(a.x, a.x, -p_x);
    p_y = fma(2 * a.x, a.y, p_y);
    p_y = fma(a.y, a.y, p_y);
    return dvec2(p_x, p_y);
}

dvec2 vec_sqr_simple_5(dvec2 a){
    double p_x, p_y;
    p_x = a.x * a.x;
    p_y = fma(a.x, a.x, -p_x);
    p_y = fma(2 * a.x, a.y, p_y);
    return dvec2(p_x, p_y);
}


#define prezoom 4e-3

//layout(binding=8, r32ui) uniform uimage2D histogram_texture;
layout(std430, binding = 4) buffer layoutName
{
    uint data_SSBO[2048];
};

uniform float invert_zoom;
uniform float complexity;
uniform dvec2 half_wnd_size;

uniform dvec4 mandel_x;
uniform dvec4 mandel_y;

int n;

float red, green, blue;
float a1, a2, a3;

void main() {
    n = 0;

    float final_step_sum = 0;
    float pre_final_step = 0;
    double final_step = 0;
    double a, b;

    dvec2 va = dvec2(0);
    dvec2 vb = dvec2(0);
    dvec2 va_sqr, vb_sqr;
    // todo: replace va and vb with single vector

    dvec2 coord_x_var = two_prod(gl_FragCoord.x - half_wnd_size.x, invert_zoom);
    dvec2 coord_y_var = two_prod(gl_FragCoord.y - half_wnd_size.y, invert_zoom);

    #if (definition > 0)
        coord_x_var = two_vec_add_vec4(coord_x_var, mandel_x.xy);
        coord_x_var = two_vec_add_vec4(coord_x_var, mandel_x.zw);
        coord_y_var = two_vec_add_vec4(coord_y_var, mandel_y.xy);
        coord_y_var = two_vec_add_vec4(coord_y_var, mandel_y.zw);
    #endif

    #if (definition > 1)
    double x, y, b2;
    x = coord_x_var.x;
    y = coord_y_var.x;
    #endif

    while (n++ + final_step * 0.5 < complexity)
    {

        #if (definition < 2)
            va_sqr = vec_sqr_simple_5(va);
            vb_sqr = -vec_sqr_simple_5(vb);
            vb = two_vec_prod_fma_s0(2 * va, vb);

            va = two_vec_add_vec4(va_sqr, vb_sqr);

            #if (definition == 0)
                va = two_vec_add_vec4(va, mandel_x.xy);
                va = two_vec_add_vec4(va, mandel_x.zw);
                vb = two_vec_add_vec4(vb, mandel_y.xy);
                vb = two_vec_add_vec4(vb, mandel_y.zw);
            #endif

            vb = two_vec_add_vec4(vb, coord_y_var);
            va = two_vec_add_vec4(va, coord_x_var);

            a = va.x;
            b = vb.x;
        #else

            b2 = b * b;
            b = fma(2 * a, b, y);
            a = fma(a, a, -b2) + x;

        #endif


        pre_final_step = float(final_step);
//        final_step = length(dvec2(va.x, vb.x)) * distance(va.x, vb.x);
        final_step = length(dvec2(a, b));
//        final_step = distance(va.x, vb.x);
    }

    ivec2 n_was_bigger = ivec2(0, step(n, complexity - 1));
//    n_was_bigger = ivec2(0, step(n + final_step * 0.5, complexity));
    ivec2 cluster_coord = ivec2((gl_FragCoord.xy / half_wnd_size / 2 + n_was_bigger) * 32);
    uint cluster_coord_flat = int(cluster_coord.y * 32 + cluster_coord.x);
    atomicAdd(data_SSBO[cluster_coord_flat], 1u);

    float comp_float = float(complexity);
    a1 = smoothstep(0, 40, pre_final_step * (20 - pre_final_step));
//    a1 = smoothstep(0, 40, final_step2 * (20 - final_step2));
//    a1 = smoothstep(0, 10000, final_step2 );
//    a1 = smoothstep(0, 10, final_step2 );
//    a1 = smoothstep(0, 1, sin(final_step2 * 1000) );
//    a1 = smoothstep(0, 1, sin(final_step2 * 100000) );
//    a1 = smoothstep(0, 1, cos(final_step2 * 100000) );
//    a1 = smoothstep(-1, 1, cos(final_step2 * 100000) );
//    a1 = smoothstep(-1, 1, tanh(final_step2 * 1) );
//    a1 = smoothstep(-10000, 10000, (final_step_sum * 1) ) * (pre_final_step - 10) * (float(maxIteration) - n - pre_final_step);
//    a1 = smoothstep(-10000, 10000, (final_step_sum * 1) ) * smoothstep(0, (floatmaxIt - n), pre_final_step);
    a2 = (pre_final_step - 10) * (comp_float - n - pre_final_step);
    a2 = smoothstep(-50, (comp_float - n) * (comp_float - n - 10), a2) * 2;
    a3 = smoothstep(0, (complexity - n), pre_final_step);
    gl_FragColor  = vec4(-a3 * 0.2 + a1 * .1, a2 * (.55 - .20 * a3*.25), a2 * .99, 1);
    //    gl_FragColor  = vec4(a3 * 0.1 + a1 * .1, a2 * .7 + a3 *.2, a3 * 1, 1);

//    a2 = smoothstep(0, float(final_step), complexity - n);
//    a2 = smoothstep(0, 1, complexity - n - pre_final_step);
//    gl_FragColor  = vec4(a2 * 0.4, a2 * .4, a2 * 1, 1);
}

#endif
