#version 330

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

#define split_constant 67108865   // 2^26+1

//vec2 two_sum(float a, float b){
//    float s = a + b;
//    float v = s - a;
//    float e = (a - (s - v)) + (b - v);
//    return vec2(s, e);
//}

//vec2 quick_two_sum(float a, float b){
//    float s = a + b;
//    float e = b - (s - a);
//    return vec2(s, e);
//}

//vec4 two_sum_comp(vec2 a, vec2 b){
//    vec2 s = a + b;
//    vec2 v = s - a;
//    vec2 e = (a - (s - v)) + (b - v);
//    return vec4(s.x, e.x, s.y, e.y);
//}

//vec2 two_vec_add(vec2 a, vec2 b){
//    vec2 s, t;
//    s = two_sum(a.x, b.x);
//    t = two_sum(a.y, b.y);
//    s.y += t.x;
//    s = quick_two_sum(s.x, s.y);
//    s.y += t.y;
//    s = quick_two_sum(s.x, s.y);
//    return s;
//}

//vec2 two_vec_add_vec(vec2 a, vec2 b){
//    vec4 st = two_sum_comp(a, b);
//    st.y += st.z;
//    st.xy = quick_two_sum(st.x, st.y);
//    st.y += st.w;
//    st.xy = quick_two_sum(st.x, st.y);
//    return st.xy;
//}

//vec2 two_vec_add_vec2(vec2 a, vec2 b){
//    float s, e;
//    vec4 st = two_sum_comp(a, b);
//    st.y += st.z;
//    s = st.x + st.y;
//    st.y = st.y - (s - st.x);
//    st.x = s;
//    st.y += st.w;
//    s = st.x + st.y;
//    e = st.y - (s - st.x);
//    return vec2(s, e);
//
////    return st.xy;
//}

//vec4 split_comp(vec2 c){
//    vec2 t = c * split_constant;
//    vec2 c_hi = t - (t - c);
//    vec2 c_lo = c - c_hi;
//    return vec4(c_hi.x, c_lo.x, c_hi.y, c_lo.y);
//}

//vec2 two_prod(float a, float b){
//    float p = a * b;
//    vec2 a_s = split(a);
//    vec2 b_s = split(b);
//    float err1 = a_s.x * b_s.x - p;
//    float err2 = err1 + a_s.y * b_s.x + a_s.x * b_s.y;
//    float err = a_s.y * b_s.y + err2;
//    return vec2 (p, err);
//}


//vec2 two_prod_fma(float a, float b){
//    float x = a * b;
//    float y = fma(a, b, -x);
//    return vec2 (x, y);
//}
//
//vec2 two_vec_prod(vec2 a, vec2 b){
//    vec2 p = two_prod(a.x, b.x);
//    p.y += a.x * b.y;
//    p.y += a.y * b.x;
//    p = quick_two_sum(p.x, p.y);
//    return p;
//}
//
//vec2 two_vec_prod_fma(vec2 a, vec2 b){
//    vec2 p = two_prod_fma(a.x, b.x);
//    p.y += a.x * b.y;
//    p.y += a.y * b.x;
//    p = quick_two_sum(p.x, p.y);
//    return p;
//}

//vec2 two_vec_prod_fma_s(vec2 a, vec2 b){
//    float x = a.x * b.x;
//    float y = fma(a.x, b.x, -x);
//    y += a.x * b.y;
//    y += a.y * b.x;
//    float s = x + y;
//    float e = y - (s - x);
//    return vec2(s, e);
//}

//vec2 two_vec_prod_fma_s1(vec2 a, vec2 b){
//    float x, y, s, e;
//    x = a.x * b.x;
//    y = fma(a.x, b.x, -x);
//    y = fma(a.x, b.y, y);
//    y = fma(a.y, b.x, y);
//    s = x + y;
//    e = y - (s - x);
//    return vec2(s, e);
//}

//vec2 single_sqr(float a){
//    float p = a * a;
//    vec2 a_s = split(a);
//    float err = ((a_s.x * a_s.x - p) + 2 * a_s.x * a_s.y) + a_s.y * a_s.y;
//    return vec2(p, err);
//}

//vec2 single_sqr_fma(float a){
//    float x = a * a;
//    float y = fma(a, a, -x);
//    return vec2 (x, y);
//}


//vec2 vec_sqr(vec2 a){
//    vec2 p = single_sqr(a.x);
//    p.y += 2 * a.x * a.y;
//    p = quick_two_sum(p.x, p.y);
//    return p;
//}

//vec2 vec_sqr_fma(vec2 a){
//    float x, y, s, e;
//    x = a.x * a.x;
//    y = fma(a.x, a.x, -x);
//    y = fma(a.x, 2 * a.y, y);
//    s = x + y;
//    e = y - (s - x);
//    return vec2(s, e);
//}

//vec2 vec_sqr_simple_1(vec2 a){
//    float p_x, p_y, s, e, t, a_hi_hi, a_hi_lo;
//    t = a.x * split_constant;
//    a_hi_hi = t - (t - a.x);
//    a_hi_lo = a.x - a_hi_hi;
//    p_x = a.x * a.x;
//    p_y = ((a_hi_hi * a_hi_hi - p_x) + 2 * a_hi_hi * a_hi_lo) + a_hi_lo * a_hi_lo;
//    p_y += 2 * a.x * a.y;
//    s = p_x + p_y;
//    e = p_y - (s - p_x);
//    return vec2(s, e);
//}

//vec2 vec_sqr_simple_2(vec2 a){
//    float p_x, p_y, s, e;
//    p_x = a.x * a.x;
//    p_y = fma(a.x, a.x, -p_x);
//    p_y = fma(2 * a.x, a.y, p_y);
//    p_y = fma(a.y, a.y, p_y);
//    s = p_x + p_y;
//    e = p_y - (s - p_x);
//    return vec2(s, e);
//}

//vec2 vec_sqr_simple_3a(vec2 a){
//    float p_x, p_y, s, e;
//    p_x = a.x * a.x;
//    p_y = fma(a.x, a.x, -p_x);
//    p_y = fma(2 * a.x, a.y, p_y);
//    p_y = fma(a.y, a.y, p_y);
//    s = p_x + p_y;
////    e = p_y - (s - p_x);
//    return vec2(s, e);
//}

//vec2 vec_sqr_simple_4(vec2 a){
//    float p_x, p_y;
//    p_x = a.x * a.x;
//    p_y = fma(a.x, a.x, -p_x);
//    p_y = fma(2 * a.x, a.y, p_y);
//    p_y = fma(a.y, a.y, p_y);
//    return vec2(p_x, p_y);
//}


#if (definition < 1)
vec2 two_vec_add_vec4(vec2 a, vec2 b){
    float ss, ee;
    vec2 s, v, e;
    s = a + b;
    v = s - a;
    e = (a - (s - v)) + (b - v);
    e.x += s.y;
    ss = s.x + e.x;
    e.x = e.x - (ss - s.x);
    s.x = ss;
    e.x += e.y;
    ss += e.x;
    ee = e.x - (ss - s.x);
    return vec2(ss, ee);
}
#else
vec2 two_vec_add_vec4(vec2 a, vec2 b){
    float ss, ee;
    vec2 s, v, e;
    s = a + b;
    v = s - a;
    e = (a - (s - v)) + (b - v);
    ee = e.x + s.y;
    ss = s.x + ee;
    ee -= (ss - s.x);
    return vec2(ss, ee);
}
#endif

vec2 split(float a){
    float t = a * split_constant;
    float a_hi = t - (t - a);
    float a_lo = a - a_hi;
    return vec2(a_hi, a_lo);
}

vec2 two_prod(float a, float b){
    float p = a * b;
    vec2 a_s = split(a);
    vec2 b_s = split(b);
    float err = ((a_s.x * b_s.x - p) + a_s.x * b_s.y + a_s.y * b_s.x) + a_s.y * b_s.y;
    return vec2(p, err);
}

vec2 two_vec_prod_fma_s0(vec2 a, vec2 b){
    float x, y, s, e;
    x = a.x * b.x;
    y = fma(a.x, b.x, -x);
    y = fma(a.x, b.y, y);
    y = fma(a.y, b.x, y);
    y = fma(a.y, b.y, y);
    s = x + y;
    e = y - (s - x);
    return vec2(s, e);
}

vec2 vec_sqr_simple_5(vec2 a){
    float p_x, p_y;
    p_x = a.x * a.x;
    p_y = fma(a.x, a.x, -p_x);
    p_y = fma(2 * a.x, a.y, p_y);
    return vec2(p_x, p_y);
}


#define prezoom 4e-3

uniform float invert_zoom;
uniform float complexity;
uniform vec2 half_wnd_size;

uniform vec4 mandel_x;
uniform vec4 mandel_y;

int n;

out vec4 FragColor;

float red, green, blue;
float a1, a2, a3;

void main() {
    n = 0;

    float final_step_sum = 0;
    float pre_final_step = 0;
    float final_step = 0;
    float a, b;

    vec2 va = vec2(0);
    vec2 vb = vec2(0);
    vec2 va_sqr, vb_sqr;
    // todo: replace va and vb with single vector

    vec2 coord_x_var = two_prod(gl_FragCoord.x - half_wnd_size.x, invert_zoom);
    vec2 coord_y_var = two_prod(gl_FragCoord.y - half_wnd_size.y, invert_zoom);

    #if (definition > 0)
        coord_x_var = two_vec_add_vec4(coord_x_var, mandel_x.xy);
        coord_x_var = two_vec_add_vec4(coord_x_var, mandel_x.zw);
        coord_y_var = two_vec_add_vec4(coord_y_var, mandel_y.xy);
        coord_y_var = two_vec_add_vec4(coord_y_var, mandel_y.zw);
    #endif

    #if (definition > 1)
    float x, y, b2;
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
        final_step = length(vec2(a, b));
    }

    ivec2 n_was_bigger = ivec2(0, step(n, complexity - 1));
    ivec2 cluster_coord = ivec2((gl_FragCoord.xy / half_wnd_size / 2 + n_was_bigger) * 32);
    uint cluster_coord_flat = uint(cluster_coord.y * 32 + cluster_coord.x);
//     atomicAdd(data_SSBO[cluster_coord_flat], 1u);
//     atomicAdd(mandel_stat_texture[cluster_coord_flat], 1u);
//     imageAtomicAdd(mandel_stat_texture, cluster_coord, 1u);


    float comp_float = float(complexity);
    a1 = smoothstep(0, 40, pre_final_step * (20 - pre_final_step));
    a2 = (pre_final_step - 10) * (comp_float - n - pre_final_step);
    a2 = smoothstep(-50, (comp_float - n) * (comp_float - n - 10), a2) * 2;
    a3 = smoothstep(0, (complexity - n), pre_final_step);
    FragColor  = vec4(-a3 * 0.2 + a1 * .1, a2 * (.55 - .20 * a3*.25), a2 * .99, 1);
}

#endif
