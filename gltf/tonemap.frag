#version 120

uniform sampler2D tex;

varying vec2 v_texcoord;

void main() {
    vec3 color = texture2D(tex, v_texcoord).rgb;

    color = max(vec3(0.0), color - vec3(0.004));
    color = (color * (vec3(6.2) * color + vec3(0.5))) / (color * (vec3(6.2) * color + vec3(1.7)) + vec3(0.06));

    gl_FragColor = vec4(color, 1.0);
}
