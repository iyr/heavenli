#version 100
precision  mediump float;

uniform    sampler2D tex;

varying    vec2 texCoord;
varying    vec4 color;   

void main() {
   gl_FragColor = texture2D(tex, texCoord)*color;
}
