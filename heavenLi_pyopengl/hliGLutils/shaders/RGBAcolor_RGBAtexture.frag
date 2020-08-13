#version 100
precision  mediump float;
varying    vec2 texCoord;
varying    vec4 color;   
uniform    sampler2D tex;
void main() {
   gl_FragColor = texture2D(tex, texCoord)*color;
}
