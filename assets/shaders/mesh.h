struct Vertex {
    float vx, vy, vz, vw;
    uint8_t nx, ny, nz, nw;
    float tu, tv;
};

struct Meshlet {
    // for some reason using `vec4` doesn't seem to work but using an array works
    float cone[4];
    uint vertices[64];
    uint8_t indices[126 * 3];
    uint8_t triangleCount;
    uint8_t vertexCount;
};