struct Vertex {
    float vx, vy, vz;
    uint8_t nx, ny, nz, nw;
    float16_t tu, tv;
};

struct Meshlet {
    // for some reason using `vec4` doesn't seem to work but using an array works
    float cone[4];
    uint data_offset;
    uint8_t vertex_count;
    uint8_t triangle_count;
};

struct MeshDraw {
    float offset[2];
    float scale[2];
};