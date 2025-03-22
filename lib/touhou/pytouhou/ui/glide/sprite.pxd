from pytouhou.game.sprite cimport Sprite

cdef struct RenderingData:
    float pos[12]
    float left, right, bottom, top
    unsigned char color[4]

cdef void render_sprite(Sprite sprite) nogil
