from libtouhou import glide
from .window import Window

def create_window(title, posx, posy, width, height, frameskip):
    glide.create_window(title, posx, posy, width, height, frameskip)
    return Window()

init = glide.init
GameRenderer = glide.GameRenderer
