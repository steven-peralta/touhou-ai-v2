cimport pytouhou.lib.gui as gui

cdef class Window(gui.Window):
    cdef void present(self) nogil
    cdef void set_window_size(self, int width, int height) nogil
    cdef void set_swap_interval(self, int interval) except *
    cdef list get_events(self)
    cdef int get_keystate(self) nogil
    cdef void toggle_fullscreen(self) nogil
