import pytouhou.lib.gui as gui
import libtouhou

cdef class Window(gui.Window):
    cdef void present(self) nogil:
        with gil:
            libtouhou.glide.buffer_swap()

    cdef void set_window_size(self, int width, int height) nogil:
        pass

    cdef void set_swap_interval(self, int interval) except *:
        pass

    cdef list get_events(self):
        return []

    cdef int get_keystate(self) nogil:
        return 0

    cdef void toggle_fullscreen(self) nogil:
        pass
