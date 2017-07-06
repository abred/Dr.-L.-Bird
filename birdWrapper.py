# from ctypes import *
# from ctypes import cdll
import ctypes
lib = ctypes.cdll.LoadLibrary('cffi/libbirdwrap.so')


lib.Wrapper_new.argtypes = []
lib.Wrapper_new.restype = ctypes.c_void_p

lib.Wrapper_processScreenShot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.Wrapper_processScreenShot.restype = None

lib.Wrapper_preprocessDataForNN.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.Wrapper_preprocessDataForNN.restype = None

lib.Wrapper_findSlingshotCenter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.Wrapper_findSlingshotCenter.restype = ctypes.c_int

lib.Wrapper_calcLives.argtypes = [ctypes.c_void_p]
lib.Wrapper_calcLives.restype = ctypes.c_int

lib.Wrapper_getCurrScore.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lib.Wrapper_getCurrScore.restype = ctypes.c_int

lib.Wrapper_getEndScore.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.Wrapper_getEndScore.restype = ctypes.c_int


class Wrapper(object):
    def __init__(self):
        print("new wrapper")
        self.obj = lib.Wrapper_new()
        print(self.obj)

    def calcLives(self):
        return lib.Wrapper_calcLives(self.obj)

    def processScreenShot(self, input, output, w, h):
        lib.Wrapper_processScreenShot(self.obj, input, output, w, h)

    def findSlingshotCenter(self, scene, width, height):
        return lib.Wrapper_findSlingshotCenter(self.obj, scene, width, height)

    def getCurrScore(self, input, output, w, h):
        return lib.Wrapper_getCurrScore(self.obj, input, output, w, h)

    def getEndScore(self, input, w, h, threshold):
        return lib.Wrapper_getEndScore(self.obj,
                                       input, w, h, threshold)

    def preprocessDataForNN(self,input, output, w, h):
        lib.Wrapper_preprocessDataForNN(self.obj, input, output, w, h)




# class BirdWrap(object):
#     def __init__(self):
#         self.obj = lib.Foo_new()

#     def bar(self):
#         lib.Foo_bar(self.obj)
