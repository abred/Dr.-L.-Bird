from ctypes import cdll
lib = cdll.LoadLibrary('cffi/libbirdwrap.so')


def processScreenShot(input, output, w, h):
    return lib.processScreenShot(input, output, w, h)


def findSlingshotCenter(scene, width, height):
    return lib.findSlingshotCenter(scene, width, height)


def getCurrScore(input, output, w, h):
    return lib.getCurrScore(input, output, w, h)


def getEndScore(input, output, w, h):
    return lib.getEndScore(input, output, w, h)


def preprocessDataForNN(input, output, w, h):
    return lib.preprocessDataForNN(input, output, w, h)




# class BirdWrap(object):
#     def __init__(self):
#         self.obj = lib.Foo_new()

#     def bar(self):
#         lib.Foo_bar(self.obj)
