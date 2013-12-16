#!/usr/bin/env python
''' 
    @title ->          neuralgui.py
    @author ->         Michael Kepple
    @date ->           15 Dec 2013
    @description ->    neuralkinect.py -> Contains code for GUI & Kinect
                       functionality.
    @note ->           Requires wxPython library:
                       http://www.wxpython.org/download.php
    @python_version -> Anaconda 32-bit (2.7)
    @usage ->          python neuralgui.py
'''
import wx, pygame
import threading
import thread
import itertools
import ctypes
from pykinect import nui
from pykinect.nui import JointId
from pygame.color import THECOLORS

skeleton_to_depth_image = nui.SkeletonEngine.skeleton_to_depth_image
SKELETON_COLORS = [THECOLORS["red"],
                   THECOLORS["blue"],
                   THECOLORS["green"],
                   THECOLORS["orange"],
                   THECOLORS["purple"],
                   THECOLORS["yellow"],
                   THECOLORS["violet"]]

LEFT_ARM = (JointId.ShoulderCenter,
            JointId.ShoulderLeft,
            JointId.ElbowLeft,
            JointId.WristLeft,
            JointId.HandLeft)
RIGHT_ARM = (JointId.ShoulderCenter,
             JointId.ShoulderRight,
             JointId.ElbowRight,
             JointId.WristRight,
             JointId.HandRight)
LEFT_LEG = (JointId.HipCenter,
            JointId.HipLeft,
            JointId.KneeLeft,
            JointId.AnkleLeft,
            JointId.FootLeft)
RIGHT_LEG = (JointId.HipCenter,
             JointId.HipRight,
             JointId.KneeRight,
             JointId.AnkleRight,
             JointId.FootRight)
SPINE = (JointId.HipCenter,
         JointId.Spine,
         JointId.ShoulderCenter,
         JointId.Head)

pygame.init()
KINECTEVENT = pygame.USEREVENT
full_screen = False
draw_skeleton = True
video_display = True
screen_lock = thread.allocate()
skeletons = None

# recipe to get address of surface: http://archives.seul.org/pygame/users/Apr-2008/msg00218.html
if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
    Py_ssize_t = ctypes.c_int
elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
    Py_ssize_t = ctypes.c_int64
else:
    raise TypeError("Cannot determine type of Py_ssize_t")
_PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
_PyObject_AsWriteBuffer.restype = ctypes.c_int
_PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                                  ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(Py_ssize_t)]

class PygameDisplay(wx.Window):   
    def __init__(self, parent, ID):        
        wx.Window.__init__(self, parent, ID)
        self.parent = parent
        self.hwnd = self.GetHandle()
        self.size = self.GetSizeTuple()
        self.size_dirty = True
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_TIMER, self.Update, self.timer)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.fps = 60.0
        self.timespacing = 1000.0 / self.fps
        self.timer.Start(self.timespacing, False)
        self.linespacing = 5
    def Update(self, event):
        self.Redraw()
    def Redraw(self):
        if self.size_dirty:
            self.screen = kinectScreen
            self.size_dirty = False
        s = pygame.image.tostring(kinectScreen, 'RGB')
        img = wx.ImageFromData(640, 480, s)
        bmp = wx.BitmapFromImage(img)
        dc = wx.ClientDC(self)
        dc.DrawBitmap(bmp, 0, 0, False)
        del dc
    def OnPaint(self, event):
        self.Redraw()
        event.Skip()
    def OnSize(self, event):
        self.size = self.GetSizeTuple()
        self.size_dirty = True
    def Kill(self, event):
        self.Unbind(event=wx.EVT_PAINT, handler=self.OnPaint)
        self.Unbind(event=wx.EVT_TIMER, handler=self.Update, source=self.timer)
 
class Frame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, size=(640, 480))
        kinectDisplay = PygameDisplay(self, -1)
        self.display = kinectDisplay
        self.statusbar = self.CreateStatusBar()
        self.statusbar.SetFieldsCount(3)
        self.statusbar.SetStatusWidths([-3, -4, -2])
        self.statusbar.SetStatusText("NeuralKinect", 0)
        self.statusbar.SetStatusText("Letter Detected: ", 1)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_CLOSE, self.Kill)
        self.curframe = 0
        self.SetTitle("NeuralKinect Gesture Recognition")
        self.slider = wx.Slider(self, wx.ID_ANY, 1000, 1, 10000, style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider.SetTickFreq(0.1, 1)
        self.button = wx.Button(self, -1, "Train Network")
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_SCROLL, self.OnScroll)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_TIMER, self.Update, self.timer)
        self.Bind(wx.EVT_BUTTON, self.ButtonClick, self.button)
        self.timer.Start((1000.0 / self.display.fps))
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer2.Add(self.slider, 1, flag=wx.EXPAND | wx.RIGHT, border=5)
        self.sizer2.Add(self.button, 0, flag=wx.EXPAND | wx.ALL, border=5)
        self.sizer.Add(self.sizer2, 0, flag=wx.EXPAND)
        self.sizer.Add(self.display, 1, flag=wx.EXPAND)
        self.SetAutoLayout(True)
        self.SetSizer(self.sizer)
        self.Layout()
    def Kill(self, event):
        self.display.Kill(event)
        self.Destroy()
    def OnSize(self, event):
        self.Layout()
    def Update(self, event):
        self.curframe += 1
        self.statusbar.SetStatusText("Michael Kepple", 2)
    def OnScroll(self, event):
        self.display.linespacing = self.slider.GetValue()
    def ButtonClick(self, event):        
        self.button.SetLabel("Training...!")
        self.Layout()
 
class App(wx.App):
    def OnInit(self):
        self.frame = Frame(parent=None)
        self.frame.Show()
        self.SetTopWindow(self.frame)
        return True

def draw_skeleton_data(pSkelton, index, positions, width=4):
    start = pSkelton.SkeletonPositions[positions[0]]
    for position in itertools.islice(positions, 1, None):
        nextPos = pSkelton.SkeletonPositions[position.value]
        curstart = skeleton_to_depth_image(start, pygame.display.Info().current_w, pygame.display.Info().current_h) 
        curend = skeleton_to_depth_image(nextPos, pygame.display.Info().current_w, pygame.display.Info().current_h)
        pygame.draw.line(kinectScreen, SKELETON_COLORS[index], curstart, curend, width)
        start = nextPos 
        
def draw_skeletons(skeletons):
    for index, data in enumerate(skeletons):
        HeadPos = skeleton_to_depth_image(data.SkeletonPositions[JointId.Head], pygame.display.Info().current_w, pygame.display.Info().current_h) 
        draw_skeleton_data(data, index, SPINE, 10)
        pygame.draw.circle(kinectScreen, SKELETON_COLORS[index], (int(HeadPos[0]), int(HeadPos[1])), 20, 0)
        draw_skeleton_data(data, index, LEFT_ARM)
        draw_skeleton_data(data, index, RIGHT_ARM)
        draw_skeleton_data(data, index, LEFT_LEG)
        draw_skeleton_data(data, index, RIGHT_LEG)

def depth_frame_ready(frame):
    if video_display:
        return
    with screen_lock:
        address = surface_to_array(kinectScreen)
        frame.image.copy_bits(address)
        del address
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        kinectDisplay.Update(None)
            
def video_frame_ready(frame):
    if not video_display:
        return
    with screen_lock:
        address = surface_to_array(kinectScreen)
        frame.image.copy_bits(address)
        del address
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        kinectDisplay.Update(None)

def surface_to_array(surface):
    buffer_interface = surface.get_buffer()
    address = ctypes.c_void_p()
    size = Py_ssize_t()
    _PyObject_AsWriteBuffer(buffer_interface,
                        ctypes.byref(address), ctypes.byref(size))
    byt = (ctypes.c_byte * size.value).from_address(address.value)
    byt.object = buffer_interface
    return byt
 
def kinectLoop():
    kinect = nui.Runtime()
    kinect.skeleton_engine.enabled = True
    def post_frame(frame):
        try:
            pygame.event.post(pygame.event.Event(KINECTEVENT, skeletons=frame.SkeletonData))
        except:
            pass
    kinect.skeleton_frame_ready += post_frame    
    kinect.depth_frame_ready += depth_frame_ready    
    kinect.video_frame_ready += video_frame_ready    
    kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)
    # main game loop
    done = False
    while not done:
        e = pygame.event.wait()
        dispInfo = pygame.display.Info()
        print(dispInfo)
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == KINECTEVENT:
            skeletons = e.skeletons
            if draw_skeleton:
                draw_skeletons(skeletons)
                kinectDisplay.Update(e)

# False -> don't redirect stderror
app = App(False)
myFrame = Frame(parent=None)
kinectDisplay = PygameDisplay(myFrame, -1)
kinectScreen = pygame.Surface((640, 480), 0, 32)

if __name__ == "__main__":
    kinectThread = threading.Thread(target=kinectLoop).start()
    app.MainLoop()
