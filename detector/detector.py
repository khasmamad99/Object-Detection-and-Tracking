import random
from ctypes import *
import numpy as np
import cv2 as cv



def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

def array_to_image(arr):
    # need to return old values to avoid python freeing memory
    arr = arr.transpose(2,0,1)
    c = arr.shape[0]
    h = arr.shape[1]
    w = arr.shape[2]
    arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
    data = arr.ctypes.data_as(POINTER(c_float))
    im = IMAGE(w,h,c,data)
    return im, arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]
                


class Detector:
    def __init__(self, cfg="detector/config/yolov3-tiny.cfg",
                       weights="detector/config/yolov3-tiny.weights", 
                       data="detector/config/coco.data", 
                       names="detector/config/coco.names",
                       gpu=True):
        self.init_functions(gpu)
        self.net = self.load_net_custom(cfg.encode("ascii"), weights.encode("ascii"), 0, 1)
        self.meta = self.load_meta(data.encode("ascii"))
        self.colors = self.init_colors(names)

    
    def detect(self, img, once=True, thresh=0.5, hier_thresh=.5, nms=.45):
        if type(img) is str:
            img = self.load_image(img.encode("ascii"), 0, 0)
        else:
            img, _ = array_to_image(img)

        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, img)
        letter_box = 0
        dets = self.get_network_boxes(self.net, img.w, img.h, thresh, hier_thresh, None, 0, pnum, letter_box)
        num = pnum[0]
        self.do_nms_sort(dets, num, self.meta.classes, nms)
        res = []
        for j in range(num):
            for i in range(self.meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    x = b.x - b.w / 2
                    y = b.y - b.h / 2
                    label = self.meta.names[i].decode("ascii")
                    res.append({"label":label, "bbox":[x, y, b.w, b.h], "perc":dets[j].prob[i]})

        self.free_detections(dets, num)
        if once:
            self.free_image(img)

        return res

    
    def visualize(self, img, dets=None, once=True, thresh=0.5, show=False):
        if dets == None:
            dets = self.detect(img, once=once, thresh=thresh)

        if type(img) is str:
            img = cv.imread(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        font = cv.FONT_HERSHEY_DUPLEX
        font_scale = 0.7
        thickness = 0.8
        line = cv.LINE_AA
        margin = 10
        print(self.colors)

        for det in dets:
            color = self.colors[det["label"]]
            print(type(color[0]))
            x, y, w, h = [int(a) for a in det["bbox"]]
            print(x, y, w, h)
            img = cv.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness, line)
            text = det["label"] + " " + str(det["perc"])[:4]
            text_width, text_height = cv.getTextSize(text, font, font_scale, thickness)[0]
            img = cv.rectangle(img, (x, y+text_height+margin), (x+text_width+margin, y), color, thickness, cv.FILLED)
            img = cv.putText(img, text, (x+margin/2, y-margin/2), font, font_scale, (0,0,0), thickness, line)

        if show:
            cv.imshow("", img)
            cv.waitKey(0)
        
        return img
            
        

    def init_colors(self, names):
        with open(names, 'r') as file:
            classes = file.readlines()
            classes = [class_name.strip("\n") for class_name in classes]
            no_of_classes = len(classes)

        colors = {}
        r = int(random.random() * 256)
        g = int(random.random() * 256)
        b = int(random.random() * 256)
        step = 256 / no_of_classes
        for i in range(no_of_classes):
            r += step
            g += step
            b += step
            r = int(r) % 256
            g = int(g) % 256
            b = int(b) % 256
            colors[classes[i]] = (r,g,b)
        
        return colors
    

    def init_functions(self, gpu=True):
        if gpu:
            lib = CDLL("detector/config/libdarknet_gpu.so", RTLD_GLOBAL)
            set_gpu = lib.cuda_set_device
            set_gpu.argtypes = [c_int]
        else:
            lib = CDLL("detector/config/libdarknet_cpu.so", RTLD_GLOBAL)
        
        self.predict = lib.network_predict_ptr
        self.predict.argtypes = [c_void_p, POINTER(c_float)]
        self.predict.restype = POINTER(c_float)

        self.copy_image_from_bytes = lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE,c_char_p]
        
        # self.network_width = lib.network_width(self.net)
        # self.network_height = lib.network_height(self.net)

        self.get_network_boxes = lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.network_predict = lib.network_predict_ptr
        self.network_predict.argtypes = [c_void_p, POINTER(c_float)]

        self.load_net = lib.load_network
        self.load_net.argtypes = [c_char_p, c_char_p, c_int]
        self.load_net.restype = c_void_p

        self.load_net_custom = lib.load_network_custom
        self.load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        self.load_net_custom.restype = c_void_p

        self.do_nms_sort = lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.free_image = lib.free_image
        self.free_image.argtypes = [IMAGE]

        self.load_meta = lib.get_metadata
        lib.get_metadata.argtypes = [c_char_p]
        lib.get_metadata.restype = METADATA

        self.load_image = lib.load_image_color
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        self.predict_image = lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)


det = Detector()
det.visualize("/home/khasmamad/Desktop/darknet/data/dog.jpg", show=True)