import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")
from utils import label_map_util
from utils import visualization_utils as vis_util
from tkinter import *
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')
NUM_CLASSES = 57
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

class App:
                
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        # open video source
        self.vid = MyVideoCapture(video_source)
        
        # Create a canvas that can fit the above video source size
        self.canvas = Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        self.botFrame = Frame(window)
        self.botFrame.pack(side=BOTTOM)

        imgpath = "GUI\\empty.png"
        img = PIL.Image.open(imgpath)
        img = img.resize((100,100), PIL.Image.ANTIALIAS)
        img = PIL.ImageTk.PhotoImage(img)

        self.StopLabelText = Label(self.botFrame, text = "")
        self.StopLabelText.grid(row = 0, column = 0, sticky='nsew')
        self.StopLabel = Label(self.botFrame, text = "StopLabel", image=img)
        self.StopLabel.grid(row = 1, column = 0, sticky='nsew')

        self.SpeedLabelText = Label(self.botFrame, text = "Ograniczenie Prędkości:")
        self.SpeedLabelText.grid(row = 0, column = 1, sticky='nsew')
        self.SpeedLabel = Label(self.botFrame, text = "SpeedLabel", image=img)
        self.SpeedLabel.grid(row = 1, column = 1, sticky='nsew')

        self.ZakazLabelText = Label(self.botFrame, text = "Zakazy")
        self.ZakazLabelText.grid(row = 0, column = 2, sticky='nsew')
        self.ZakazLabel = Label(self.botFrame, text = "ZakazLabel", image=img)
        self.ZakazLabel.grid(row = 1, column = 2, sticky='nsew')

        self.NakazLabelText = Label(self.botFrame, text = "Nakazy")
        self.NakazLabelText.grid(row = 0, column = 3, sticky='nsew')
        self.NakazLabel = Label(self.botFrame, text = "NakazLabel", image=img)
        self.NakazLabel.grid(row = 1, column = 3, sticky='nsew')

        self.UwagaLabelText = Label(self.botFrame, text = "Uwaga")
        self.UwagaLabelText.grid(row = 0, column = 4, sticky='nsew')
        self.UwagaLabel = Label(self.botFrame, text = "UwagaLabel", image=img)
        self.UwagaLabel.grid(row = 1, column = 4, sticky='nsew')

        self.InfoLabelText = Label(self.botFrame, text = "Info")
        self.InfoLabelText.grid(row = 0, column = 5, sticky='nsew')
        self.InfoLabel = Label(self.botFrame, text = "InfoLabel", image=img)
        self.InfoLabel.grid(row = 1, column = 5, sticky='nsew')
        
        #self.canvas.create_window(100,100, window=label1)
        
        
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()
        
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            frame_expanded = np.expand_dims(frame, axis=0)
            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})
            TempClasses = classes
            # Draw the results of the detection (aka 'visulaize the results')
            frame = vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)
            
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame[0]))
            self.canvas.create_image(0,0,image=self.photo, anchor=tkinter.NW)
        self.window.after(self.delay, self.update)

        if(frame[1].startswith("speed")):
            imgpath = "GUI\speed"+frame[1][-2:]+".png"
            img = PIL.Image.open(imgpath)
            img = img.resize((100,100), PIL.Image.ANTIALIAS)
            img = PIL.ImageTk.PhotoImage(img)
            self.SpeedLabel.configure(image = img)
            self.SpeedLabel.image = img
        if(frame[1].startswith("koniec_zakazu")):
            imgpath = "GUI\koniec_zakazu.png"
            img = PIL.Image.open(imgpath)
            img = img.resize((100,100), PIL.Image.ANTIALIAS)
            img = PIL.ImageTk.PhotoImage(img)
            self.SpeedLabel.configure(image = img)
            self.SpeedLabel.image = img
            
        if(frame[1].startswith("zakaz_")):
            imgpath = "GUI\\"+frame[1]+".png"
            img = PIL.Image.open(imgpath)
            img = img.resize((100,100), PIL.Image.ANTIALIAS)
            img = PIL.ImageTk.PhotoImage(img)
            self.ZakazLabel.configure(image = img)
            self.ZakazLabel.image = img
            
        if(frame[1].startswith("nakaz_")):
            imgpath = "GUI\\"+frame[1]+".png"
            img = PIL.Image.open(imgpath)
            img = img.resize((100,100), PIL.Image.ANTIALIAS)
            img = PIL.ImageTk.PhotoImage(img)
            self.NakazLabel.configure(image = img)
            self.NakazLabel.image = img
            
        if(frame[1].startswith("stop") or frame[1] == "inne_niebezpieczenstwo"):
            imgpath = "GUI\stop.png"
            img = PIL.Image.open(imgpath)
            img = img.resize((100,100), PIL.Image.ANTIALIAS)
            img = PIL.ImageTk.PhotoImage(img)
            self.StopLabel.configure(image = img)
            self.StopLabel.image = img

        if(frame[1]=="piciong" or frame[1].startswith("stromo") or frame[1].startswith("skret") or frame[1].startswith("przejscie") or frame[1]=="zakrety_zakrety" or frame[1]=="roboty" or frame[1]=="przejazd_kol_z_zaporami" or frame[1]=="dzieci"):
            imgpath = "GUI\\"+frame[1]+".png"
            img = PIL.Image.open(imgpath)
            img = img.resize((100,100), PIL.Image.ANTIALIAS)
            img = PIL.ImageTk.PhotoImage(img)
            self.UwagaLabel.configure(image = img)
            self.UwagaLabel.image = img

        if(frame[1] == "droga_dla_rowerow" or frame[1] == "droga_ekspresowa" or frame[1] == "rondo" or frame[1] == "zawracanie" ):
            imgpath = "GUI\\"+frame[1]+".png"
            img = PIL.Image.open(imgpath)
            img = img.resize((100,100), PIL.Image.ANTIALIAS)
            img = PIL.ImageTk.PhotoImage(img)
            self.InfoLabel.configure(image = img)
            self.InfoLabel.image = img
        
class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
        self.window.mainloop()
        
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
