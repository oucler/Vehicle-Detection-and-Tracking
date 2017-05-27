import os,glob 
import cv2
import numpy as np
import skimage
from skimage import data, color, exposure
from scipy.ndimage.measurements import label
import matplotlib.pylab as plt
import matplotlib.image as mpimg
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

class vehicleDetectionTracking:
    def __init__(self,args):
        self.vehicle_dir = args.vehicle_dir 
        self.nvehicle_dir = args.nvehicle_dir
        self.batch_size = args.batch_size
        self.nb_epoch = args.nb_epoch
        self.model_name = args.model_name
        self.cars = []
        self.non_cars = []
        self.X = []
        self.y = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.vehicle_shape = None
        vehicleDetectionTracking.parse_dir(self)
        vehicleDetectionTracking.read_images(self)
        self.cross_validation()
    def parse_dir(self):
        self.cars = glob.glob(self.vehicle_dir+"/*/*.png")
        self.non_cars = glob.glob(self.nvehicle_dir+"/*/*.png")
        self.y = np.concatenate([np.ones(len(self.cars)), np.zeros(len(self.non_cars))-1])
    def read_images(self):
        for name in self.cars:    
            self.X.append(skimage.io.imread(name))
        for name in self.non_cars:    
            self.X.append(skimage.io.imread(name))
        self.X = np.array(self.X)
    
    def cross_validation(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.10, random_state=42)
        self.X_train = self.X_train.astype('float32')
        self.X_test = self.X_test.astype('float32')
        print ("Training Data Size = {}".format(self.X_train.shape[0]))
        print ("Test Data Size = {}".format(self.X_test.shape[0]))
        print ("Shape of image = {}".format(self.X_train[0].shape))
        self.vehicle_shape = self.X_train[0].shape
    def model(self,input_shape=(64,64,3), filename=None):
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
        model.add(Convolution2D(10, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))
        model.add(Convolution2D(10, 3, 3, activation='relu', name='conv2',border_mode="same"))
        model.add(MaxPooling2D(pool_size=(8,8)))
        model.add(Dropout(0.5))
        model.add(Convolution2D(128,8,8,activation="relu",name="dense1")) # This was Dense(128)
        model.add(Dropout(0.5))
        model.add(Convolution2D(1,1,1,name="dense2", activation="tanh")) # This was Dense(1)
        if filename:
            model.load_weights(filename) 
        model.summary()
        return model
    def run_save_model(self):
        model = self.model()
        model.add(Flatten())
        model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=1, validation_data=(self.X_test, self.y_test))
        score = model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        model.save_weights(self.model_name)        
    def add_heat(self,heatmap, x,y):
        # Iterate through list of bboxes
        for i,j in zip(x,y):
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            if (i > 100 and (j < 54 and j >43)):
                heatmap[j*8:j*8+64, i*8:i*8+64] += 1
        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
    def apply_threshold(self,heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap
    
    def draw_labeled_bboxes_test(self,img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img
    def plot_image(self,orig_img,draw_img,heatmap):
        fig = plt.figure(figsize=(15,10))
        plt.subplot(131)
        plt.imshow(orig_img)
        plt.title("Original Image")
        plt.subplot(132)
        plt.imshow(draw_img)
        plt.title('Car Positions')
        plt.subplot(132)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')
        fig.tight_layout()  
        plt.show()
    def run(self,img,visual=False):
        if (visual==True):
            img = skimage.io.imread(img)
        heatmodel = self.model(input_shape=(None,None,3), filename=self.model_name)
        heatmap = heatmodel.predict(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))
        # Read in image similar to one shown above 
        image =  np.copy(img)
        #image = mpimg.imread(img_name)
        #image = mpimg.imread("straight_lines2.jpg")
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        xx, yy = np.meshgrid(np.arange(heatmap.shape[2]),np.arange(heatmap.shape[1]))
        x = (xx[heatmap[0,:,:,0]>0.99])
        y = (yy[heatmap[0,:,:,0]>0.99])
        # Add heat to each box in box list
        heat = self.add_heat(heat,x,y)
        # Apply threshold to help remove false positives
        heat = self.apply_threshold(heat,0.99)
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes_test(np.copy(image), labels)
        if (visual==True):
            self.plot_image(img,draw_img,heatmap)
        return draw_img

"""
        
cars = glob.glob("./vehicles/*/*.png")
non_cars = glob.glob("./non-vehicles/*/*.png")

# Generate Y Vector
Y = np.concatenate([np.ones(len(cars)), np.zeros(len(non_cars))-1])

# Read X Vector
X = []
for name in cars:    
    X.append(skimage.io.imread(name))
for name in non_cars:    
    X.append(skimage.io.imread(name))
X = np.array(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
input_shape =  (3,64,64)




def get_conv(input_shape=(64,64,3), filename=None):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(10, 3, 3, activation='relu', name='conv1',input_shape=input_shape, border_mode="same"))
    model.add(Convolution2D(10, 3, 3, activation='relu', name='conv2',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128,8,8,activation="relu",name="dense1")) # This was Dense(128)
    model.add(Dropout(0.5))
    model.add(Convolution2D(1,1,1,name="dense2", activation="tanh")) # This was Dense(1)
    if filename:
        model.load_weights(filename)        
    return model

#==============================================================================
# model = get_conv()
# model.add(Flatten())
# #model.add(Dense(100, activation='relu'))
# #model.add(Dense(50, activation='relu'))
# #model.add(Dense(10, activation='relu'))
# #model.add(Dense(1)) 
# model.compile(loss='mse',optimizer='adadelta',metrics=['accuracy'])
# 
# model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test))
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# model.save_weights("localize_001.h5")
#==============================================================================

video_list = []

def parse_dir():
    #TODO: Change project_video.mp4 to more generic
    if (len(video_list) == 0):
        for video in glob.glob(os.path.join('videos','project_video.mp4')):
            video_list.append(video)
        return video_list
    else:
        return video_list
    
def add_heat(heatmap, x,y):

    # Iterate through list of bboxes
    for i,j in zip(x,y):
    #for box in bbox_list:
        #print ("i= {} and j = {}".format(i,j))
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        if (i > 100 and (j < 54 and j >43)):
            heatmap[j*8:j*8+64, i*8:i*8+64] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes_test(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


heatmodel = get_conv(input_shape=(None,None,3), filename="localize_001.h5")
# that is it!


def run(img):
    #img = (skimage.io.imread(img_name))
    #img = (skimage.io.imread("straight_lines2.jpg"))
    #draw_img = get_BB_new_img(img)
    #img = generate_lane_mask(img, 400)
    #img = img_process(img,cspace='HLS')
    heatmap = heatmodel.predict(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))
    
    # Read in image similar to one shown above 
    image =  np.copy(img)
    #image = mpimg.imread(img_name)
    #image = mpimg.imread("straight_lines2.jpg")
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    xx, yy = np.meshgrid(np.arange(heatmap.shape[2]),np.arange(heatmap.shape[1]))
    x = (xx[heatmap[0,:,:,0]>0.99])
    y = (yy[heatmap[0,:,:,0]>0.99])
    
    
    # Add heat to each box in box list
    heat = add_heat(heat,x,y)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,0.99)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes_test(np.copy(image), labels)
    
    return draw_img
    
#==============================================================================
#     fig = plt.figure(figsize=(15,10))
#     plt.subplot(121)
#     plt.imshow(draw_img)
#     plt.title('Car Positions')
#     plt.subplot(122)
#     plt.imshow(heatmap, cmap='hot')
#     plt.title('Heat Map')
#     fig.tight_layout()
#==============================================================================

def generate():
   video_list = parse_dir()

   clip1 = VideoFileClip(video_list[VIDEO_INDEX])
   project_clip = clip1.fl_image(run)
   
   project_output = video_list[VIDEO_INDEX][:-4] + '_out01.mp4'
   project_clip.write_videofile(project_output, audio=False) 
   
   


#run("test1.jpg")


#generate()

#==============================================================================
# for video in glob.glob(os.path.join('../Advanced-Lane-Finding/videos','project_video.mp4')):
#     title = video.split("\\")[-1]
#     #self.list_videos.append(title)
#     #if (title == "project_video.mp4"):
#     print (video)
#     clip1 = VideoFileClip(video)   
#     project_clip = clip1.fl_image()
#     video_name = video.split(".")
#     video_name = video_name[0] + '_output.mp4'
#     print ("split name {} ".format(video_name))
#     project_clip.write_videofile(video_name, audio=False)
#     #project_output = VIDEOS[SELECTED_VIDEO][:-4] + '_ann.mp4'
#==============================================================================
"""