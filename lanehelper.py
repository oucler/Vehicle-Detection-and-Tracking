import os, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class camera_cal:
    def __init__(self,args):
        self.cal_dir = args.cal_dir
        self.dict_calibration = {}
        self.objpoints = []
        self.imgpoints = []
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.col_num = args.col_num
        self.row_num = args.row_num
        self.camera_cal_out = args.cal_o_dir
        #self.img_size = (1280,720,3)
        camera_cal.image_dir(self)
    def image_dir(self):
        for cal_image in glob.glob(os.path.join(self.cal_dir,'*.jpg')):
            title = cal_image.split("\\")[-1]
            title = title.split(".")[0]
            self.dict_calibration[title] = cal_image
    def read_image(self,img_dir):
        img = cv2.imread(img_dir)
        return img
    def bgr2rgb(self,img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img    
    def gray_image(self, img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return gray
    def populate_objp(self):
        objp = np.zeros((self.col_num*self.row_num,3),np.float32)
        objp[:,:2] = np.mgrid[0:self.col_num,0:self.row_num].T.reshape(-1,2)
        return objp
    def undistort(self,img):
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return dst
    def calibrateCamera(self,img):
        img_size = img.shape[0:2]
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size,None,None)
        return ret, mtx, dist, rvecs, tvecs
    def corners(self,gray):
        ret, corners = cv2.findChessboardCorners(gray,(self.col_num,self.row_num),None)
        return ret, corners
    def draw_corners(self,img,corners,ret):
        drawn_img = cv2.drawChessboardCorners(img,(self.col_num,self.row_num),corners,ret)
        return drawn_img
    def calibrate(self,visual=False):
        for img_dir in self.dict_calibration.values():
            img = self.read_image(img_dir)   
            gray = self.gray_image(img)
            #img = self.read_image(img_dir)               
            ret, corners = self.corners(gray)
            if ret == True:
               objp = self.populate_objp()
               self.objpoints.append(objp)
               self.imgpoints.append(corners)
               self.draw_corners(gray,corners,ret) 

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = self.calibrateCamera(gray)

    def save_img(self,images,img_dir):
        for title,img in images.items():
            title = img_dir+"/"+title+".jpg"
            mpimg.imsave(title,img,cmap='gray')


             
class data_prep(camera_cal):
    def __init__(self,args):
        camera_cal.__init__(self,args)
        self.raw_dir = args.raw_dir
        self.dict_raw_image = {}
        data_prep.image_dir(self)
    def image_dir(self):
        for raw_image in glob.glob(os.path.join(self.raw_dir,'*.jpg')):
            title = raw_image.split("\\")[-1]
            title = title.split(".jpg")[0]
            self.dict_raw_image[title] = raw_image

    def undistort_raw_image(self):
        self.image_dir()
        for img_name,img_dir in self.dict_raw_image.items():
            img = self.read_image(img_dir)
            img = self.bgr2rgb(img)
            img_size = (img.shape[1],img.shape[0])            
            undistorted_img = self.undistort(img,img_size)
            if (img_name == "straight_lines2" or img_name == "test5" or img_name == "test1"):
                camera_cal.visualize(self,original_img=img,mod_img=None,undistorted_img=undistorted_img,img_name=img_name)

class perspective_transform(data_prep):
    def __init__(self,args):    
        data_prep.__init__(self,args)
        self.offset = args.offset
        self.img_size = (1280,720)
        self.test_out_dir = args.test_o_dir
        self.src = np.float32(
                [[(self.img_size[0] / 2) - 55, self.img_size[1] / 2 + 100],
                [((self.img_size[0] / 6) - 10), self.img_size[1]],
                [(self.img_size[0] * 5 / 6) + 60, self.img_size[1]],
                [(self.img_size[0] / 2 + 55), self.img_size[1] / 2 + 100]])
        self.dst = np.float32(
                [[(self.img_size[0] / 4), 0],
                [(self.img_size[0] / 4), self.img_size[1]],
                [(self.img_size[0] * 3 / 4), self.img_size[1]],
                [(self.img_size[0] * 3 / 4), 0]])
        #"""
    def transform(self,img,src,dst):
        M = cv2.getPerspectiveTransform(src,dst)
        warped = cv2.warpPerspective(img,M, self.img_size)
        return warped
    def inverse_transform(self,img,dst,src):
        Minv = cv2.getPerspectiveTransform(dst,src)
        warped = cv2.warpPerspective(img,Minv, self.img_size)
        return warped
    def run_calibration(self,index=1,row=1,col=2, visual=False):
        if (len(self.objpoints) == 0 and len(self.imgpoints) == 0):
            self.calibrate()
        images = {}        
        camera_cal.image_dir(self)
        orig_img = self.read_image(list(self.dict_calibration.values())[index])
        images["Original_Image"] = orig_img 
        
        img = self.read_image(list(self.dict_calibration.values())[index])
        gray = self.gray_image(img)
        ret, corners = self.corners(gray)
        src = np.float32([corners[0], corners[self.col_num-1], corners[-1], corners[-self.col_num]])
        dst = np.float32([[self.offset, self.offset], [self.img_size[0]-self.offset, self.offset], 
                                         [self.img_size[0]-self.offset, self.img_size[1]-self.offset], 
                                         [self.offset, self.img_size[1]-self.offset]])        
        if ret == True:
            drawn_img = self.draw_corners(gray,corners,ret) 
            images["Chessbord_Corners"] = drawn_img 
            
            undistorted_img = self.undistort(img)
            images["Undistorted_Image"] = undistorted_img
            
            warped_img = self.transform(undistorted_img,src,dst)
            images["Warped_Image"] = warped_img
            
        self.save_img(images,self.camera_cal_out)   
        
        if(visual):
            binary_threshold.visualize(self,images,row,col)
    def birds_eye_view(self):
        data_prep.image_dir(self)
        for img_name,img_dir in self.dict_raw_image.items():
            img_name += "tessssst"
            img = self.read_image(img_dir)
            img = self.bgr2rgb(img)
            img_size = (img.shape[1],img.shape[0])            
            undistorted_img = self.undistort(img,img_size)
 
            #M = cv2.getPerspectiveTransform(self.src,self.dst)
            #warped = cv2.warpPerspective(undistorted_img,M, img_size)
            warped = self.transform(undistorted_img,img_size)            
            camera_cal.visualize(self,original_img=undistorted_img,mod_img=None,undistorted_img=warped,img_name=img_name)
            
class binary_threshold(perspective_transform):
    def __init__(self,args):
        perspective_transform.__init__(self,args)
        self.threshold_min = args.threshold_min
        self.threshold_max = args.threshold_max
    def gray_color_space(self,img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray_img
    def hls_color_space(self,img,channel):
        hls_img = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)[:,:,channel]
        return hls_img
    def hsv_color_space(self,img,channel):
        hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)[:,:,channel]
        return hsv_img
    def luv_color_space(self,img,channel):
        luv_img = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)[:,:,channel]
        return luv_img        
    def lab_color_space(self,img,channel):
        lab_img = cv2.cvtColor(img,cv2.COLOR_BGR2Lab)[:,:,channel]
        return lab_img      
    def sobel(self,img,stype='x'):
        gray = self.gray_color_space(img)
        if (stype == 'x'):
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
        else: 
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1) # Take the derivative in y
        abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        return scaled_sobel
    def binary_img(self,cspace,threshold=(0,255)):
        c_img = np.zeros_like(cspace)
        c_img[(cspace >= threshold[0]) & (cspace <= threshold[1])] = 1
        return c_img
    def visualize(self,images,row,col):
        f ,ax = plt.subplots(row,col,figsize=(20,5))
        f.tight_layout()
        for ax, key in zip(ax.ravel(),images.keys()):
            if (key == "Original Image"):            
                ax.imshow(images[key])
            else:
                ax.imshow(images[key],cmap='gray')
            ax.set_title("{}".format(key),fontsize=15)
 
    def binary(self,img,visual=False):
        images = {}
        images["Original Image"]= img
            
       
        s_img = self.hls_color_space(img,2)
        s_binary_img = self.binary_img(s_img,(self.threshold_min+20,self.threshold_max+55))
        images["HLS color space and separate the S channel"] = s_binary_img            
            
        sobel_img = self.sobel(img)
        sobel_binary_img = self.binary_img(sobel_img,(self.threshold_min-130,self.threshold_max-100))
        images["Stacked thresholds"] = sobel_binary_img

        color_binary = np.dstack(( np.zeros_like(sobel_binary_img), sobel_binary_img, s_binary_img))
        combined_binary = np.zeros_like(sobel_binary_img)
        combined_binary[(s_binary_img == 1) | (sobel_binary_img == 1)] = 1 
        images["Combined S channel and gradient thresholds"] = combined_binary

        if (visual):
            binary_threshold.visualize(self,images,row=1, col=len(images))
        return combined_binary
    
class line_finding(binary_threshold):
    def __init__(self,args):
        binary_threshold.__init__(self,args)
        self.left_fit = []
        self.right_fit = []
        self.left_fitx = []
        self.right_fitx = []
        self.ploty = []
        self.left_lane_detected = False;
        self.right_lane_detected = False
        self.left_line = None
        self.right_line = None
        self.line_segments = 10
        self.image_offset = 0
        self.n_fames = 7
         
    def visualize_histogram(self,img):
        histogram = np.sum(img[img.shape[0]//2:,:],axis=0)
        plt.plot(histogram)

    def sliding_windows(self,warped):
        binary_warped = self.binary(warped)
        slc = int(binary_warped.shape[0]/2)
        histogram = np.sum(binary_warped[slc:,:], axis=0)
        #histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255   
        
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 70
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)   
        

        #"""
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        f ,ax = plt.subplots(figsize=(20,5))
        f.tight_layout()      
        ax.imshow(out_img)
        ax.plot(left_fitx, ploty, color='yellow')
        ax.plot(right_fitx, ploty, color='yellow')
        #ax.xlim(0, 1280)
        #ax.ylim(720, 0)
        #"""
        o_dir = self.test_out_dir+"/"+"Sliding_Windows.jpg"
        f.savefig(o_dir)
        self.left_fit = left_fit
        self.right_fit = right_fit
        #self.visualize_histogram(warped_img)
        return left_fit,right_fit
    
    def draw(self,img):
        #data_prep.image_dir(self)
        #img = self.read_image(list(self.dict_raw_image.values())[2])
        binary_warped = self.binary(img)
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        
        #left_fit,right_fit = self.sliding_windows(img)
        left_fit = self.left_fit
        right_fit = self.right_fit
        
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        
        #return result
        
        f ,ax1 = plt.subplots(figsize=(20,5))
        
        ax1.imshow(result)
        ax1.plot(left_fitx, ploty, color='yellow')
        ax1.plot(right_fitx, ploty, color='yellow')
        #plt.xlim(0, 1280)
        #plt.ylim(720, 0)
        o_dir = self.test_out_dir + "/" + "Drawn_Lines.jpg"
        f.savefig(o_dir)
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx
        self.ploty = ploty
        return left_fitx, right_fitx, ploty
                    
    def draw_lines(self,warped,img,src,dst):

        #warped = self.transform(undistorted_img,self.img_size,src,dst)
        # Create an image to draw the lines on
        warped_binary = self.binary(warped)
        warp_zero = np.zeros_like(warped_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        left_fitx, right_fitx, ploty = self.left_fitx,self.right_fitx,self.ploty
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
        
        Minv = cv2.getPerspectiveTransform(dst, src)
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, self.img_size) 
        # Combine the result with the original image
        
        src_full = np.float32(
                    [[2*self.offset, 2*self.offset],[self.img_size[0]-2.5*self.offset, 2*self.offset],
                     [self.img_size[0]-2.5*self.offset, self.img_size[1]-2*self.offset],
                    [2*self.offset, self.img_size[1]-2*self.offset]])
        dst_full = np.float32([[self.offset, self.offset], [self.img_size[0]-self.offset, self.offset], 
                                     [self.img_size[0]-self.offset, self.img_size[1]-self.offset], 
                                     [self.offset, self.img_size[1]-self.offset]])
        undistorted_img = self.undistort(img)
        warped_img =  self.transform(undistorted_img,src_full,dst_full)
        #result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
        result = cv2.addWeighted(undistorted_img, 1, newwarp, 0.3, 0)
        f ,ax = plt.subplots(figsize=(20,5))
        ax.imshow(result)
   
    
    #/// \Calculates the curvature of a line in meters
    #/// \param fit_cr:
    #/// \return: radius of curvature in meters
    def calc_curvature(self,fit_cr):
  
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    
        y = np.array(np.linspace(0, 719, num=10))
        x = np.array([fit_cr(x) for x in y])
        y_eval = np.max(y)
    
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    
        return curverad
       
        
                 
 
                     
