import numpy as np
from lanehelper import line_finding
from ImageUtils import *
from Line import Line, calc_curvature
#from vehicle_detection_tracking import run
from vehicle_detection_tracking import vehicleDetectionTracking
#/// \Tracks lane lines on images or a video stream using techniques like Sobel operation, 
#/// \color thresholding and sliding histogram.


class LaneDetector(line_finding,vehicleDetectionTracking):
    def __init__(self, args):
        line_finding.__init__(self,args)
        vehicleDetectionTracking.__init__(self,args)
        src = np.float32([
            (132, 703),
            (540, 466),
            (740, 466),
            (1147, 703)])
        
        dst = np.float32([
            (src[0][0] + args.transform_offset, 720),
            (src[0][0] + args.transform_offset, 0),
            (src[-1][0] - args.transform_offset, 0),
            (src[-1][0] - args.transform_offset, 720)])           

        self.n_frames = args.n_frames
        self.line_segments = args.line_segments
        self.image_offset = args.transform_offset
        #"""
        self.cam_calibration = None
        self.left_line = None
        self.right_line = None
        self.center_poly = None
        self.curvature = 0.0
        self.offset = 0.0

        self.perspective_src = src
        self.perspective_dst = dst

        self.dists = []
        # Calibrating the camera
        self.calibrate()
        
    #/// \Determines if pixels describing two line are plausible lane lines based on curvature and distance.
    #/// \param left: Tuple of arrays containing the coordinates of detected pixels
    #/// \param right: Tuple of arrays containing the coordinates of detected pixels
    #/// \return:
    def line_possibility(self, left, right):

        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:
            new_left = Line(y=left[0], x=left[1])
            new_right = Line(y=right[0], x=right[1])
            return are_lanes_plausible(new_left, new_right)

      
    #/// \Compares two line to each other and to their last prediction.
    #/// \param left_x:
    #/// \param left_y:
    #/// \param right_x:
    #/// \param right_y:
    #/// \return: boolean tuple (left_detected, right_detected)
    def verify_lines(self, left_x, left_y, right_x, right_y):
 
        left_detected = False
        right_detected = False

        if self.line_possibility((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line is not None and self.right_line is not None:
            if self.line_possibility((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.line_possibility((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        return left_detected, right_detected


        
    #/// \Draws information about the center offset and the current lane curvature onto the given image.
    #/// \param img:
    def draw_info(self, img):

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Radius of Curvature = %d(m)' % self.curvature, (50, 50), font, 1, (255, 255, 255), 2)
        left_or_right = 'left' if self.offset < 0 else 'right'
        cv2.putText(img, 'Vehicle is %.2fm %s of center' % (np.abs(self.offset), left_or_right), (50, 100), font, 1,
                    (255, 255, 255), 2)

        
    #/// \Draws the predicted lane onto the image. Containing the lane area, center line and the lane lines.
    #/// \param img:

    def draw_lane_borders(self, img):

        overlay = np.zeros([*img.shape])
        mask = np.zeros([img.shape[0], img.shape[1]])

        # lane area
        lane_area = calculate_lane_area((self.left_line, self.right_line), img.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        #mask = self.perspective_transformer.inverse_transform(mask)
        mask = self.inverse_transform(mask,self.perspective_dst,self.perspective_src)

        overlay[mask == 1] = (0, 255, 0)
        selection = (overlay != 0)
        img[selection] = img[selection] * 0.3 + overlay[selection] * 0.7

        """
        # center line
        mask[:] = 0
        mask = draw_poly_arr(mask, self.center_poly, 20, 255, 5, True, tip_length=0.5)
        #mask = self.perspective_transformer.inverse_transform(mask)
        mask = self.inverse_transform(mask,self.perspective_dst,self.perspective_src)
        img[mask == 255] = (255, 75, 2)
        """
        # lines best
        mask[:] = 0
        mask = draw_poly(mask, self.left_line.best_fit_poly, 5, 255)
        mask = draw_poly(mask, self.right_line.best_fit_poly, 5, 255)
        #mask = self.perspective_transformer.inverse_transform(mask)
        mask = self.inverse_transform(mask,self.perspective_dst,self.perspective_src)
        img[mask == 255] = (255, 200, 2)

    # /// \ Lane detection on video frame
    # /// \param video frame
    # /// \return: annotated video frame
    def process_frame(self, frame):
        orig_frame = np.copy(frame)

        # Undistorting raw image
        frame = self.undistort(frame)

        # Use color transforms, gradients, etc., to create a thresholded binary image.
        frame = generate_lane_mask(frame, 400)
        
        # Returns a warped image
        frame = self.transform(frame,self.perspective_src,self.perspective_dst)


        left_detected = right_detected = False
        left_x = left_y = right_x = right_y = []

        # If there have been lanes detected in the past, the algorithm will first try to
        # find new lanes along the old one. This will improve performance
        if self.left_line is not None and self.right_line is not None:
            left_x, left_y = detectLane(frame, self.left_line.best_fit_poly, self.line_segments)
            right_x, right_y = detectLane(frame, self.right_line.best_fit_poly, self.line_segments)

            left_detected, right_detected = self.verify_lines(left_x, left_y, right_x, right_y)

        # If no lanes are found a histogram search will be performed
        if not left_detected:
            left_x, left_y = histogram_lane_detection(
                frame, self.line_segments, (self.image_offset, frame.shape[1] // 2), h_window=7)
            left_x, left_y = remove_outlier(left_x, left_y)
        if not right_detected:
            right_x, right_y = histogram_lane_detection(
                frame, self.line_segments, (frame.shape[1] // 2, frame.shape[1] - self.image_offset), h_window=7)
            right_x, right_y = remove_outlier(right_x, right_y)

        if not left_detected or not right_detected:
            left_detected, right_detected = self.verify_lines(left_x, left_y, right_x, right_y)

        # Updated left lane information.
        if left_detected:
            # switch x and y since lines are almost vertical
            if self.left_line is not None:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.n_frames, left_y, left_x)

        # Updated right lane information.
        if right_detected:
            # switch x and y since lines are almost vertical
            if self.right_line is not None:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.n_frames, right_y, right_x)

        # Add information onto the frame
        if self.left_line is not None and self.right_line is not None:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.center_poly = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = self.calc_curvature(self.center_poly)
            self.offset = (frame.shape[1] / 2 - self.center_poly(719)) * 3.7 / 700
            orig_frame = self.run(orig_frame)
            self.draw_lane_borders(orig_frame)
            self.draw_info(orig_frame)

        return orig_frame
