#!/usr/bin/env python
""" gui.py
Create a Graphical User Interface (GUI) showing what the machine and viewing
and thinking.
"""

__author__ = "Vinicius Guimaraes Goecks"
__version__ = "0.0.0"
__status__ = "Prototype"
__date__ = "June 21, 2017"

# import
import numpy as np
import cv2
import time

class MachineGUI():
    """
    Class to handle Machine Graphical User Interface (MachineGUI) showing what
    the machine and viewing and thinking.\
    """
    def __init__(self,depth_width, depth_height,start_gui=False):
        self.name = 'MachineGUI'
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.gui_factor = 4 # 4 is the standard

        if start_gui:
            self.start_blank_gui()

    def create_gui(self,img_plot, act):
        """
        Create basic window and buttons.
        """
        def nothing(x):
            pass

        # create base background
        gf = self.gui_factor # to scale everything and improve visualization
        height = 100
        width = 68

        # text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (0,255,0)
        title_size = .6
        section_size = .5
        label_size = .4
        line_thick = 1

        img = np.zeros((height*gf,width*gf,3), np.uint8)
        cv2.namedWindow(self.name)

        # # add title
        # cv2.putText(img,self.name,(int(gf*width/2)-width,30), font, title_size,text_color,line_thick)

        # add space for depth pic
        self.d_x_offset = 2*gf
        self.d_y_offset = 8*gf

        # for some reason, need to update all three channels
        img[self.d_y_offset:self.d_y_offset+img_plot.shape[0], self.d_x_offset:self.d_x_offset+img_plot.shape[1],1] = img_plot

        # add title of depth pic
        cv2.putText(img,'Depth sensor',(self.d_x_offset,self.d_y_offset-10), font, section_size,text_color,line_thick)

        # add space for RGB pic
        self.rgb_x_offset = 2*gf
        self.rgb_y_offset = (42+8)*gf

        # for some reason, need to update all three channels
        ### *** USING SAME AS DEPTH FOR NOW ****
        img[self.rgb_y_offset:self.rgb_y_offset+img_plot.shape[0], self.rgb_x_offset:self.rgb_x_offset+img_plot.shape[1],0] = img_plot

        # add title of depth pic
        cv2.putText(img,'RGB sensor',(self.rgb_x_offset,self.rgb_y_offset-10), font, section_size,text_color,line_thick)

        # add space for action
        self.act_bar_x = self.depth_width*gf
        self.act_bar_y = int(self.depth_height/8)*gf
        act_bar = 100*np.ones((self.act_bar_y,self.act_bar_x,1), np.uint8)

        if act > 0:
            act_bar_idx = int((1 + act)*self.act_bar_x/2)

            act_bar[:,int(self.act_bar_x/2):act_bar_idx,0] = 200
        elif act < 0:
            act_bar_idx = int((1 + act)*self.act_bar_x/2)
            act_bar[:,act_bar_idx:int(self.act_bar_x/2),0] = 200

        self.a_x_offset = 2*gf
        self.a_y_offset = 92*gf
        img[self.a_y_offset:self.a_y_offset+act_bar.shape[0], self.a_x_offset:self.a_x_offset+act_bar.shape[1]] = act_bar

        # add title for action
        cv2.putText(img,'Action',(self.a_x_offset,self.a_y_offset-10), font, section_size,text_color,line_thick)

        # add label for action
        cv2.putText(img,'0',(int(width/2*gf)-5,self.a_y_offset+6*gf+3), font, label_size,text_color,line_thick)
        cv2.putText(img,'min',(self.a_x_offset,self.a_y_offset+6*gf+3), font, label_size,text_color,line_thick)
        cv2.putText(img,'max',((width-9)*gf,self.a_y_offset+6*gf+3), font, label_size,text_color,line_thick)

        return img

    def start_blank_gui(self):
        """
        Start blank gui.
        """
        # create black image and zero action
        depth = 100*np.ones((self.depth_height,self.depth_width,1), np.uint8)
        act = 0

        # scale input image to gui size
        img = self.preprocess(depth)

        # create it
        img_gui = self.create_gui(img, act)
        self.img_gui = img_gui

    def preprocess(self, img):
        """
        Resize image. Converts and down-samples the input image.
        """
        res = cv2.resize(img,None,fx=self.gui_factor, fy=self.gui_factor, interpolation = cv2.INTER_LINEAR)
        return res

    def display(self, img, act):
        """
        Display on gui image and action inputs.
        """
        # scale input image to gui size
        img = 255*self.preprocess(img)

        # update depth
        self.img_gui[self.d_y_offset:self.d_y_offset+img.shape[0], self.d_x_offset:self.d_x_offset+img.shape[1],1] = img

        # update rgb (not supported yet)

        # update action
        act_bar = 100*np.ones((self.act_bar_y,self.act_bar_x,1), np.uint8)

        if act > 0:
            act_bar_idx = int((1 + act)*self.act_bar_x/2)

            act_bar[:,int(self.act_bar_x/2):act_bar_idx,0] = 200
        elif act < 0:
            act_bar_idx = int((1 + act)*self.act_bar_x/2)
            act_bar[:,act_bar_idx:int(self.act_bar_x/2),0] = 200

        self.img_gui[self.a_y_offset:self.a_y_offset+act_bar.shape[0], self.a_x_offset:self.a_x_offset+act_bar.shape[1]] = act_bar


        # show it
        cv2.imshow(self.name,self.img_gui)
        k = cv2.waitKey(1) & 0xFF

    def test(self):
        """
        Test class and its new features.
        """
        print('** TEST MachineGUI **')
        # create basic gui
        # img = self.create_gui()

        # test data
        # load it
        print("Loading data...")
        root_file = '../data/test_imit_'
        data = np.genfromtxt(root_file+'0.csv', delimiter=',')
        print('Data loaded. Displaying...')

        # split states and actions
        x = data[:,:-1]
        y = data[:,-1]

        # reshape arrays to 2d images
        # samples, height, width, channels
        samples = x.shape[0]
        x = x.reshape((samples, self.depth_height, self.depth_width, 1))
        y = y.reshape((samples, 1))

        n = 0
        try:
            while(1):
                # example image and action
                img_plot = self.preprocess(x[n,:,:,0])
                act = y[n,:]

                # create gui
                img = self.create_gui(img_plot, act)
                time.sleep(1/20)
                n += 1

                cv2.imshow(self.name,img)
                k = cv2.waitKey(1) & 0xFF
                # if k == 27:
                #     break

            # cv2.destroyAllWindows()
        except:
            print('No more data.')

if __name__ == '__main__':
    machine = MachineGUI(depth_width=64,depth_height=36)
    machine.test()
