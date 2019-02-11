#!/usr/bin/env python
'''
===============================================================================
Interactive Image Processing
---
Two windows will show up, one for original image and one for processed.
Modify the processed image as needed, using the keys below:
---
Esc  =>  Exit
' '  =>  Save and move to the next image
'b'  =>  Go back to the previous image
'f'  =>  Flip the image horizontally
'c'  =>  Crop the image using the mouse
'p'  =>  Pad the image with black border on top and bottom
'l'  =>  Rotate the image 90 degrees left
'o'  =>  Load the original (uncropped) image
'r'  =>  Reset all adjustments
'a'  =>  Flag the image for later attention
===============================================================================
'''
import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
import sys

rectangle = False                # flag for drawing rect
width     = 480                  # resize all images to
height    = 320                  # resize all images to
path_ori  = '../train/'          # folder with original images
path_src  = '../cropped/train/'  # folder with cropped images
path_des  =    'reduced/train/'  # folder to place processed images

def on_mouse(event,x,y,flags,param):
    #global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
    global x0,y0,x1,y1,rectangle,img1,img2

    # Draw Rectangle
    if event == cv.EVENT_LBUTTONDOWN:
        x0, y0 = x, y
        rectangle = True

    elif event == cv.EVENT_MOUSEMOVE:
        if rectangle == True:
            img_c = img1.copy()
            cv.rectangle(img_c,(x0,y0),(x,y),[255,0,0],2)
            cv.imshow('window1',img_c)

    elif event == cv.EVENT_LBUTTONUP:
        rectangle = False
        x1, y1 = x, y
        img_c = img1.copy()
        img_c = img_c[y0:y1, x0:x1]
        img2  = cv.resize(img_c, (width, height))
        img2  = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        cv.imshow('window1',img1)
        cv.imshow('window2',img2)
        print('cropped.. ', end='', flush=True)

if __name__ == '__main__':

    # print documentation
    print(__doc__)

    traindf = pd.read_csv('traindf.csv')

    i = 0
    wait = 0
    back = 0

    while True:
        
        if wait == 0:
            if back == 0:
                if traindf.iloc[i]['Processed'] == 1:
                    i += 1
                    continue
            back = 0
            img_name   = traindf.iloc[i]['Image']
            img1_fname = path_src + img_name
            img2_fname = path_des + img_name
            img1       = cv.imread(img1_fname)
            img2       = cv.resize(img1,(width, height))
            img2       = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
            
            print('{:>5}: Image {}.. '.format(i,img_name),end='',flush=True)

            # create windows
            cv.namedWindow('window1')
            cv.namedWindow('window2')
            
            cv.imshow('window1',img1)
            cv.imshow('window2',img2)

        k = cv.waitKey(0)

        # reset any mouse callback
        cv.setMouseCallback('window1', lambda *args: None)

        if k==27: # esc
            print('Good bye')
            break

        elif k==ord(' '):
            wait = 0
            traindf.loc[i, ['Processed']] = 1
            cv.imwrite(img2_fname, img2)
            print('saved')

        elif k==ord('b'):
            wait = 0
            back = 1
            i -= 1
            print('going back')
            continue

        elif k==ord('f'):
            # flip image horizontally, then wait
            wait = 1
            img2 = cv.flip(img2, 0)
            cv.imshow('window2',img2)
            print('flipped.. ', end='', flush=True)
            continue

        elif k==ord('c'):
            # crop image to region defined by user
            wait = 1
            print('crop.. ', end='', flush=True)
            cv.setMouseCallback('window1', on_mouse, param=None)
            continue

        elif k==ord('p'):
            # pad image with top/bottom border (black)
            wait = 1
            print('pad.. ',end='',flush=True)
            b    = 10
            newh = img2.shape[0]-(2*b)
            img2 = cv.resize(img2,(width, newh))
            img2 = cv.copyMakeBorder(img2,b,b,0,0,cv.BORDER_CONSTANT,value=[0,0,0])
            cv.imshow('window2',img2)
            continue

        elif k==ord('l'):
            # rotate left 90 degrees
            wait = 1
            print('left.. ',end='',flush=True)
            
            # grab the dimensions of the image and determine the center
            (h,w) = img1.shape[:2]
            (cX,cY) = (w//2, h//2)
         
            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv.getRotationMatrix2D((cX,cY),90,1.0)
            cos = np.abs(M[0,0])
            sin = np.abs(M[0,1])
            # compute the new bounding dimensions of the image
            nW = int((h*sin)+(w*cos))
            nH = int((h*cos)+(w*sin))
            # adjust the rotation matrix to take into account translation
            M[0,2] += (nW/2)-cX
            M[1,2] += (nH/2)-cY
            # perform the actual rotation and return the image
            img1 = cv.warpAffine(img1,M,(nW,nH))

            # rows,cols = img1.shape[:2]
            # M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
            # img1 = cv.warpAffine(img1,M,(cols,rows))
            img2      = cv.resize(img1,(width, height))
            img2      = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
            cv.imshow('window1',img1)
            cv.imshow('window2',img2)
            continue

        elif k==ord('o'):
            # view original (uncropped) image
            wait = 1
            print('original.. ', end='', flush=True)
            img_fname = path_ori + img_name
            img1      = cv.imread(img_fname)
            img2      = cv.resize(img1,(width, height))
            img2      = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
            cv.imshow('window1',img1)
            cv.imshow('window2',img2)
            continue

        elif k==ord('r'):
            # reset
            wait = 0
            print('reset')
            cv.setMouseCallback('window1', lambda *args: None)
            continue

        elif k==ord('a'):
            # flag for attention (or unflag if already flagged), then wait
            wait = 1
            cur_val = traindf.loc[i, ['Attention']][0]
            new_val = 1 if cur_val==0 else 0
            traindf.loc[i, ['Attention']] = new_val
            msg = 'attention' if new_val==1 else 'attention removed'
            print('{}.. '.format(msg), end='', flush=True)
            continue
        else:
            wait = 1
            print('?.. ', end='', flush=True)
            continue

        i += 1

traindf.to_csv('traindf.csv', index=False)
cv.destroyAllWindows()
