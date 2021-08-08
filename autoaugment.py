
# Objective: Augments your initial dataset using a series of effects: 
# noise, gaussian blur, flip horizontal & vertical, occlusion
#
# --------------------------------------------------------------------
#
# File Structure: Make sure you have one folder named "Images" with all your labelled images
# at same location as this script following this format:
# | Images
# | autoaugment.py
#
# How to use:
# Execute main srcipt using python autoaugment.py
# 
# Following variables may be changed at convenience:
#
# Main run-time: noise          (default: true)
#              | gaussian blur  (default: true)
#              | flips          (default: true)
#              | occlude        (default: true)
#
# AddNoise:  prob       (default : 0,05)
# Gaussian:  Blur loops (default: 4)
# Occlude:   occ_p      (default: 0,25)

import numpy as np
import os
import random
import cv2
import xml.etree.ElementTree
import sys
import argparse
import shutil

# Adds salt and pepper noise to image using a probability value
def AddNoise(image, prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]): # x loop
        for j in range(image.shape[1]): # y loop 
            rdn = random.random()
            if rdn < prob: # bottom threshold
                output[i][j] = 0
            elif rdn > thres: # top threshold
                output[i][j] = 255
            else: # copy base value
                output[i][j] = image[i][j]
    return output

# Applies a Gaussian Blur to the specified image, repeated [loops] number of times
def ApplyGaussianBlur(image, loops):
    output = cv2.GaussianBlur(image,(5,5),0)
    for i in range (loops - 1): 
        output = cv2.GaussianBlur(output,(5,5),0)
    return output

def FlipVertical(image):
    output = cv2.flip(image, 0)
    return output

def FlipHorizontal(image):
    output = cv2.flip(image, 1)
    return output

# uses defined boxes in annotated image and occludes parts of it scaling with occ_p
def Occlude(image, file, occ_p):

    if (not os.path.isfile("Images/" + file + ".xml")): return # skip (can't work on non-anotated images)

    # initialize output
    output = np.zeros(image.shape,np.uint8)

    # Open original XML file
    et = xml.etree.ElementTree.parse("Images/" + file + ".xml")
    root = et.getroot()

    # Collect all bounding box coordinates
    allcoords = [] # xmin - ymin - xmax - ymax
    occludecoords = [] # the boxes we will occlude
    allboxes = root.findall("object") # objects contain name - pose - truncated - difficult - bndbox
    for box in allboxes:  # loop over objects
        for element in list(box):
            if element.tag == "bndbox": # the coords of the bndbox
                for coords in list(element):
                    allcoords.append(int(coords.text)) # store all the coordinates here

    for i in range (0, int(len(allcoords)), 4): # loop 4 by 4 (for every bndbox)
        xmin = allcoords[i] # unpack
        ymin = allcoords[i + 1]
        xmax = allcoords[i + 2]
        ymax = allcoords[i + 3]

        desire_w = int((xmax - xmin) * occ_p - 1) # the desired w of occlusion box
        desire_h = int((ymax - ymin) * occ_p - 1) # the desired h of occlusion box

        # now choose xmin randomly so it fits in the bndbox
        d_xmin = random.randrange(xmin, xmax - desire_w)
        d_xmax = d_xmin + desire_w
        # now choose ymin randomly so it fits in the bndbox
        d_ymin = random.randrange(ymin, ymax - desire_h)
        d_ymax = d_ymin + desire_h

        # store in the occlude coords for a single pass along the image
        occludecoords.append(d_xmin)
        occludecoords.append(d_ymin)
        occludecoords.append(d_xmax)
        occludecoords.append(d_ymax)

    # copy base image
    for i in range(image.shape[0]): # x loop
            for j in range(image.shape[1]): # y loop 
                    output[i][j] = image[i][j] # copy image first
    
    # occlude zones --> set pixels to white in the image
    for index in range (0, int(len(occludecoords)), 4):
        output[occludecoords[index + 1]: occludecoords[index + 3], occludecoords[index]: occludecoords[index + 2]] = [255, 255, 255]
                  
    return output

def DarkenLighten(image, _value):

    # initialize output
    output1 = np.zeros(image.shape, np.uint8)
    output2 = np.zeros(image.shape, np.uint8)

    for i in range(image.shape[0]): # x loop
        for j in range(image.shape[1]): # y loop 
            output1[i][j] = max(image[i][j] - _value, 0) # darkened version
            output2[i][j] = min(image[i][j] + _value, 255) # darkened version

    return output1, output2


# Checks if the specified point should be occluded or not
def IsPointIn(x, y, coords, index):
    if (x > coords[index] and x < coords[index + 2] and 
        y > coords[index + 1] and y < coords[index + 3]) : 
        #print ("Returned true on coords: " + str(x) + " " + str(y) + " in zone " + str(coords[index]) + " " + str(coords[index + 2]) + " " + str(coords[index + 1]) + " " + str(coords[index + 3]))
        return True
    else : 
        return False


# Runs all selected augmentation operations on initial dataset
def RunAll(filename, noise, blur, flips, occlude, darkenlighten, _prob):

    inputdir = "Images/"

    basefilename = os.path.splitext(filename)[0] # name of the file without extension
    image = cv2.imread(inputdir + filename) # with colours
    image_gs = cv2.imread(inputdir + filename,0) # convert to grayscale
    img_h = image.shape[0] # the height of the image
    img_w = image.shape[1] # the width of the image

    prob = _prob # copy variable

    if (not os.path.isfile("Images/" + basefilename + ".xml")): 
        prob = max(_prob - 0.05, 0.1) # lower augmentations for un-annotated images
        #return # don't augment un-annotated images

    # Noise
    if noise and Roll(prob):
        noise_img_gs = AddNoise(image_gs,0.05) # applying to grayscale only here is more interesting
        cv2.imwrite(inputdir + basefilename + '_noise_gs.jpg', noise_img_gs) 
        CreateXML(basefilename, '_noise_gs')

    # Gaussian Blur
    if blur and Roll(prob):
        blur_img = ApplyGaussianBlur(image, 4) # second parameter defines number of blurs applied
        cv2.imwrite(inputdir + basefilename + '_gblur.jpg', blur_img)
        CreateXML(basefilename, "_gblur")

    # Flips
    if flips and Roll(prob):
        flip_v = FlipVertical(image)
        flip_h = FlipHorizontal(image)
        cv2.imwrite(inputdir + basefilename + '_flip_v.jpg', flip_v)
        cv2.imwrite(inputdir + basefilename + '_flip_h.jpg', flip_h)
        CreateFlippedXML(basefilename, "_flip_v", 0, img_h, img_w) # adjust bndbox coordinates to match flip
        CreateFlippedXML(basefilename, "_flip_h", 1, img_h, img_w) 

    # Occlude
    if occlude and Roll(prob):
        occluded = Occlude(image, basefilename, 0.15)
        if (occluded is not None):
            cv2.imwrite(inputdir + basefilename + '_occluded.jpg', occluded)
            CreateXML(basefilename, "_occluded")

    # Darken / Lighten
    if darkenlighten and Roll(prob):
        darken, lighten = DarkenLighten(image_gs, 45)
        cv2.imwrite(inputdir + basefilename + '_darkened.jpg', darken)
        CreateXML(basefilename, "_darkened")
        cv2.imwrite(inputdir + basefilename + '_lightened.jpg', lighten)
        CreateXML(basefilename, "_lightened")

# Updates XML data using base XML annotated file in "Images/"
def CreateXML(file, newextension):

    if (not os.path.isfile("Images/" + file + ".xml")): return # skip

    # Open original XML file
    et = xml.etree.ElementTree.parse("Images/" + file + ".xml")
    root = et.getroot()

    # Update Path
    currentpath = root.find("folder").text.split(file)[0] # get current path up to here
    currentpath +="\\"+ file + newextension + ".xml" # update to match what we want here
    root.find("folder").text = currentpath

    et.write(os.path.join(currentpath)) # save

# Updates XML data using base XML annotated file in "Images/"
def CreateFlippedXML(file, newextension, fliporientation, img_h, img_w):

    if (not os.path.isfile("Images/" + file + ".xml")): return # skip

    # Open original XML file
    et = xml.etree.ElementTree.parse("Images/" + file + ".xml")
    root = et.getroot()

    # Update path
    currentpath = root.find("folder").text.split(file)[0] # get current path up to here
    currentpath +="\\"+file  +newextension + ".xml" # update to match what we want here
    root.find("folder").text = currentpath

    # Convert *.xml coordinates to *.txt format
    allboxes = root.findall("object") # objects contain name - pose - truncated - difficult - bndbox
    for box in allboxes:  # loop over objects
        for element in list(box):
            if element.tag == "bndbox": # the coords of the bndbox
                bndbox = list(element)

                if (fliporientation == 1 or fliporientation == 2): # horizontal flip
                    oldmin = bndbox[0].text
                    oldmax = bndbox[2].text
                    bndbox[0].text = str(img_w - int(oldmax) - 1)
                    bndbox[2].text = str(img_w - int(oldmin) - 1)

                if (fliporientation == 0 or fliporientation == 2): # vertical flip
                    oldmin = bndbox[1].text
                    oldmax = bndbox[3].text
                    bndbox[1].text = str(img_h - int(oldmax) - 1)
                    bndbox[3].text = str(img_h - int(oldmin) - 1)

    et.write(os.path.join(currentpath)) # save

def CheckIfImage(filename):
    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp") or filename.endswith(".jpeg")
    or filename.endswith(".PNG") or filename.endswith(".JPG") or filename.endswith(".BMP") or filename.endswith(".JPEG")):
        return True
    
    return False

def Roll(_prob):
    if (random.random() < _prob): return True
    return False

# EXECUTE
if __name__ == "__main__":

    if not os.path.exists("Backup/"): # backup is used for rollback 
        os.makedirs("Backup/")

    for filename in os.listdir("Images/"):
        shutil.copyfile("Images/" + filename, "Backup/" + filename) # save to backup before proceeding

    for filename in os.listdir("Images/"): # where the annotated images should be
        if CheckIfImage(filename): # check extensions
            print("running on file " + filename)
            RunAll(filename, True, True, True, False, True, 0.8) # Noise - Gaussian Blur - Flip Image - Occlude - Darken/Lighten

    if not os.path.exists("XML/"): # create a new XML folder
        os.makedirs("XML/")

    for filename in os.listdir("Images/"): # move all XMLs to that XML folder
        if (filename.endswith(".xml")):
            os.replace("Images/" + filename, "XML/" + filename)

    os.system("python dataformatter.py")


