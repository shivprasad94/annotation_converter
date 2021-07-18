__Author__ = "Shliang"
__Email__ = "shliang0603@gmail.com"

import os
import xml.etree.ElementTree as ET
from xml.dom.minidom import Document
import cv2

'''
import xml
xml.dom.minidom.Document().writexml()
def writexml(self,
             writer: Any,
             indent: str = "",
             addindent: str = "",
             newl: str = "",
             encoding: Any = None) -> None
'''

class YOLO2VOCConvert:
    def __init__(self, txts_path, xmls_path, imgs_path):
        self.txts_path = txts_path   # Annotated yolo format label file path
        self.xmls_path = xmls_path   # Save path after converting to voc format label
        self.imgs_path = imgs_path   # Read the path and name of the picture, and store it in the xml tag file
        self.classes = ["person", "car"]

    # Extract all categories from all txt files. The label format category in yolo format is the number 0,1,...
    # When writer is True, save the extracted categories to the file'./Annotations/classes.txt'
    def search_all_classes(self, writer=False):
        # Read each txt label file and take out the label information of each target
        all_names = set()
        txts = os.listdir(self.txts_path)
        # Use list generation to filter out only tag files with the suffix txt
        txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
        print(len(txts), txts)
        # 11 ['0002030.txt', '0002031.txt', ... '0002039.txt', '0002040.txt']
        for txt in txts:
            txt_file = os.path.join(self.txts_path, txt)
            with open(txt_file, 'r') as f:
                objects = f.readlines()
                for object in objects:
                    object = object.strip().split(' ')
                    print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']
                    all_names.add(int(object[0]))
            # print(objects)  # ['2 0.506667 0.553333 0.490667 0.658667\n', '0 0.496000 0.285333 0.133333 0.096000\n', '8 0.501333 0.412000 0.074667 0.237333\n']

        print("All category tags:", all_names, "Co-labeled data set: %d sheets" % len(txts))

        # Write the categories extracted from the xmls tag file into the file'./Annotations/classes.txt'
        # if writer:
        #     with open('./Annotations/classes.txt', 'w') as f:
        #         for label in all_names:
        #             f.write(label + '\n')

        return list(all_names)

    def yolo2voc(self):
        # Create a folder to save the xml tag file
        if not os.path.exists(self.xmls_path):
            os.mkdir(self.xmls_path)

        # # Read each picture, get the size information of the picture (shape)
        # imgs = os.listdir(self.imgs_path)
        # for img_name in imgs:
        #     img = cv2.imread(os.path.join(self.imgs_path, img_name))
        #     height, width, depth = img.shape
        # # print(height, width, depth) # h is how many rows (corresponding to the height of the picture), w is how many columns (corresponding to the width of the picture)
        #
        # # Read each txt label file and take out the label information of each target
        # all_names = set()
        # txts = os.listdir(self.txts_path)
        # # Use list generation to filter out only tag files with the suffix txt
        # txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
        # print(len(txts), txts)
        # # 11 ['0002030.txt', '0002031.txt', ... '0002039.txt', '0002040.txt']
        # for txt_name in txts:
        #     txt_file = os.path.join(self.txts_path, txt_name)
        #     with open(txt_file, 'r') as f:
        #         objects = f.readlines()
        #         for object in objects:
        #             object = object.strip().split(' ')
        #             print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']

        # Rewrite the above two loops into one loop:
        imgs = os.listdir(self.imgs_path)
        txts = os.listdir(self.txts_path)
        txts = [txt for txt in txts if not txt.split('.')[0] == "classes"]  # Filter out the classes.txt file
        print(txts)
        # Note, here keep the number of pictures equal to the number of label txt files, and ensure that the names are in one-to-one correspondence (later improvement, just by judging whether the txt file name is in imgs)
        if len(imgs) == len(txts):   # Note: ./Annotation_txt Do not put the classes.txt file into it
            map_imgs_txts = [(img, txt) for img, txt in zip(imgs, txts)]
            txts = [txt for txt in txts if txt.split('.')[-1] == 'txt']
            print(len(txts), txts)
            for img_name, txt_name in map_imgs_txts:
                # Read the scale information of the picture
                print("Read picture:", img_name)
                img = cv2.imread(os.path.join(self.imgs_path, img_name))
                height_img, width_img, depth_img = img.shape
                print(height_img, width_img, depth_img)   # h is the number of rows (corresponding to the height of the picture), w is the number of columns (corresponding to the width of the picture)

                # Get the label information in the label file txt
                all_objects = []
                txt_file = os.path.join(self.txts_path, txt_name)
                with open(txt_file, 'r') as f:
                    objects = f.readlines()
                    for object in objects:
                        object = object.strip().split(' ')
                        all_objects.append(object)
                        print(object)  # ['2', '0.506667', '0.553333', '0.490667', '0.658667']

                # Create tags in the xml tag file
                xmlBuilder = Document()
                # Create an annotation tag, which is also the root tag
                annotation = xmlBuilder.createElement("annotation")

                # Add a subtag to the label annotation
                xmlBuilder.appendChild(annotation)

                # Create subtag folder
                folder = xmlBuilder.createElement("folder")
                # Store content in the subtag folder, the content in the folder tag is the folder where the pictures are stored, for example: JPEGImages
                folderContent = xmlBuilder.createTextNode(self.imgs_path.split('/')[-1])  # Tag memory
                folder.appendChild(folderContent)  # Save content to label
                annotation.appendChild(folder)   # Put the stored folder tag under the annotation root tag

                # Create subtag filename
                filename = xmlBuilder.createElement("filename")
                # Store the content in the subtag filename, the content in the filename tag is the name of the picture, for example: 000250.jpg
                filenameContent = xmlBuilder.createTextNode(txt_name.split('.')[0] + '.jpg')  # Label content
                filename.appendChild(filenameContent)
                annotation.appendChild(filename)

                # Store the shape of the picture in the xml tag
                size = xmlBuilder.createElement("size")
                # Create subtag width for size tag
                width = xmlBuilder.createElement("width")  # size subtag width
                widthContent = xmlBuilder.createTextNode(str(width_img))
                width.appendChild(widthContent)
                size.appendChild(width)   # Add width as a subtag of size
                # Create a subtag height for the size tag
                height = xmlBuilder.createElement("height")  # size subtag height
                heightContent = xmlBuilder.createTextNode(str(height_img))  # The content stored in the xml tag is a string
                height.appendChild(heightContent)
                size.appendChild(height)  # Add width as a subtag of size
                # Create a subtag depth for the size tag
                depth = xmlBuilder.createElement("depth")  # size subtag width
                depthContent = xmlBuilder.createTextNode(str(depth_img))
                depth.appendChild(depthContent)
                size.appendChild(depth)  # Add width as a subtag of size
                annotation.appendChild(size)   # Add size as a subtag of annotation

                # Stored in each object is ['2', '0.506667', '0.553333', '0.490667', '0.658667'] an annotation target
                for object_info in all_objects:
                    # Start creating a label to label the label information of the target
                    object = xmlBuilder.createElement("object")  # Create object tag
                    # Create label category label
                    # Create name tag
                    imgName = xmlBuilder.createElement("name")  # Create name tag
                    imgNameContent = xmlBuilder.createTextNode(self.classes[int(object_info[0])])
                    imgName.appendChild(imgNameContent)
                    object.appendChild(imgName)  # Add name as a subtag of object

                    # Create pose tag
                    pose = xmlBuilder.createElement("pose")
                    poseContent = xmlBuilder.createTextNode("Unspecified")
                    pose.appendChild(poseContent)
                    object.appendChild(pose)  # Add pose as the tag of object

                    # Create truncated tags
                    truncated = xmlBuilder.createElement("truncated")
                    truncatedContent = xmlBuilder.createTextNode("0")
                    truncated.appendChild(truncatedContent)
                    object.appendChild(truncated)

                    # Create difficult tags
                    difficult = xmlBuilder.createElement("difficult")
                    difficultContent = xmlBuilder.createTextNode("0")
                    difficult.appendChild(difficultContent)
                    object.appendChild(difficult)

                    # First convert the coordinates
                    # (objx_center, objy_center, obj_width, obj_height)->(xmin，ymin, xmax,ymax)
                    x_center = float(object_info[1])*width_img + 1
                    y_center = float(object_info[2])*height_img + 1
                    xminVal = int(x_center - 0.5*float(object_info[3])*width_img)   # The elements in the object_info list are all string types
                    yminVal = int(y_center - 0.5*float(object_info[4])*height_img)
                    xmaxVal = int(x_center + 0.5*float(object_info[3])*width_img)
                    ymaxVal = int(y_center + 0.5*float(object_info[4])*height_img)



                    # Create bndbox label (three-level label)
                    bndbox = xmlBuilder.createElement("bndbox")
                    # Create four more sub-labels (xmin, ymin, xmax, ymax) under the bndbox label to mark the coordinates and width and height information of the object
                    # In the voc format, label information: coordinates of the upper left corner (xmin, ymin) (xmax, ymax) coordinates of the lower right corner
                    # 1. Create xmin label
                    xmin = xmlBuilder.createElement("xmin")  # Create xmin label (four-level label)
                    xminContent = xmlBuilder.createTextNode(str(xminVal))
                    xmin.appendChild(xminContent)
                    bndbox.appendChild(xmin)
                    # 2, create ymin label
                    ymin = xmlBuilder.createElement("ymin")  # Create ymin label (four-level label)
                    yminContent = xmlBuilder.createTextNode(str(yminVal))
                    ymin.appendChild(yminContent)
                    bndbox.appendChild(ymin)
                    # 3. Create xmax label
                    xmax = xmlBuilder.createElement("xmax")  # Create xmax label (four-level label)
                    xmaxContent = xmlBuilder.createTextNode(str(xmaxVal))
                    xmax.appendChild(xmaxContent)
                    bndbox.appendChild(xmax)
                    # 4. Create a ymax label
                    ymax = xmlBuilder.createElement("ymax")  # Create ymax label (four-level label)
                    ymaxContent = xmlBuilder.createTextNode(str(ymaxVal))
                    ymax.appendChild(ymaxContent)
                    bndbox.appendChild(ymax)

                    object.appendChild(bndbox)
                    annotation.appendChild(object)  # Add object as a subtag of annotation
                f = open(os.path.join(self.xmls_path, txt_name.split('.')[0]+'.xml'), 'w')
                xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
                f.close()








if __name__ == '__main__':
    txts_path1 = r'C:\AIML_COE\EagleView_Assignment\dataset\New'
    xmls_path1 = 'Annotations_xml'
    imgs_path1 = r'C:\AIML_COE\EagleView_Assignment\dataset\New_images'

    yolo2voc_obj1 = YOLO2VOCConvert(txts_path1, xmls_path1, imgs_path1)
    labels = yolo2voc_obj1.search_all_classes()
    print('labels: ', labels)
    yolo2voc_obj1.yolo2voc()
