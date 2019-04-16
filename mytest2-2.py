# -*- coding: utf-8 -*

import os
import sys

# app_dir = '/home/gxh/works/tf/tensorflow/tensorflow/models-master/research'  # '/home/gxh/works/tf/jy/mytrain/'
# environment_file = '/home/gxh/works/mystart2.sh'
app_dir='/home/ubutnu/work/dataArea/gxh-1/models-master/research'
environment_file='/home/ubutnu/work/dataArea/gxh-1/set_ev.sh'
os.system('source %s' % environment_file)
sys.path.append(app_dir)

import cv2
import time
import datetime
import six.moves.urllib as urllib
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

sys.path.append("..")
from object_detection.utils import ops as utils_ops

# from utils import label_map_util
# from utils import visualization_utils as vis_util

# 为了显示 pyplot 的 figure， 需要提前设置 matplotlib 的 backend.
# print(matplotlib.get_backend())  # 默认的是 agg， 即不显示 figure
# 编辑 from object_detection.utils import visualization_utils as vis_util 中的 visualization_utils.py，注释掉以下部分：
# import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
SHOW_MODE = True

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

imgTypes = [".png", ".jpg", ".bmp"]
IMAGE_SIZE = (36, 24)


# print(sys.argv[0])
# targetDir = sys.argv[1]

def jpg_mask(output_dict, frame, category_index):
    frame2 = frame
    num = output_dict['num_detections']
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    instance_masks = output_dict.get('detection_masks')
    # print("num= ", num)

    # print("classes= ", classes)
    # print(" len(classes)= ", len(classes))
    # print("self.category_index= ", self.category_index)
    print("instance_masks[0]= ", instance_masks[0])
    print("instance_masks.shape= ", instance_masks.shape)
    grayvalues = np.unique(instance_masks)
    print("instance_masks.grayvalues= ", grayvalues)
    grayvalues = np.unique(instance_masks[0])
    print("instance_masks[0].grayvalues= ", grayvalues)
    max_idx = -1
    max_area = 0
    height, width = frame.shape[:2]  # 读取图像的宽和高
    for i in range(num):
        idx = classes[i]
        # print("idx= ", idx)
        itm = category_index[idx]
        name = itm['name']
        idx = itm['id']
        # print("itm= ", itm)
        # print("name= ", name)
        if idx == 3 or name == 'car':
            thisbox = boxes[i]
            # print("thisbox= ", thisbox)
            ymin = int(thisbox[0] * height)
            xmin = int(thisbox[1] * width)
            ymax = int(thisbox[2] * height)
            xmax = int(thisbox[3] * width)
            area = (xmax - xmin) * (ymax - ymin)
            if area > max_area:
                max_area = area
                max_idx = i
    if max_idx >= 0:
        # for i in range(num):
        i = max_idx
        idx = classes[i]
        # print("idx= ", idx)
        itm = category_index[idx]
        name = itm['name']
        idx = itm['id']
        # print("itm= ", itm)
        # print("name= ", name)
        if idx == 3 or name == 'car':
            print("find the car")
            class_ids = np.unique(instance_masks[i])
            # print("class_ids= ", class_ids)
            # layer = np.where(instance_masks != idx, 0, 255)
            layer = np.where(instance_masks[i] != 1, 0, 255)
            # print("layer=",layer)
            grayvalues = np.unique(layer)
            # print("grayvalues= ", grayvalues)

            layer2 = layer.astype(np.uint8)
            grayvalues = np.unique(layer2)
            # print("layer2= ", layer2)
            # print("grayvalues= ", grayvalues)

            contours, hierarchy = cv2.findContours(layer2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            h, w = layer2.shape[:2]

            ###绘制轮廓
            frame3 = cv2.merge([layer2, layer2, layer2])
            # frame3 = frame.copy()
            frame3 = cv2.drawContours(frame3, contours, -1, (255, 255, 255,), 90)
            frame3 = cv2.bitwise_and(frame, frame3)
            # cv2.imshow("frame3", frame3)
            frame2 = frame3
    return frame2


'''     
            k = len(contours)
            idx2 = 0
            for cnt in contours:
                # 用绿色(0, 255, 0)来画出最小的矩形框架
                x, y, w, h = cv2.boundingRect(cnt) #计算每个轮廓的box
                print("(x, y, w, h)= ", (x, y, w, h))
                xc = x + (w >> 1)
                yc = y + (h >> 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                        
                #h, w = image.shape[:2]  # 读取图像的宽和高
                #mask = np.zeros([h + 2, w + 2], np.uint8)  # 新建图像矩阵  +2是官方函数要求
                #mask[1:h+1,1:w+1] = layer2
                #cv2.floodFill(frame, mask, (100, 100), (0, 0, 255), cv2.FLOODFILL_MASK_ONLY)
                
                                        
                x1, y1, w1, h1 = cv2.boundingRect(contours1[idx2]) #计算每个轮廓的box
                print("(x1, y1, w1, h1)= ", (x1, y1, w1, h1))
                xc1 = x1 + (w1 >> 1)
                yc1 = y1 + (h1 >> 1)
                dx = xc1 - xc
                dy = yc1 - yc
                print("(dx, dy)= ", (dx, dy))
                                        
                layer4 = layer2.copy()
                height, width = layer2.shape[:2]  # 读取图像的宽和高
                layer4 = layer3[dy:(dy+height), dx:(dx + width)]
                                                                            
                mask_copy = cv2.merge([layer4,layer4,layer4])
                #mask_copy = cv2.merge([layer2,layer2,layer2])#要进行图像操作通道数目要一致,用merge函数合并出一个三通道图像
                frame2 = cv2.bitwise_and(frame, mask_copy)#两个图像相与,mask中非标记区域为0与出来的结果也是0,起到屏蔽作用
                idx2 += 1
    return frame2
'''


class TOD(object):
    def __init__(self):
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/test/result/frozen_inference_graph.pb'
        # self.PATH_TO_LABELS = r'/home/gxh/works/tf/test/pet_label_map_gxh.pbtxt'
        self.PATH_TO_CKPT='/home/ubutnu/work/dataArea/gxh-1/model/mask_rcnn_inception_resnet_v2_atrous-frozen_inference_graph.pb'
        self.PATH_TO_LABELS='/home/ubutnu/work/dataArea/gxh-1/model/mscoco_label_map.pbtxt'
        self.NUM_CLASSES = 1
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/test/model/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/test/model/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'
        # self.PATH_TO_LABELS = r'/home/gxh/works/tf/test/pet_label_map.pbtxt'
        # self.PATH_TO_LABELS = r'/home/gxh/works/tf/test/mscoco_label_map.pbtxt'
        # self.NUM_CLASSES = 37
        # self.NUM_CLASSES = 90
        ###
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/test/result/frozen_inference_graph.pb'
        # self.PATH_TO_LABELS = r'/home/gxh/works/tf/test/pet_label_map_gxh.pbtxt'
        # self.NUM_CLASSES = 1
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/direction_train/result-no-crop-6k-ss-4/frozen_inference_graph.pb'
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/direction_train/result-f-jy-damage/frozen_inference_graph.pb'
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/jy/mask_rcnn_resnet101_atrous_coco_2018_01_28/frozen_inference_graph.pb'
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/jy/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/jy/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/direction_train/result/frozen_inference_graph.pb'
        #self.PATH_TO_LABELS = r'/home/gxh/works/tf/direction_train/data/direction_label_map_gxh.pbtxt'
        # self.PATH_TO_LABELS = r'/home/gxh/works/tf/direction_train/data/damage_label_map_gxh.pbtxt'
        # self.PATH_TO_LABELS = r'/home/gxh/works/tf/direction_train/data/mscoco_label_map.pbtxt'
        self.NUM_CLASSES = 8
        self.NUM_CLASSES = 52
        self.NUM_CLASSES = 81
        self.NUM_CLASSES = 9
        self.NUM_CLASSES = 90
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()
        self.tensor_dict = {}
        self.width = 1000
        self.height = 1000

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def _load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # writer = tf.summary.FileWriter("/Users/xiaohuigao/works/tf/test/logs/", sess.graph)
                # sess.run(tf.global_variables_initializer())
                start = time.clock()
                start_time = time.time()
                # 这个array在之后会被用来准备为图片加上框和标签
                # image_np = self._load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                # used for plt
                image_np = np.array(image).astype(np.uint8)
                image_np = image
                # 扩展维度，应为模型期待: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                # 每个框代表一个物体被侦测到.
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                # 每个分值代表侦测到物体的可信度.
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                # 执行侦测任务.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
                # Visualization of the results of a detection.
                # 图形化.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=2)
                end = time.clock()

                # print 'frame:',1.0/(end - start)
                # plt.figure(figsize=IMAGE_SIZE)
                # plt.imshow(image_np)
                # plt.show()
                cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
                cv2.imshow("detection", image)
                cv2.waitKey(0)

    def cropjpg(self, orgImg, dst, x0, x1, y0, y1):
        if int(x0) < 0:
            x0 = 0
        if int(y0) < 0:
            y0 = 0
        # orgImg = cv2.imread(src, cv2.CV_LOAD_IMAGE_COLOR)
        # cv2.namedWindow("org", cv2.cv.CV_WINDOW_NORMAL)
        # cv2.moveWindow("org", 100, 100)
        # cv2.imshow("org", orgImg)
        # roiImg = orgImg[int(x0):int(x1),int(y0):int(y1)]
        roiImg = orgImg[int(y0):int(y1), int(x0):int(x1)]
        cv2.imwrite(dst, roiImg)
        # cv2.namedWindow("crop", cv2.cv.CV_WINDOW_NORMAL)
        # cv2.moveWindow("crop", 200, 200)
        # cv2.imshow("crop", roiImg)
        # cv2.waitKey(0)

    def cropImgs(self, orgImg, dstDir, boxs, names):
        # print datetime.datetime.now()
        thistime = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        sp = orgImg.shape
        height = sp[0]
        width = sp[1]
        newbox = []
        count = 0;
        for thisbox in boxs:
            ymin = thisbox[0] * height
            xmin = thisbox[1] * width
            ymax = thisbox[2] * height
            xmax = thisbox[3] * width
            w = xmax - xmin
            h = ymax - ymin
            xmin = xmin - 0.15 * w
            xmax = xmax + 0.15 * w
            x0 = int(xmin)
            x1 = int(xmax)
            if int(x0) < 0:
                x0 = 0
            if int(x1) > width:
                x1 = width
            w = x1 - x0
            w = (w >> 4) << 4
            x1 = x0 + w

            ymin = ymin - 0.15 * h
            ymax = ymax + 0.15 * h
            y0 = int(ymin)
            y1 = int(ymax)
            if int(y0) < 0:
                y0 = 0
            if int(y1) > height:
                y1 = height
            h = y1 - y0
            h = (h >> 4) << 4
            y1 = y0 + h

            newbox.append((y0, x0, y1, x1))
            # print newbox
            # print names[count][0]
            classname = names[count][0].split(':')[0]
            strscores = names[count][0].split(':')[1]
            strscores = strscores.split('%')[0]
            # print strscores
            scores = float(strscores) / 100
            # print scores
            targetFile = dstDir
            ###targetFile = os.path.join(dstDir,  classname)
            if not os.path.isdir(targetFile):
                os.makedirs(targetFile)
            dstname = thistime + '-' + str(count) + '.jpg'
            targetFile = os.path.join(targetFile, dstname)
            # print targetFile
            self.cropjpg(orgImg, targetFile, x0, x1, y0, y1)
            count += 1

    def detect_all(self):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/raccoon_dataset/images/'
                # PATH_TEST_IMAGE = r'/home/gxh/works/tf/jy/coco/val2014/'
                # PATH_TEST_IMAGE = r'/home/gxh/works/tf/jy/2017.11.01/1/'
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/jy/distill-2/train/'
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/jy/dazhong/'
                # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/output-0/xingbian/'
                # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/output-0/whole/'
                # PATH_TEST_IMAGE = r'/home/gxh/works/tf/jy/distill-2/val/'
                # PATH_TEST_IMAGE = r"/home/gxh/works/tf/jy/fangwei/"
                # PATH_TEST_IMAGE = r'/home/gxh/works/tf/jy/wgl/FFOutput/'
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/output_img/'
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/crop-lingjian/'
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/lingjian_img/'
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/test/gxh3/'
                PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/from-xq/evl/'
                count = 0
                for root, dirs, files in os.walk(PATH_TEST_IMAGE):
                    for afile in files:
                        ffile = root + afile
                        if ffile[ffile.rindex("."):].lower() in imgTypes:
                            print ('src=', ffile)
                            image = cv2.imread(ffile)
                            sp = image.shape
                            print (sp)
                            height = sp[0]
                            width = sp[1]
                            # img2 = cv2.resize(image,(1288,720),interpolation=cv2.INTER_CUBIC)
                            # image = img2
                            # writer = tf.summary.FileWriter("/Users/xiaohuigao/works/tf/test/logs/", sess.graph)
                            # sess.run(tf.global_variables_initializer())
                            start = time.clock()
                            start_time = time.time()
                            # 这个array在之后会被用来准备为图片加上框和标签
                            # image_np = self._load_image_into_numpy_array(image)
                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            # used for plt
                            image_np = np.array(image).astype(np.uint8)
                            image_np = image
                            # 扩展维度，应为模型期待: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image, axis=0)
                            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                            # 每个框代表一个物体被侦测到.
                            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                            # 每个分值代表侦测到物体的可信度.
                            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                            # Actual detection.
                            # 执行侦测任务.
                            (boxes, scores, classes, num_detections) = sess.run(
                                [boxes, scores, classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})
                            print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
                            # Visualization of the results of a detection.
                            # 图形化.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                self.category_index,
                                use_normalized_coordinates=True,
                                line_thickness=2)
                            # print 'category_index =',self.category_index
                            print (vis_util.categoriesDetected)
                            # print vis_util.boxDetected
                            end = time.clock()
                            # print 'frame:',1.0/(end - start)
                            # plt.figure(figsize=IMAGE_SIZE)
                            # plt.imshow(image_np)
                            # plt.show()
                            count += 1
                            ###image2 = cv2.imread(ffile)
                            ###self.cropImgs(image2, targetDir, vis_util.boxDetected, vis_util.categoriesDetected)
                            print('count= ', count)
                            ###
                            cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
                            cv2.imshow("detection", image)
                            cv2.waitKey(0)

    def run_inference_for_single_image(self, image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    # And instead of running single image inference, you should do object detection on a batch of images.
    # The images need to be of same size though.
    def run_inference_one_image(self, image, sess, graph):
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        # tensor_dict = {}
        tensor_dict = self.tensor_dict
        if (len(tensor_dict) == 0) or (self.width != image.shape[1]) or (self.height != image.shape[0]):
            print ("init tensor_dict")
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                self.tensor_dict = tensor_dict
                self.width = image.shape[1]
                self.height = image.shape[0]
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        # Run inference
        print ("###########################################000000000000000000000")
        print (image.shape)
        # start_time = time.time()
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        # print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
            print ("with mask")
        return output_dict

    def detect_all2(self):
        # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea2/raw-data/val2017/'
        # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/back/dazhong/'
        # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/back/dazhong-3/'
        # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/imgnet/damage/'
        # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/zhanma/'
        # PATH_TEST_IMAGE = r'/home/gxh/works/tf/dataArea/back/fangwei/'
        PATH_TEST_IMAGE='/home/ubutnu/work/dataArea/gxh-1/jpg/'
        count = 0
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                print ("sess#######################################################")
                # src_dir = '../../dataArea/from-xq/banghao/distill-new-part0-guaca/a-0/'
                src_dir = PATH_TEST_IMAGE
                count = 0
                count = 0
                count2 = 0
                deletelis = []
                filelist0 = os.listdir(src_dir)
                filelist = []
                for ffile in filelist0:
                    if ffile[ffile.rindex("."):].lower() in imgTypes:
                        filelist.append(ffile)
                n = len(filelist)  # / 2
                root = src_dir
                idx = 0
                status = True
                # for afile in filelist:
                # for idx in range(len(filelist)):
                while idx < len(filelist):
                    afile = filelist[idx]
                    idx += 1
                    # print(afile)
                    ffile = os.path.join(root, afile)
                    if ffile != None:
                        # for root,dirs,files in os.walk(PATH_TEST_IMAGE):
                        #    for afile in files:
                        #        ffile=root+afile
                        if ffile[ffile.rindex("."):].lower() in imgTypes:
                            image_path = ffile
                            # image = Image.open(image_path)

                            start_time = time.time()

                            image = cv2.imread(ffile)
                            img2 = cv2.resize(image, (1000, 1000), interpolation=cv2.INTER_CUBIC)
                            image = img2
                            frame = image.copy()
                            # the array based representation of the image will be used later in order to prepare the
                            # start = time.clock()
                            ##start_time = time.time()
                            # result image with boxes and labels on it.
                            # image_np = self._load_image_into_numpy_array(image)
                            ##image_np = np.array(image).astype(np.uint8)
                            image_np = image
                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)
                            # Actual detection.
                            output_dict = self.run_inference_one_image(image_np, sess, self.detection_graph)
                            ##print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
                            # Visualization of the results of a detection.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                output_dict['detection_boxes'],
                                output_dict['detection_classes'],
                                output_dict['detection_scores'],
                                self.category_index,
                                instance_masks=output_dict.get('detection_masks'),
                                use_normalized_coordinates=True,
                                line_thickness=8)
                            ###

                            frame2 = jpg_mask(output_dict, frame, self.category_index)
                            print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))

                            # cv2.waitKey(0)
                            count += 1
                            print('count= ', count)
                            ###
                            if SHOW_MODE:
                                cv2.imshow("frame", frame2)
                                thisText = "(" + str(count2) + "/" + str(count) + ")" + " / " + str(n)
                                cv2.putText(image, thisText, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                                cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
                                cv2.imshow("detection", image)

                                k = cv2.waitKey(0) & 0xff
                                if k == ord('q'):
                                    cv2.destroyAllWindows()
                                    break
                                elif k == ord('b'):
                                    print("key down b: ", idx)
                                    # cv2.destroyAllWindows()
                                    idx -= 2
                                    count -= 2
                                    if idx < 0:
                                        idx = 0
                                    if count < 0:
                                        count = 0
                                    continue
                print("finish!")
                cv2.destroyAllWindows()


if __name__ == '__main__':
    print('Start detection.')
    detecotr = TOD()
    detecotr.detect_all2()
    #####################
    ##PATH_TEST_IMAGE = r'/Users/xiaohuigao/works/tf/mydata/car-1/20171030_143218.jpg'
    ##PATH_TEST_IMAGE = r'/Users/xiaohuigao/works/tf/raccoon_dataset/images/raccoon-12.jpg'
    # PATH_TEST_IMAGE = r'/Users/xiaohuigao/works/tf/mydata/images/Abyssinian_100.jpg'
    # PATH_TEST_IMAGE = r'/Users/xiaohuigao/works/tf/mydata/images/yorkshire_terrier_1.jpg'
    # PATH_TEST_IMAGE = r'/Users/xiaohuigao/works/tf/mydata/coco/val2014/COCO_val2014_000000000257.jpg'
    # PATH_TEST_IMAGE = r'/Users/xiaohuigao/works/tf/mydata/car-1/20171030_143218.jpg'
    # PATH_TEST_IMAGE = image_path
    ##img1 = Image.open(PATH_TEST_IMAGE)
    # plt.figure(figsize=IMAGE_SIZE)
    # plt.imshow(img1)
    # plt.show()
    # plt.pause(5)
    # plt.close()
    # cv2.imshow('resize',img2)
    # cv2.waitKey(0)
    # img2 = cv2.imread(PATH_TEST_IMAGE)
    # img2 = cv2.resize(img1,(1288,720),interpolation=cv2.INTER_CUBIC)
    # cv2.imshow("detection", img2)
    # cv2.waitKey(0)
    # detecotr = TOD()
    # detecotr.detect(img2)
