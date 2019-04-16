# -*- coding: utf-8 -*

import os
import sys
import cv2
import time
import datetime
import six.moves.urllib as urllib
import tarfile
import zipfile
import re
from collections import defaultdict
from io import StringIO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

PRJ_ROOT = r'/home/ubutnu/work/dataArea/gxh-1/'

PATH_TEST_IMAGE = PRJ_ROOT + r'/jpg/'
#PATH_TEST_IMAGE = r'/home/gxh/works/dataArea/share/data_center/others/'

sys.path.append("..")
from object_detection.utils import ops as utils_ops

# from object_detection.utils import ops as utils_ops


# from utils import label_map_util
# from utils import visualization_utils as vis_util

# 为了显示 pyplot 的 figure， 需要提前设置 matplotlib 的 backend.
# print(matplotlib.get_backend())  # 默认的是 agg， 即不显示 figure
# 编辑 from object_detection.utils import visualization_utils as vis_util 中的 visualization_utils.py，注释掉以下部分：
# import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements

#if tf.__version__ < '1.4.0':
#    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')
print(tf.__version__)


imgTypes = [".png", ".jpg", ".bmp"]
IMAGE_SIZE = (36, 24)


# print(sys.argv[0])
# targetDir = sys.argv[1]

zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
def contain_zh(word):
    #word = word.decode() #python2
    word = word.encode().decode()
    global zh_pattern
    match = zh_pattern.search(word)
    return match

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = PRJ_ROOT + r'model/mask_rcnn_inception_resnet_v2_atrous-frozen_inference_graph.pb'
        # self.PATH_TO_CKPT = r'/home/gxh/works/tf/direction_train/result/frozen_inference_graph.pb'
        #self.PATH_TO_LABELS = r'/home/gxh/works/tf/direction_train/data/direction_label_map_gxh.pbtxt'
        #self.PATH_TO_LABELS = r'/home/gxh/works/tf/direction_train/data/damage_label_map_gxh.pbtxt'
        self.PATH_TO_LABELS = PRJ_ROOT + r'model/mscoco_label_map.pbtxt'
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
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' #使用 GPU 0
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 使用 GPU 0，1
        ###动态申请显存
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)
        ###
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # config = tf.ConfigProto(allow_soft_placement=True, allow_soft_placement=True)
        # config.gpu_options.per_process_gpu_memory_fraction = 0.4
        # sess = tf.Session(config=config)
        self.sess = tf.Session(graph=self.detection_graph, config=config)

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
            dstname = thistime + '-' + str(count) + '.jpg';
            targetFile = os.path.join(targetFile, dstname)
            # print targetFile
            self.cropjpg(orgImg, targetFile, x0, x1, y0, y1)
            count += 1

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
        start_time = time.time()
        output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
        print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
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
        count = 0
        with self.detection_graph.as_default():
            #with tf.Session() as sess:
            sess = self.sess
            if sess is not None:
                print ("sess#######################################################")
                for root, dirs, files in os.walk(PATH_TEST_IMAGE):
                    for afile in files:
                        ffile = root + afile
                        ffile = os.path.join(root, afile)

                        if ffile[ffile.rindex("."):].lower() in imgTypes:
                            image_path = ffile
                            print ("ffile= ", ffile)
                            print ("count= ", count)
                            count += 1
                            # ffile2 = ffile.decode('gbk', 'ignore') #python2
                            # ffile2 = afile.encode().decode('gbk', 'ignore')
                            # if contain_zh(ffile2):
                            #    print("cn filename")
                            #    print ("ffile= ", ffile)
                            #    continue
                            image = Image.open(image_path)
                            image = cv2.imread(ffile)
                            img2 = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC)
                            image = img2
                            # the array based representation of the image will be used later in order to prepare the
                            # start = time.clock()
                            # start_time = time.time()
                            # result image with boxes and labels on it.
                            # image_np = self._load_image_into_numpy_array(image)
                            ##image_np = np.array(image).astype(np.uint8)
                            image_np = image
                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)
                            # Actual detection.
                            output_dict = self.run_inference_one_image(image_np, sess, self.detection_graph)
                            # print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
                            # Visualization of the results of a detection.
                            vis_util.visualize_boxes_and_labels_on_image_array(
                                image_np,
                                output_dict['detection_boxes'],
                                output_dict['detection_classes'],
                                output_dict['detection_scores'],
                                self.category_index,
                                instance_masks=output_dict.get('detection_masks'),
                                use_normalized_coordinates=True,
                                line_thickness=2)
                            mm = output_dict.get('detection_masks')
                            cl = np.unique(output_dict.get('detection_masks'))
                            aa = np.array(output_dict.get('detection_masks')).astype(np.uint8)
                            # for i in range(len(aa)):
                            #    for j in range(len(aa[0])):
                            #        print(aa[i][j])
                            for i in aa:
                                for j in i:
                                    for num in j:
                                        if num==1:
                                            print('This is front!')
                                            left=aa[num-1][j]
                                            print(left)
                                   # if point==1:
                                   #     print('This is front!')
                                       # left=aa[i-1][j]
                                       # pass
                                   # else:
                                   #     continue
                            # print(np.ndim(aa))
                            # for i in aa:
                            #     for j in range(0, len(aa[0])):
                            #         print(aa[i][j])
                            #         point = aa[i][j]
                            #         print(point)
                                    # if point == 1:
                                    #     left = aa[i-1][j]
                                    #     pass
                                    # else:
                                    #     continue
                                    # pass
                            # print(output_dict.get('detection_masks'))
                            #cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
                            cv2.imshow("detection", image)
                            k = cv2.waitKey(0) &0xff
                            if k == ord('q'):
                                cv2.destroyAllWindows()
                                break

    from PIL import Image
    def test(self, img_file):
        with self.detection_graph.as_default():
            sess = self.sess
            if sess is not None:
                image = cv2.imread(img_file)
                img2 = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_CUBIC)
                image = img2
                image_np = image
                image_np_expanded = np.expand_dims(image_np, axis=0)
                output_dict = self.run_inference_one_image(image_np, sess, self.detection_graph)
                mm = output_dict.get('detection_masks')
                print(mm)
                print(type(mm))

                img=Image.fromarray(np.uint8(mm)).convert('RGB')
                cv2.copyMakeBorder(img,10,10,10,10,cv2.BORDER_REFLECT)

                # img.show()
                # print(img)
                # img_data = tf.image.convert_image_dtype(mm, dtype=tf.uint8)
                # encoded_image = tf.image.encode_jpeg(mm)
                # print(encoded_image)
                # masked = cv2.bitwise_and(image_np, image_np, mask=mm)
                # masked=cv2.bitwise_not(mm)
                # cv2.imshow('mask',img_data)
                # cv2.waitKey(0)
                # cl = np.unique(output_dict.get('detection_masks'))
                # aa = np.array(output_dict.get('detection_masks')).astype(np.uint8)
                # print(type(aa))
                # print(np.shape(aa))
                # img= cv2.copyMakeBorder(mm, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
                # print(img)
                # height = aa.shape[0]
                # width = aa.shape[1]
                # print(height, width)
                # for i in range(height):
                #     for j in range(width):
                #         print(i,j)
                # list1=[]
                # for i in aa:
                #     for j in range(0,len(aa[0])):
                #         # print(aa[i][j])
                #         point=aa[i,j]
                #         print(point)
                #         if point==1:
                #             left=aa[i-1][j]
                #             right=aa[i+1][j]
                #             top=aa[i][j+1]
                #             blow=aa[i][j-1]
                #             if left==1:
                #                 list1.append(left)
                #             else:
                #                 continue
                #             if right==1:
                #                 list1.append(right)
                #             else:
                #                 continue
                #             if top==1:
                #                 list1.append(top)
                #             else:
                #                 continue
                #             if blow==1:
                #                 list1.append(blow)
                #             else:
                #                 continue
                #
                #         else:
                #             continue

                # for i in aa:
                #     for j in i:
                #         for num in j:
                #             if num == 1:
                #                 print('This is front!')
                #                 left = aa[num - 1][j]
                #                 print(left)

if __name__ == '__main__':
    print('Start detection.')
    detecotr = TOD()
    # detecotr.detect_all2()
    detecotr.test('/home/ubutnu/work/dataArea/gxh-1/jpg/17-CH.jpg')

