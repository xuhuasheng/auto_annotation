
# =========================================================
# @purpose: auto annotation mask and store to xml
# @date：   2019/9
# @version: v1.0
# @author： Xu Huasheng
# @github： https://github.com/xuhuasheng/auto_annotation
# =========================================================

import cv2
import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np 

PROJECT_PATH = '/home/watson/Documents/mask_THzDatasets/All_Data/'
IMGS_PATH = PROJECT_PATH + 'imgs/' # 图片读取路径
MASK_XMLS_PATH = PROJECT_PATH + 'xmls/' # 自动化mask标注xml存储路径
OLD_XMLS_PATH = '/home/watson/Documents/aug_THzDatasets/All_Data/xmls/' # 原先的xml标注文件



'''
purpose: 自动化标注mask，存储到xml
'''  
def auto_annotation():
    imgs_list = os.listdir(IMGS_PATH)
    old_xmls_list = os.listdir(OLD_XMLS_PATH)
    img_cnt = 0
    img_num = len(imgs_list)
    for img_fileName in imgs_list:
        img_cnt += 1
        img_fullFileName = IMGS_PATH + img_fileName
        print('auto_annotating(%d/%d)... %s' % (img_cnt, img_num, img_fullFileName))
        src_img = cv2.imread(img_fullFileName) 
        src_image = src_img.copy() # 拷贝原图的副本
        # cv2.imshow('src_img', src_img)

        img_name = img_fileName.split('.')[0] # 去掉后缀名，获得图片名字
        xml_name = img_name
        old_xml_fileName = xml_name + '.xml'

        # 判断对应xml文件是否存在
        if old_xml_fileName not in old_xmls_list:
            print('WARNING:' + old_xml_fileName + 'is not exist in' + OLD_XMLS_PATH)
            continue

        # 拷贝xml作为副本
        old_xml_fullFileName = os.path.join(OLD_XMLS_PATH, old_xml_fileName)
        mask_xml_fullFileName = MASK_XMLS_PATH + xml_name + '_mask.xml'
        shutil.copyfile(old_xml_fullFileName, mask_xml_fullFileName)

        # 解析xml副本
        tree = ET.parse(mask_xml_fullFileName)    # 解析xml元素树
        root = tree.getroot()                     # 获得树的根节点

        # 更新xml元素文本
        get_element(root, 'folder').text = IMGS_PATH.split('/')[-2]
        get_element(root, 'filename').text = img_fileName
        get_element(root, 'path').text = IMGS_PATH + img_fileName
        get_element(root, 'segmented').text = str(1)
        get_element(get_element(root, 'size'), 'depth').text = str(3)

        # 遍历目标
        for obj in get_elements(root, 'object'):
            
            # 获取bbox的坐标
            bndbox = get_element(obj, 'bndbox')
            xmin = int(get_element(bndbox, 'xmin').text) 
            ymin = int(get_element(bndbox, 'ymin').text) 
            xmax = int(get_element(bndbox, 'xmax').text)
            ymax = int(get_element(bndbox, 'ymax').text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            bbox_width = xmax - xmin
            bbox_height = ymax - ymin

            ###########################################################################
            # # 裁剪图片
            # crop_margin = 2 # 裁剪留白
            # crop_ymin = ymin - crop_margin
            # crop_ymax = ymax + crop_margin
            # crop_xmin = xmin - crop_margin
            # crop_xmax = xmax + crop_margin
            # obj_image = src_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            # # cv2.imshow('obj_image', obj_image)

            # # 图像处理：灰度、滤波、二值化、腐蚀、膨胀
            # processed_image = image_processing(obj_image)
            
            # # 寻找最大轮廓
            # maxContour = findMaxContour(processed_image)
            # # cv2.drawContours(obj_image, [maxContour], 0, (0,255,0), 1)  
            # # cv2.imshow('maxContour', obj_image)

            # # 获得maxContour坐标:下采样并展平
            # seg = get_ContourCoordinate(maxContour, subsampleRate = 23)

            # # 把目标轮廓坐标转化为全局坐标
            # seg_x = []
            # seg_y = []
            # seg_len = len(seg)
            # for i in range(seg_len):
            #     if i%2 == 0: # x
            #         seg[i] += crop_xmin
            #         seg_x.append(seg[i])
            #     else: # y
            #         seg[i] += crop_ymin
            #         seg_y.append(seg[i])
            # # print(seg)
            
            # # 测试：显示mask
            # if get_element(obj, 'name').text == 'gun':
            #     draw_mask_edge(src_image, seg_x, seg_y, (0, 255, 0), 2)
            # elif get_element(obj, 'name').text == 'phone':
            #     draw_mask_edge(src_image, seg_x, seg_y, (255, 0, 0), 2)

            # # 创建并添加segmentation子元素
            # element_segmentation = create_element('segmentation', {}, str(seg))
            # add_childElement(obj, element_segmentation)
            ############################################################################


            # 如果目标为gun
            if get_element(obj, 'name').text == 'gun':
                
                # 裁剪图片
                crop_margin = 2 # 裁剪留白
                crop_ymin = ymin - crop_margin
                crop_ymax = ymax + crop_margin
                crop_xmin = xmin - crop_margin
                crop_xmax = xmax + crop_margin
                obj_image = src_image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
                # cv2.imshow('obj_image', obj_image)

                # 图像处理：灰度、滤波、二值化、腐蚀、膨胀
                processed_image = image_processing(obj_image)
                
                # 寻找最大轮廓
                maxContour = findMaxContour(processed_image)
                # cv2.drawContours(obj_image, [maxContour], 0, (0,255,0), 1)  
                # cv2.imshow('maxContour', obj_image)

                # 获得maxContour坐标:下采样并展平
                seg = get_ContourCoordinate(maxContour)
                # print(len(seg))

                # 把目标轮廓坐标转化为全局坐标
                seg_x = []
                seg_y = []
                seg_len = len(seg)
                for i in range(seg_len):
                    if i%2 == 0: # x
                        seg[i] = float(seg[i] + crop_xmin)
                        seg_x.append(seg[i])
                    else: # y
                        seg[i] = float(seg[i] + crop_ymin)
                        seg_y.append(seg[i])
                # print(seg)
                
                # 测试：显示mask
                draw_mask_edge(src_image, seg_x, seg_y, (0, 255, 0), 2)

                # 创建并添加segmentation子元素
                element_segmentation = create_element('segmentation', {}, str(seg))
                add_childElement(obj, element_segmentation)

            # 如果目标为phone
            elif get_element(obj, 'name').text == 'phone':
                seg = []
                #left_top
                seg.append(xmin)
                seg.append(ymin)
                #left_bottom
                seg.append(xmin)
                seg.append(ymin + bbox_height)
                #right_bottom
                seg.append(xmin + bbox_width)
                seg.append(ymin + bbox_height)
                #right_top
                seg.append(xmin + bbox_width)
                seg.append(ymin)

                # 测试：显示mask
                seg_x = seg[::2] # 取奇数项
                seg_y = seg[1::2] # 取偶数项
                draw_mask_edge(src_image, seg_x, seg_y, (255, 0, 0), 2)


                # 创建并添加segmentation子元素
                element_segmentation = create_element('segmentation', {}, str(seg))
                add_childElement(obj, element_segmentation)

        # cv2.imshow('mask', src_image)
        # cv2.waitKey(0)
        # 写入xml
        tree.write(mask_xml_fullFileName) 



'''
input：
    @root: 根节点  
    @childElementName: 字节点tag名称
output：
    @elements:根节点下所有符合的子元素对象    
''' 
def get_elements(root, childElementName):
    elements = root.findall(childElementName)
    return elements


'''
input：
    @root: 根节点  
    @childElementName: 字节点tag名称
output：
    @elements:根节点下第一个符合的子元素对象    
''' 
def get_element(root, childElementName):
    element = root.find(childElementName)
    return element


'''
input：
    @tag: 元素tag名称  
    @property_map: 元素属性的键值对
    @content:元素文本内容
output：
    @element:新建的元素对象   
''' 
def create_element(tag, property_map, content):
    element = ET.Element(tag, property_map)
    element.text = content
    return element


'''
purpose: 添加子节点
input：
    @parentElement: 父节点  
    @childElement: 子节点
'''        
def add_childElement(parentElement, childElement):
    parentElement.append(childElement)


'''
purpose: 图像处理：灰度、滤波、二值化、腐蚀、膨胀
input：
    @input_image: 输入图片  
output: 
    @output_image: 输出图片
'''    
def image_processing(input_image):
    '''
    下面的参数基本调到最佳，不要轻易改动
    '''
    # 灰度
    obj_gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('obj_gray_image', obj_gray_image)

    # 均值滤波
    obj_blur_image = cv2.GaussianBlur(obj_gray_image, (11,11), 7)
    # cv2.imshow('obj_blur_image', obj_blur_image)

    # 自适应二值化
    obj_binary_image = cv2.adaptiveThreshold(obj_blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 23, 3)
    # cv2.imshow('obj_binary_image', obj_binary_image)

    # 腐蚀
    obj_erosion_image = cv2.erode(obj_binary_image, np.ones((3, 3), np.uint8))  
    # cv2.imshow('erode', obj_erosion_image)
    # 膨胀
    obj_dilation_image = cv2.dilate(obj_erosion_image, np.ones((5, 5), np.uint8)) 
    # cv2.imshow('dilate', obj_dilation_image)

    output_image = obj_dilation_image
    return output_image


'''
input：
    @binary_image:二值化图片  
output：
    @maxContour:最大轮廓    
''' 
def findMaxContour(binary_image):
    # 寻找所有轮廓
    image, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 轮廓面积
    contour_Area = 0
    # 遍历所有轮廓 寻找面积最大的轮廓
    for i in range(len(contours)):
        contour_Area_temp = cv2.contourArea(contours[i])
        if contour_Area_temp > contour_Area:
            contour_Area = contour_Area_temp
            maxContour = contours[i] # 最大的轮廓

    return maxContour


'''
purpose: 获得并下采样轮廓坐标，并展平到list
input：
    @maxContour:轮廓  
    @subsampleRate: 下采样率
output：
    @seg:轮廓坐标list    
''' 
def get_ContourCoordinate(contour, subsampleRate = None, coordinateNum = None):
    seg = []
    contour_length = contour.shape[0]
    if coordinateNum == None:
        coordinateNum = 20 # 控制输出的轮廓坐标点的个数
    if subsampleRate == None:
        subsampleRate = int(contour_length / coordinateNum)
    for i in range(contour_length):
        if i % subsampleRate == 0: # 下采样轮廓坐标
            seg.append(contour[i].tolist())

    # 降维到一维list    
    str_seg = str(seg) #转化为字符串
    str_seg = str_seg.replace('[','').replace(']','') # 替换掉'['和']'
    seg = list(eval(str_seg)) # 最后转化成列表

    return seg


'''
purpose: 画mask边界
input：
    @src_image:轮廓  
    @seg_x: 轮廓x坐标
    @seg_y: 轮廓y坐标   
    @color
    @thickness
''' 
def draw_mask_edge(src_image, seg_x, seg_y, color, thickness):
    start_point = (int(seg_x[0]), int(seg_y[0]))
    for j in range(len(seg_x) - 1):
        end_point = (int(seg_x[j+1]), int(seg_y[j+1]))
        cv2.line(src_image, start_point, end_point, color, thickness)
        start_point = end_point 
    cv2.line(src_image, start_point, (int(seg_x[0]), int(seg_y[0])), color, thickness)


if __name__ == "__main__":
    print('auto_annotation start!')
    auto_annotation()
    print('auto_annotation finished!')


    