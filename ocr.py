# -*- coding: utf-8 -*-
import os, cv2
import argparse
import sys, numpy as np
import math
import json
import re


DEBUG = False

# PaddleOCR引擎（延迟初始化）
_paddle_ocr = None

def _get_ocr_engine():
    global _paddle_ocr
    if _paddle_ocr is None:
        from paddleocr import PaddleOCR
        _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
    return _paddle_ocr


def detect(img):
    """
    使用PaddleOCR识别身份证信息
    :param img: cv2读取的图片
    :return: (success, result_list_or_error_msg, tilt_path)
    """
    if img is None:
        return False, '无法读取图片', ''

    ocr_engine = _get_ocr_engine()

    # PaddleOCR识别
    result = ocr_engine.ocr(img, cls=True)

    if not result or not result[0]:
        return False, '无法识别，请换一张清晰度更高的照片', ''

    # 收集所有识别到的文本和位置
    texts = []
    for line in result[0]:
        box = line[0]       # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        text = line[1][0]   # 识别文本
        conf = line[1][1]   # 置信度
        # 计算文本行中心y坐标和x坐标，用于排序
        cy = (box[0][1] + box[2][1]) / 2
        cx = (box[0][0] + box[2][0]) / 2
        texts.append({
            'text': text,
            'cy': cy,
            'cx': cx,
            'conf': conf
        })

    # 按y坐标排序（从上到下）
    texts.sort(key=lambda t: t['cy'])

    # 提取身份证信息
    card_name = ''
    card_sex = ''
    card_ethnic = ''
    card_year = ''
    card_mon = ''
    card_day = ''
    card_addr = ''
    card_num = ''

    # 识别身份证号码（18位数字+X）
    for t in texts:
        num = re.sub(r'[^0-9Xx]', '', t['text'])
        if len(num) == 18 or len(num) == 15:
            card_num = num.upper()
            break

    if not card_num:
        return False, '未检测到身份证号码，请确认图片是否为身份证正面', ''

    # 从身份证号码提取出生日期
    if len(card_num) == 18:
        card_year = card_num[6:10]
        card_mon = str(int(card_num[10:12]))
        card_day = str(int(card_num[12:14]))
    else:
        card_year = '19' + card_num[6:8]
        card_mon = str(int(card_num[8:10]))
        card_day = str(int(card_num[10:12]))

    # 从身份证号码推断性别
    if int(card_num[-2]) % 2 == 0:
        card_sex = '女'
    else:
        card_sex = '男'

    # 识别姓名：通常在"姓名"关键字之后
    for i, t in enumerate(texts):
        if '姓名' in t['text']:
            # 姓名可能在同一行"姓名XXX"或下一行
            name_text = t['text'].replace('姓名', '').strip()
            if name_text:
                card_name = name_text
            elif i + 1 < len(texts):
                card_name = texts[i + 1]['text'].strip()
            break

    # 如果没找到"姓名"关键字，取最上面第一个短文本作为姓名
    if not card_name and texts:
        for t in texts:
            clean = t['text'].strip()
            if 2 <= len(clean) <= 4 and not any(kw in clean for kw in ['性别', '民族', '出生', '住址', '公民', '身份', '号码']):
                card_name = clean
                break

    # 识别性别和民族
    for t in texts:
        text = t['text']
        if '男' in text and card_sex == '':
            card_sex = '男'
        if '女' in text and card_sex == '':
            card_sex = '女'

        # 民族识别
        ethnic_match = re.search(r'(汉|蒙古|回|藏|维吾尔|苗|彝|壮|布依|朝鲜|满|侗|瑶|白|土家|哈尼|哈萨克|傣|黎|傈僳|佤|畲|高山|拉祜|水|东乡|纳西|景颇|柯尔克孜|土|达斡尔|仫佬|羌|布朗|撒拉|毛南|仡佬|锡伯|阿昌|普米|塔吉克|怒|乌孜别克|俄罗斯|鄂温克|德昂|保安|裕固|京|塔塔尔|独龙|鄂伦春|赫哲|门巴|珞巴|基诺|其他)', text)
        if ethnic_match:
            card_ethnic = ethnic_match.group(1)
        elif '民族' in text:
            ethnic_text = text.replace('民族', '').strip()
            if ethnic_text:
                card_ethnic = ethnic_text

    # 默认汉族
    if not card_ethnic:
        card_ethnic = '汉'

    # 识别住址：通常在"住址"关键字之后，可能跨多行
    addr_found = False
    addr_lines = []
    for i, t in enumerate(texts):
        if '住址' in t['text']:
            addr_part = t['text'].replace('住址', '').strip()
            if addr_part:
                addr_lines.append(addr_part)
            # 继续收集后续行直到遇到"公民身份号码"
            for j in range(i + 1, len(texts)):
                if '公民' in texts[j]['text'] or '身份' in texts[j]['text'] or '号码' in texts[j]['text']:
                    break
                addr_lines.append(texts[j]['text'].strip())
            addr_found = True
            break

    card_addr = ''.join(addr_lines)

    ret = [card_name, card_sex, card_ethnic, card_year, card_mon, card_day, card_addr, card_num]
    return True, ret, ''


def calculateElement(img):
    #根据图片大小粗略计算腐蚀 或膨胀所需核的大小
    sp = img.shape
    width = sp[1]  # width(colums) of image
    kenaly = math.ceil((width / 400.0) * 12)
    kenalx = math.ceil((kenaly / 5.0) * 4)
    a = (int(kenalx), int(kenaly))

    return a

def preprocess(gray, algoFunc):
    # 1. Sobel算子，x方向求梯度
    #sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)

    #获取二值化阈值
    thr = bz.myThreshold()
    #threshold = thr.get1DMaxEntropyThreshold(gray)
    threshold = getattr(thr, algoFunc)(gray)
    if threshold <= 0:
        raise Exception("获取二值化阈值失败")

    # 2. 二值化
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    #获取核大小
    calculateElement(gray)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, calculateElement(gray))

    #微处理去掉小的噪点
    dilation = cv2.dilate(binary, element1, iterations=1)
    binary = cv2.erode(dilation, element1, iterations=1)

    #文字膨胀与腐蚀使其连成一个整体
    erosion = cv2.erode(binary, element2, iterations=1)
    dilation = cv2.dilate(erosion, element1, iterations=1)

    # 7. 存储中间图片
    # cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)
    #
    # cv2.namedWindow("dilation2", cv2.WINDOW_NORMAL)
    # cv2.imshow("dilation2", erosion)
    # cv2.waitKey(0)
    #
    # cv2.namedWindow("dilation2", cv2.WINDOW_NORMAL)
    # cv2.imshow("dilation2", dilation)
    # cv2.waitKey(0)

    cv2.destroyAllWindows()
    #sys.exit(0)



    return dilation


def findTextRegion(img):
    region = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if(area < 1000):
            continue

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)

        # 计算高和宽 参考：http://blog.csdn.net/lanyuelvyun/article/details/76614872
        width = rect[1][0]
        hight = rect[1][1]

        # 筛选那些太细的矩形，留下扁的
        if hight > width:
            if hight < width * 5:
                continue
        else:
            if width < hight * 5:
                continue

        region.append(rect)

    return region


def nothing(x):
    pass


def fushiyupengzhang(pathtoimage):
    """
    腐蚀与膨胀动态取值预览
    :param pathtoimage:
    :return:
    """
    img = cv2.imread(pathtoimage)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #sobel = cv2.Sobel(im_gray, cv2.CV_8U, 1, 0, ksize=3)

    # 获取二值化阈值
    thr = bz.myThreshold()
    threshold = thr.getMinimumThreshold(im_gray)
    if threshold <= 0:
        raise Exception("获取二值化阈值失败")
    retval, img = cv2.threshold(im_gray, threshold, 255, cv2.THRESH_BINARY)

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.createTrackbar('Er/Di', 'image', 0, 1, nothing)
    # 创建腐蚀或膨胀选择滚动条，只有两个值
    cv2.createTrackbar('x', 'image', 0, 100, nothing)
    # 创建卷积核大小滚动条

    cv2.createTrackbar('y', 'image', 0, 100, nothing)

    while (1):
        s = cv2.getTrackbarPos('Er/Di', 'image')
        x = cv2.getTrackbarPos('x', 'image')
        y = cv2.getTrackbarPos('y', 'image')
        # 分别接收两个滚动条的数据

        if x == 0:
            x = 1
        if y == 0:
            y = 1

        k = cv2.waitKey(1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (x, y))
        # 根据滚动条数据确定卷积核大小
        erroding = cv2.erode(img, kernel, iterations=1)
        dilation = cv2.dilate(img, kernel, iterations=1)
        if k == 27:
            break
            # esc键退出
        if s == 0:
            cv2.imshow('image', erroding)
        else:
            cv2.imshow('image', dilation)
            # 判断是腐蚀还是膨胀


def  imgRotation(pathtoimg):
    #图片自动旋正
    from PIL import Image
    img = Image.open(pathtoimg)

    if hasattr(img, '_getexif') and img._getexif() is not None:
        # 获取exif信息
        dict_exif = img._getexif()
        if 274 in dict_exif:
            if dict_exif[274] == 3:
                #顺时针180
                new_img = img.rotate(-180)
                new_img.save(pathtoimg)
            elif dict_exif[274] == 6:
                #顺时针90°
                new_img = img.rotate(-90)
                new_img.save(pathtoimg)
            elif dict_exif[274] == 8:
                #逆时针90°
                new_img = img.rotate(90)
                new_img.save(pathtoimg)


    return None


def enhanceImage(pathtoimg):
    from PIL import Image
    from PIL import ImageEnhance

    # 原始图像
    image = Image.open(pathtoimg)

    # 对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    image_contrasted.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='path to image file')
    args = parser.parse_args()

    pathtoimg = args.image
    if not os.path.isfile(pathtoimg):
        print("请提供有效的图片文件路径")
        sys.exit(1)

    # for i in range(31, 40):
    #     pathtoimg = r'D:\OCR\p\w%s.jpg' % (i)
    #     #pathtoimg = r'D:\OCR\p\sam_xie.jpg'

    if DEBUG:
        pathtoimg = r'images\w1.jpg'

    # 读取文件
    img = cv2.imread(pathtoimg)

    try:
        ret, msg, path = detect(img)
        if path != '':
            # 读取文件
            img = cv2.imread(path)

            ret, msg, _ = detect(img)
            os.unlink(path)
            if ret:
                result = [{i: msg[i]} for i in range(len(msg))]
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(msg)

        else:
            if ret:
                result = [{i: msg[i]} for i in range(len(msg))]
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(msg)
    except Exception as e:
        print(e)

    #fushiyupengzhang(pathtoimg)