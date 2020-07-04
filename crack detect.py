from skimage import morphology,draw
import numpy as np
import cv2
import matplotlib.pyplot as plt
#加载图像
img= cv2.imread("图片路径")
#去除光照阴影
def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver
    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3,3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    return dst
if __name__ == '__main__':
    blockSize = 30
    dst = unevenLightCompensate(img, blockSize)
    cv2.namedWindow("result",0)
    cv2.imshow('result', dst)
    cv2.waitKey(0)
# 线性变换图像增强
out = 2.0*dst
out[out > 255] = 255# 进行数据截断，大于255的值截断为255
# 数据类型转换
out = np.around(out)
out = out.astype(np.uint8)
#OTSU阈值分割
out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
ret1, th = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#颜色反转
src=np.repeat(th[...,np.newaxis],3,2)
height, width, channels = src.shape
for row in range(height):
    for list in range(width):
        for c in range(channels):
            pv = src[row, list, c]
            src[row, list, c] = 255 - pv
#连通域分析
#src=np.repeat(src[...,np.newaxis],3,2)
w,h,n = src.shape
im = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
nccomps = cv2.connectedComponentsWithStats(im)
_ = nccomps[0]
labels = nccomps[1]
centroids = nccomps[3]
status = nccomps[2]
aa = []
for j in range(len(status)):
    i = status[j]
    aa.append(i[4])
print(aa)
n = len(aa)
for x in range(n-1):
    for y in range(n-1-x):
        if aa[y]>aa[y+1]:
            aa[y],aa[y+1] = aa[y+1],aa[y]
print(aa)
v = aa[-2]
#以动态阈值抽取目标区域
_, labels, stats, centroids = cv2.connectedComponentsWithStats(im)
i=0
for istat in stats:
    if istat[4]<v:
        #print(istat[0:2])
        if istat[3]>istat[4]:
            r=istat[3]
        else:r=istat[4]
        cv2.rectangle(im,tuple(istat[0:2]),tuple(istat[0:2]+istat[2:4]) , 0,thickness=-1)  
    i=i+1
cra = im.copy()
#面积计算
#im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    area = cv2.contourArea(contours[c])
    im=np.repeat(im[...,np.newaxis],3,2)
    cv2.drawContours(img, contours, c, (0, 0, 255), 2)
b=w*h
print(area/b)
#长度与宽度计算
image = np.repeat(cra[...,np.newaxis],3,2)
skeleton =morphology.skeletonize(image)
skeleton=cv2.cvtColor(skeleton,cv2.COLOR_BGR2GRAY)
contours = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
skeleton=np.repeat(skeleton[...,np.newaxis],3,2)
for c in contours:
    cv2.drawContours(skeleton, [c], -1, (255, 255, 255), 1)
a = cv2.cvtColor(skeleton,cv2.COLOR_BGR2GRAY)
a= cv2.dilate(a,kernel = np.ones((3,3),np.uint8))
contours, hierarchy = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    arclen = cv2.arcLength(contours[c], True)
    a=np.repeat(a[...,np.newaxis],3,2)
    cv2.putText(img, "Length:" + str(round(arclen/2,2)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, .60, (0, 255, 255),2)
    cv2.putText(img, "width:" + str(round(2*area/arclen,2)), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, .60, (0, 255, 255),2)
cv2.imshow("img-out",img)
cv.imwrite("输出图像路径", img)
cv2.waitKey(0)
cv2.destroyAllWindows()





