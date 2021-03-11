#為了不出現 OMP: Error #15
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img = cv2.imread('img/redCar.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# matplotlib需使用RGB
# plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB))

#優化過濾
bfilter = cv2.bilateralFilter(gray_img,11,17,17)
#邊緣檢測
edged = cv2.Canny(bfilter,30,200)
# plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

#找輪廓 由簡單的線條表示
keypoints = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#imutils取輪廓並傳回
contours = imutils.grab_contours(keypoints)
#排序 由上到下找 返回前10個輪廓
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]

#取四邊形位置
location = None
for contour in contours:
    #數字越高 越粗略計算圖形
    approx = cv2.approxPolyDP(contour,10,True)
    if len(approx) == 4:
        location = approx
        break
print(location)

#遮罩得到車牌 先創一個和圖形大小一樣的空白
mask = np.zeros(gray_img.shape, np.uint8)
#在圖像繪製輪廓 
new_img = cv2.drawContours(mask,[location],0,255,-1)
#僅顯示位置的部位
new_img = cv2.bitwise_and(img,img,mask=mask)
# plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

#選取白色部位
(x,y) = np.where(mask==255)
#左上角
(x1,y1) = (np.min(x),np.min(y))
#右下角
(x2,y2) = (np.max(x),np.max(y))
#擷取車牌部位
cropped_img = gray_img[x1:x2+1,y1:y2+1]
plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

#使用的語言
reader = easyocr.Reader(['en'])
#讀取裁減後圖片(車牌)的文字
result = reader.readtext(cropped_img)
#print(result)

#車牌文字位置 從後面數來第二個
text = result[0][-2]
#字體
font = cv2.FONT_HERSHEY_SIMPLEX
#圖片 文字型態
res = cv2.putText(img, text=text, org=(approx[0][0][0],approx[1][0][1]+30),fontFace=font,fontScale=1,color=(255,255,0),thickness=2,lineType=cv2.LINE_AA)
#繪製矩形
res = cv2.rectangle(img,tuple(approx[0][0]),tuple(approx[2][0]),(255,255,0),3)
plt.imshow(cv2.cvtColor(res,cv2.COLOR_BGR2RGB))