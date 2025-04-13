import numpy as np
import cv2
import matplotlib.pyplot as plt
# heat 为某层的特征图，自己手动获取
heat = cv2.imread('/data3/YG/FRINet/code/pred/1/COD10K-CAM-1-Aquatic-4-Crocodile-114.png')
heat = np.transpose(heat, (2, 0, 1))
# heat = heat.data.cpu().numpy()	     # 将tensor格式的feature map转为numpy格式
# heat = np.squeeze(heat, 0)	         # ０维为batch维度，由于是单张图片，所以batch=1，将这一维度删除
# heat = heat[145*3:145*3+3,:]　       # 切片获取某几个通道的特征图
heatmap = np.maximum(heat, 0)        # heatmap与0比较
heatmap = np.mean(heatmap, axis=0)  # 多通道时，取均值
heatmap /= np.max(heatmap)          # 正则化到 [0,1] 区间，为后续转为uint8格式图做准备
#plt.matshow(heatmap)				# 可以通过 plt.matshow 显示热力图
#plt.show()

# 用cv2加载原始图像
img = cv2.imread('/data3/YG/FRINet/code/data/visual/image/COD10K-CAM-1-Aquatic-4-Crocodile-114.jpg')
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 特征图的大小调整为与原始图像相同
heatmap = np.uint8(255 * heatmap)  # 将特征图转换为uint8格式
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将特征图转为伪彩色图
heat_img = cv2.addWeighted(img, 0.4, heatmap, 1, 0)     # 将伪彩色图与原始图片融合
#heat_img = heatmap * 0.5 + img 　　　　　　		  　　　 # 也可以用这种方式融合
cv2.imwrite('/data3/YG/FRINet/Crocodile.jpg', heat_img) # 将图像保存
