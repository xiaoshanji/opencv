# 获取摄像头

```python
import cv2

if __name__ == '__main__':

    cv2.namedWindow('video',cv2.WINDOW_NORMAL) # 创建一个窗口
    cv2.resizeWindow('video',800,600)

    cap = cv2.VideoCapture(0) #打开摄像头
    # cap = cv2.VideoCapture("E:\资源\视频\阳光电影www.ygdy8.com.误杀.HD.1080p.国语中英双字.mp4") #打开视频

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        cv2.imshow('video',frame)


        key = cv2.waitKey(1000 // 48)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1.0:
            break

    cap.release()
    cv2.destroyAllWindows()
```



# 录制视频

```python
import cv2


if __name__ == '__main__':

    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video',800,600)

    cap = cv2.VideoCapture(0) #打开摄像头

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # *:解包操作，视频类型为 mp4
    # 文件名，格式，帧率，分辨率
    vw = cv2.VideoWriter('output.mp4',fourcc,1000 // 48,(640,480))

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        vw.write(frame) # 写入视频文件
        cv2.imshow('video',frame)

        key = cv2.waitKey(1000 // 48)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1.0:
            break

    cap.release()
    vw.release() # 保存到磁盘
    cv2.destroyAllWindows()
```



# 鼠标事件

```python

Event:
#define CV_EVENT_MOUSEMOVE 0                滑动
#define CV_EVENT_LBUTTONDOWN 1            左键点击
#define CV_EVENT_RBUTTONDOWN 2            右键点击
#define CV_EVENT_MBUTTONDOWN 3            中键点击
#define CV_EVENT_LBUTTONUP 4              左键放开
#define CV_EVENT_RBUTTONUP 5              右键放开
#define CV_EVENT_MBUTTONUP 6              中键放开
#define CV_EVENT_LBUTTONDBLCLK 7          左键双击
#define CV_EVENT_RBUTTONDBLCLK 8          右键双击
#define CV_EVENT_MBUTTONDBLCLK 9          中键双击

flags:
#define CV_EVENT_FLAG_LBUTTON 1           左键拖曳
#define CV_EVENT_FLAG_RBUTTON 2           右键拖曳
#define CV_EVENT_FLAG_MBUTTON 4           中键拖曳
#define CV_EVENT_FLAG_CTRLKEY 8         (8~15)按Ctrl不放事件
#define CV_EVENT_FLAG_SHIFTKEY 16       (16~31)按Shift不放事件
#define CV_EVENT_FLAG_ALTKEY 32         (32~39)按Alt不放事件



import cv2

# 鼠标事件处理函数，参数为：事件类型，鼠标坐标，鼠标状态（组合时使用），绑定事件时传递的参数
def mouse_event_callback(event,x,y,flags,userdata):
    print('event:{}'.format(event))
    print('location: x:{0} y:{1}'.format(str(x),str(y)))
    print('flags:{}'.format(flags))
    print('data:{}\n'.format(userdata))

if __name__ == '__main__':

    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video',800,600)

    cap = cv2.VideoCapture(0) #打开摄像头

    # 绑定鼠标事件，最后一个参数会传递给回调函数的 userdata 形参
    cv2.setMouseCallback('video',mouse_event_callback,'xiaoshanshan')


    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        cv2.imshow('video',frame)

        key = cv2.waitKey(1000 // 48)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1.0:
            break

    cap.release()
    cv2.destroyAllWindows()
```



# trackbar

```python
import cv2

# trackbar 回调函数
def trackbar_callback(value):
    print('trackbar_value:{}'.format(value))

if __name__ == '__main__':

    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video',800,600)

    cap = cv2.VideoCapture(0) #打开摄像头

    # trackbar,拖动条
    cv2.createTrackbar('R','video',0,255,trackbar_callback)
    cv2.createTrackbar('G', 'video', 0, 255, trackbar_callback)
    cv2.createTrackbar('B', 'video', 0, 255, trackbar_callback)

    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        cv2.imshow('video',frame)

        r = cv2.getTrackbarPos('R','video') # 获取 trackbar 的值
        
        key = cv2.waitKey(1000 // 48)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1.0:
            break

    cap.release()
    cv2.destroyAllWindows()
```



# 颜色空间

​		`RGB`：是通过对红、绿、蓝三个颜色通道的变化以及它们相互之间的叠加来得到各式各样的颜色的，`RGB`即是代表红、绿、蓝三个通道的颜色。

​		`BGR`：与`RGB`的差别在于三原色通道的顺序。

​		`HSV`：

![](image/HSV.webp)

​				`H`：色相，取值范围为`0 ~ 360`。红色为`0`，绿色为`120`，蓝色为`240`。

​				`S`：饱和度。取值为`0 ~ 100`，值越大，颜色越深艳。

​				`V`：明度。取值为`0 ~ 100`，值越大，颜色越明亮。

​		`HSL`：

​				![](image/HSL.webp)



```python
import cv2


def trackbar_callback(value):
    print('trackbar_value:{}'.format(value))

if __name__ == '__main__':

    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video',800,600)

    cap = cv2.VideoCapture(0) #打开摄像头

    # 颜色空间转换
    colorspaces = [cv2.COLOR_BGR2RGBA,cv2.COLOR_BGR2BGRA,cv2.COLOR_BGR2GRAY,cv2.COLOR_BGR2HSV,cv2.COLOR_BGR2YUV]
    cv2.createTrackbar('curcolor', 'video', 0, 4, trackbar_callback)

    while cap.isOpened():
        ret,frame = cap.read()


        index = cv2.getTrackbarPos('curcolor','video')
        frame = cv2.cvtColor(frame,colorspaces[index])

        if not ret:
            break
        cv2.imshow('video',frame)


        key = cv2.waitKey(1000 // 48)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1.0:
            break

    cap.release()
    cv2.destroyAllWindows()
```



## 合并与分离

```python
import cv2

if __name__ == '__main__':

    while True:
        frame = cv2.imread('E:\pic\savanna\\20210409165937.jpg')

        b,g,r = cv2.split(frame) # 颜色通道分离

        b[10:100, 10:100] = 255
        g[10:100, 10:100] = 255

        img = cv2.merge((b,g,r)) # 合并颜色通道

        cv2.imshow('video',frame)
        cv2.imshow('video1', img)

        key = cv2.waitKey(1000 // 48)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1.0:
            break

    cv2.destroyAllWindows()
```

# Mat

​		`Mat`是`OpenCV`在`C++`语言中用来表示图像数据的一种数据结构，在`python`中转化为`numpy`的`ndarray`。

​		`Mat`由`header`和`data`组成，`header`中记录了图片的维数大小，数据类型等数据。

```python
import cv2
import numpy as np

if __name__ == '__main__':

    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('video',1280,768)



    while True:
        frame = cv2.imread('E:\pic\savanna\\20210409165937.jpg')

		# 浅拷贝，header 不同，data 相同
        img = frame.view()
		# 深拷贝，header data 均不相同
        img2 = frame.copy()

        frame[10:100,10:100] = [0,0,255]

        cv2.imshow('video',np.hstack((frame,img,img2)))


        key = cv2.waitKey(1000 // 48)
        if key & 0xFF == ord('q') or cv2.getWindowProperty('video', cv2.WND_PROP_VISIBLE) < 1.0:
            break

    cv2.destroyAllWindows()
```

