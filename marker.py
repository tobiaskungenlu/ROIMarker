import cv2
import numpy as np
import os

class ROIMarker():

    def __init__(self, data_path, resize_factor):
        self.data_path = data_path
        self.resize_factor = resize_factor

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (0,0), fx=self.resize_factor[0], fy=self.resize_factor[1])
        return img

    def label_data_polygon(self):
        total = len(os.listdir(self.data_path))
        i = 0
        for file in os.listdir(self.data_path):
            print("Image {} of {}".format(str(i), total))
            i+=1
           
            if not file.startswith("top") or file.endswith("yuv"):  # If you dont need bottom images/YUV images, comment these
                continue
            if file.startswith("label"):
                continue
            seq = file.split('.')[0]
            print("Current num: {}".format(seq))
            label_name = "label_" + seq + ".bmp"
            if label_name in os.listdir(self.data_path):
                print("Label file already exists.")
                continue
            _dir = os.path.join(self.data_path, file)
            img = self.load_img(_dir)

            new_image = np.zeros(img.shape, img.dtype)
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    for c in range(img.shape[2]):
                        new_image[y,x,c] = np.clip(img[y,x,c], 0, 255)
            new_file_name = "label_" + seq + ".bmp"
            new_path = os.path.join(self.data_path, new_file_name)
            pts = []
            param = [new_image, new_path, pts]

            cv2.namedWindow('ROI')
            cv2.setMouseCallback('ROI', self.draw_roi, param)

            while True:

                cv2.imshow('ROI', new_image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):    
                    print("[INFO] ROI saved.")
                    break
                if key == ord("q"):
                    break
            cv2.destroyAllWindows()

    def draw_roi(self, event, x, y, flags, param): # param: [img, path, points]

        img, path, pts = param
        img2 = img.copy()
        
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))  
    
        if event == cv2.EVENT_RBUTTONDOWN:
            pts.pop()  
    
        if event == cv2.EVENT_MBUTTONDOWN:
            
            pts.append((img.shape[1], img.shape[0]))
            pts.append((0, img.shape[0]))
            mask = np.zeros(img.shape, np.uint8)
            points = np.array(pts, np.int32)
            points = points.reshape((-1, 1, 2))

            mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
            mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
            mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))
            show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
    
            cv2.imshow("show_img", show_image)

            mask4 = cv2.cvtColor(mask2, cv2.COLOR_BGR2GRAY)
            cv2.imshow("mask", mask4)
    
            ROI = cv2.bitwise_and(mask2, img)
            cv2.imshow("ROI", ROI)
            cv2.waitKey(0)
            cv2.imwrite(path, mask4)
            print("Saved")
            return
    
        if len(pts) > 0:
            cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)
    
        if len(pts) > 1:
            for i in range(len(pts) - 1):
                cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)
                cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)
    
        cv2.imshow('image', img2)


if __name__ == "__main__":
    marker = ROIMarker(".\data\\1\\", (0.2, 0.2))
    marker.label_data_polygon()