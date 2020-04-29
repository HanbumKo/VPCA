import cv2
import numpy as np

from tkinter import *


class UI():
    def __init__(self, ut, camera, model_trt):
        self.camera = camera
        self.model_trt = model_trt
        self.ut = ut

        self.root = Tk()
        self.root.wm_title("VPCA")
        self.root.geometry("750x500")
        self._make_buttons()
        self._run()

    def _make_buttons(self):
        pose1 = Button(self.root, text="Deadlift", command=self.pose1)
        pose2 = Button(self.root, text="Dumbell Shoulder Press", command=self.pose2)
        pose3 = Button(self.root, text="Squat", command=self.pose3)
        pose4 = Button(self.root, text="Pose4", command=self.pose4)
        pose5 = Button(self.root, text="Pose5", command=self.pose5)
        pose6 = Button(self.root, text="Pose6", command=self.pose6)

        pose1.place(x=0, y=0, width=250, height=250)
        pose2.place(x=250, y=0, width=250, height=250)
        pose3.place(x=500, y=0, width=250, height=250)
        pose4.place(x=0, y=250, width=250, height=250)
        pose5.place(x=250, y=250, width=250, height=250)
        pose6.place(x=500, y=250, width=250, height=250)

    def _run(self):
        self.root.mainloop()

    def pose1(self):
        print("pose1 clicked")
        cap = cv2.VideoCapture('pose_videos/deadlift.avi')

        while True:
            user_image = self.camera.read()
            ret, example_image = cap.read()
            if not ret:
                break
            data = self.ut.preprocess(user_image)
            cmap, paf = self.model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            # counts, objects, peaks = ut.parseObjects(cmap, paf)
            counts, objects, peaks = self.ut.parseObjects(cmap, paf)
            self.ut.drawObjects(user_image, counts, objects, peaks)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.imshow("Image", concatenated_image)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cv2.destroyAllWindows()

    def pose2(self):
        print("pose2 clicked")
        cap = cv2.VideoCapture('pose_videos/press.avi')

        while True:
            user_image = self.camera.read()
            ret, example_image = cap.read()
            if not ret:
                break
            data = self.ut.preprocess(user_image)
            cmap, paf = self.model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            # counts, objects, peaks = ut.parseObjects(cmap, paf)
            counts, objects, peaks = self.ut.parseObjects(cmap, paf)
            self.ut.drawObjects(user_image, counts, objects, peaks)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.imshow("Image", concatenated_image)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cv2.destroyAllWindows()

    def pose3(self):
        print("pose3 clicked")
        cap = cv2.VideoCapture('pose_videos/squat.avi')

        while True:
            user_image = self.camera.read()
            ret, example_image = cap.read()
            if not ret:
                break
            data = self.ut.preprocess(user_image)
            cmap, paf = self.model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            # counts, objects, peaks = ut.parseObjects(cmap, paf)
            counts, objects, peaks = self.ut.parseObjects(cmap, paf)
            self.ut.drawObjects(user_image, counts, objects, peaks)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.imshow("Image", concatenated_image)
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cv2.destroyAllWindows()

    def pose4(self):
        print("pose4 clicked")

    def pose5(self):
        print("pose5 clicked")

    def pose6(self):
        print("pose6 clicked")
