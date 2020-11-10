import cv2
import numpy as np
from tkinter import font as tkFont

from calc import *
from tkinter import *


class UI():
    def __init__(self, ut, camera, model_trt):
        self.camera = camera
        self.model_trt = model_trt
        self.ut = ut

        self.pose1_set_num = 0
        self.pose2_set_num = 0
        self.pose3_set_num = 0

        self.root = Tk()
        self.root.wm_title("VPCA")
        width = 1000
        height = 333
        scrwdth = self.root.winfo_screenwidth()
        scrhgt = self.root.winfo_screenheight()
        xLeft = int((scrwdth/2) - (width/2))
        yTop = int((scrhgt/2) - (height/2))
        self.root.geometry(str(width) + "x" + str(height) + "+" + str(xLeft) + "+" + str(yTop))
        self._make_buttons()
        self._run()

    def _make_buttons(self):
        helv36 = tkFont.Font(family='Helvetica', size=36, weight=tkFont.BOLD)
        pose1 = Button(self.root, text="Deadlift", command=self.pose1, font=helv36)
        pose2 = Button(self.root, text="Shoulder\nPress", command=self.pose2, font=helv36)
        pose3 = Button(self.root, text="Squat", command=self.pose3, font=helv36)
        # pose4 = Button(self.root, text="Pose4", command=self.pose4)
        # pose5 = Button(self.root, text="Pose5", command=self.pose5)
        # pose6 = Button(self.root, text="Pose6", command=self.pose6)

        pose1.place(x=0, y=0, width=333, height=333)
        pose2.place(x=333, y=0, width=333, height=333)
        pose3.place(x=666, y=0, width=333, height=333)
        # pose4.place(x=0, y=250, width=250, height=250)
        # pose5.place(x=250, y=250, width=250, height=250)
        # pose6.place(x=500, y=250, width=250, height=250)

    def _run(self):
        self.root.mainloop()

    def pose1(self):
        cap = cv2.VideoCapture('pose_videos/deadlift.avi')
        good_points_dict = self._load_good_points("pose_points/deadlift.txt")
        frame_idx = 0
        total_l2_distance = 0
        self.pose1_set_num += 1

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
            user_points = self.ut.drawObjects(user_image, counts, objects, peaks)
            user_image = cv2.resize(user_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            example_image = cv2.resize(example_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.namedWindow("Deadlift")        # Create a named window
            cv2.moveWindow("Deadlift", 40,30)  # Move it to (40,30)
            cv2.imshow("Deadlift", concatenated_image)

            if frame_idx in good_points_dict:
                total_l2_distance += l2_dist(user_points, good_points_dict[frame_idx])
            
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        print()
        print()
        print("="*60)
        print("Correctness :", total_l2_distance)
        print("You've done %d set of Deadlift!!!" % self.pose1_set_num)
        print("You've done %d set of Shoulder Press!!!" % self.pose2_set_num)
        print("You've done %d set of Squat!!!" % self.pose3_set_num)
        print("="*60)
        print()
        print()
        cv2.destroyAllWindows()

        # Feedback

    def pose2(self):
        cap = cv2.VideoCapture('pose_videos/press.avi')
        good_points_dict = self._load_good_points("pose_points/press.txt")
        frame_idx = 0
        total_l2_distance = 0
        self.pose2_set_num += 1

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
            user_points = self.ut.drawObjects(user_image, counts, objects, peaks)
            user_image = cv2.resize(user_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            example_image = cv2.resize(example_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.namedWindow("Shoulder Press")        # Create a named window
            cv2.moveWindow("Shoulder Press", 40,30)  # Move it to (40,30)
            cv2.imshow("Shoulder Press", concatenated_image)
            
            if frame_idx in good_points_dict:
                total_l2_distance += l2_dist(user_points, good_points_dict[frame_idx])
            
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        print()
        print()
        print("="*60)
        print("Correctness :", total_l2_distance)
        print("You've done %d set of Deadlift!!!" % self.pose1_set_num)
        print("You've done %d set of Shoulder Press!!!" % self.pose2_set_num)
        print("You've done %d set of Squat!!!" % self.pose3_set_num)
        print("="*60)
        print()
        print()
        cv2.destroyAllWindows()

    def pose3(self):
        cap = cv2.VideoCapture('pose_videos/squat.avi')
        good_points_dict = self._load_good_points("pose_points/squat.txt")
        frame_idx = 0
        total_l2_distance = 0
        self.pose3_set_num += 1

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
            user_points = self.ut.drawObjects(user_image, counts, objects, peaks)
            user_image = cv2.resize(user_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            example_image = cv2.resize(example_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.namedWindow("Squat")        # Create a named window
            cv2.moveWindow("Squat", 40,30)  # Move it to (40,30)
            cv2.imshow("Squat", concatenated_image)

            if frame_idx in good_points_dict:
                total_l2_distance += l2_dist(user_points, good_points_dict[frame_idx])
            
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        print()
        print()
        print("="*60)
        print("Correctness :", total_l2_distance)
        print("You've done %d set of Deadlift!!!" % self.pose1_set_num)
        print("You've done %d set of Shoulder Press!!!" % self.pose2_set_num)
        print("You've done %d set of Squat!!!" % self.pose3_set_num)
        print("="*60)
        print()
        print()
        cv2.destroyAllWindows()

    def pose4(self):
        print("pose4 clicked")
        cap = cv2.VideoCapture('pose_videos/stretching_1.avi')

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
            user_points = self.ut.drawObjects(user_image, counts, objects, peaks)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.imshow("Image", concatenated_image)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cv2.destroyAllWindows()

    def pose5(self):
        print("pose5 clicked")
        cap = cv2.VideoCapture('pose_videos/stretching_2.avi')

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
            user_points = self.ut.drawObjects(user_image, counts, objects, peaks)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.imshow("Image", concatenated_image)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cv2.destroyAllWindows()

    def pose6(self):
        print("pose6 clicked")
        cap = cv2.VideoCapture('pose_videos/stretching_3.avi')

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
            user_points = self.ut.drawObjects(user_image, counts, objects, peaks)
            concatenated_image = np.concatenate((user_image, example_image), axis=1)
            cv2.imshow("Image", concatenated_image)

            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

        cv2.destroyAllWindows()
    
    def _parse_line(self, points_str):
        splitted = points_str.split()
        if len(splitted) == 0:
            return -1, []
        frame_idx = int(splitted[0])
        points = [
            [splitted[1], splitted[2]],
            [splitted[3], splitted[4]],
            [splitted[5], splitted[6]],
            [splitted[7], splitted[8]],
            [splitted[9], splitted[10]],
            [splitted[11], splitted[12]],
            [splitted[13], splitted[14]],
            [splitted[15], splitted[16]],
            [splitted[17], splitted[18]],
            [splitted[19], splitted[20]],
            [splitted[21], splitted[22]],
            [splitted[23], splitted[24]],
            [splitted[25], splitted[26]],
            [splitted[27], splitted[28]],
            [splitted[29], splitted[30]],
            [splitted[31], splitted[32]],
            [splitted[33], splitted[34]],
            [splitted[35], splitted[36]],
        ]

        return frame_idx, points
    
    def _load_good_points(self, points_path):
        with open(points_path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        points_dict = dict()
        for points_str in content:
            frame_idx, points = self._parse_line(points_str)
            points_dict[frame_idx] = points
        return points_dict

