import torchvision.transforms as transforms
import cv2
import torch
import trt_pose
from PIL import Image


class Utils():
    def __init__(self, topology):
        self.topology = topology
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.cmap_threshold = 0.1
        self.link_threshold = 0.1
        self.cmap_window = 5
        self.line_integral_samples = 7
        self.max_num_parts = 100
        self.max_num_objects = 100

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms.functional.to_tensor(image).cuda()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def parseObjects(self, cmap, paf):
        peak_counts, peaks = trt_pose.plugins.find_peaks(cmap, self.cmap_threshold, self.cmap_window, self.max_num_parts)
        normalized_peaks = trt_pose.plugins.refine_peaks(peak_counts, peaks, cmap, self.cmap_window)
        score_graph = trt_pose.plugins.paf_score_graph(paf, self.topology, peak_counts, normalized_peaks, self.line_integral_samples)
        connections = trt_pose.plugins.assignment(score_graph, self.topology, peak_counts, self.link_threshold)
        object_counts, objects = trt_pose.plugins.connect_parts(connections, self.topology, peak_counts, self.max_num_objects)
        
        return object_counts, objects, normalized_peaks

    def drawObjects(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        # print("object_counts :", object_counts.shape)
        # print("objects :", objects.shape)
        # print("normalized_peaks :", normalized_peaks.shape)
        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        points = []
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            points = []
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)
                    points.append([x, y])

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)

        return points
        
    def printObjects(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        # print("object_counts :", object_counts.shape)
        # print("objects :", objects.shape)
        # print("normalized_peaks :", normalized_peaks.shape)
        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            points = []
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)
                    points.append(j)
            
            print(points)

            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), color, 2)

    def save_point(self, image, object_counts, objects, normalized_peaks, frame_idx):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]
        # print("object_counts :", object_counts.shape)
        # print("objects :", objects.shape)
        # print("normalized_peaks :", normalized_peaks.shape)
        
        K = topology.shape[0]
        count = int(object_counts[0])
        K = topology.shape[0]
        for i in range(count):
            color = (0, 255, 0)
            obj = objects[0][i]
            C = obj.shape[0]
            points = []
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 3, color, 2)
                    # points.append(j)
                    points.append([x, y])
            
            if len(points) == 18:
                with open("pose_points/test.txt", "a") as f:
                    f.write(str(frame_idx))
                    for x, y in points:
                        f.write(" " + str(x) + " " + str(y))
                    f.write("\n")
                # print(frame_idx, points)
