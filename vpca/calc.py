import math


def l2_dist(user_points, good_points):
    total = 0
    if len(user_points) == len(good_points):
        for i in range(len(good_points)):
            good_x = int(good_points[i][0])
            good_y = int(good_points[i][1])
            user_x = int(user_points[i][0])
            user_y = int(user_points[i][1])
            dist = ((good_x - user_x)**2 + (good_y - user_y)**2)**0.5
            total += dist

    return total
