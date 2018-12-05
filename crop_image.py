import cv2
import os
import pickle
import numpy as np


def get_position(points, image_file_name="512.jpg", result_file_name="./512_1.jpg", display=False):
    img = cv2.imread(image_file_name)

    img2 = np.array(img)
    h, w, c = img2.shape

    margin = 5
    p_w, p_h = 50, 50
    for point_index, point in enumerate(points):

        point[0] = 0 if point[0] < 0 else point[0]
        point[1] = 0 if point[1] < 0 else point[1]
        point[2] = w if point[2] > w else point[2]
        point[3] = h if point[3] > h else point[3]

        person = img[point[1] - margin: point[3] + margin, point[0] - margin: point[2] + margin, :]
        person = cv2.resize(person, dsize=(p_w, p_h))

        img2[h - p_h - margin: -margin, point_index * (p_w + margin) + margin: (point_index + 1) * (p_w + margin), :] = person
        cv2.putText(img2, "{:.2f}".format(point[4]), (point_index * (p_w + margin) + 6, h - p_h - margin - 5),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255))
        pass

    # 加框加字
    for point in points:
        _margin = 3
        cv2.rectangle(img2, (point[0] - _margin, point[1] - _margin), (point[2] + _margin, point[3] + _margin), (0, 255, 0), 2)
        # cv2.putText(img2, "{:.2f}".format(point[4]), (point[0], point[1] - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
        #             (0, 0, 255))
        pass

    cv2.imwrite(result_file_name, img2)

    if display:
        cv2.imshow("mmm", img2)
        cv2.waitKey(0)

    pass


if __name__ == '__main__':

    # 只需要`原图` + `pkl`即可
    videos_images_path_ = "/home/ubuntu/data1.5TB/video/video_deal/show2/20181009_20181009090613_20181009090627_090339"
    result_images_path_ = "/home/ubuntu/data1.5TB/video/video_deal/face/20181009_20181009090613_20181009090627_090339"
    pkl_path_ = "/home/ubuntu/data1.5TB/video/video_deal/face/20181009_20181009090613_20181009090627_090339.pkl"

    if not os.path.exists(result_images_path_):
        os.makedirs(result_images_path_)
        pass

    video_images_path_filename = os.listdir(videos_images_path_)

    with open(pkl_path_, 'rb') as f:
        objs = pickle.load(f)

    for index, video_image_filename in enumerate(video_images_path_filename):
        image_filename = os.path.join(videos_images_path_, video_image_filename)
        result_file_name = os.path.join(result_images_path_, video_image_filename)
        get_position(objs[index], image_file_name=image_filename, result_file_name=result_file_name)
        pass

    pass
