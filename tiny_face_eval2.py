# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf
import tiny_face_model
import util
from argparse import ArgumentParser
import cv2
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

import pylab as pl
import time
import os
import sys
from scipy.special import expit
import glob

MAX_INPUT_DIM = 5000.0


def overlay_bounding_boxes(raw_img, refined_bboxes, lw):
    # Overlay bounding boxes on an image with the color based on the confidence.
    for r in refined_bboxes:
        _score = expit(r[4])
        cm_idx = int(np.ceil(_score * 255))
        rect_color = [int(np.ceil(x * 255)) for x in util.cm_data[cm_idx]]  # parula
        _lw = lw
        if lw == 0:  # line width of each bounding box is adaptively determined.
            bw, bh = r[2] - r[0] + 1, r[3] - r[0] + 1
            _lw = 1 if min(bw, bh) <= 20 else max(2, min(3, min(bh / 20, bw / 20)))
            _lw = int(np.ceil(_lw * _score))

        _r = [int(x) for x in r[:4]]
        cv2.rectangle(raw_img, (_r[0] - 5, _r[1] - 5), (_r[2] + 5, _r[3] + 5), rect_color, _lw)
        pass

    pass


def evaluate(weight_file_path, data_dir, output_dir, prob_thresh=0.5, nms_thresh=0.1, lw=3, display=False):

    # placeholder of input images. Currently batch size of one is supported.
    x = tf.placeholder(tf.float32, [1, None, None, 3])  # n, h, w, c

    # Create the tiny face model which weights are loaded from a pretrained model.
    model = tiny_face_model.Model(weight_file_path)
    score_final = model.tiny_face(x)

    # Find image files in data_dir.
    filenames = []
    for ext in ('*.png', '*.gif', '*.jpg', '*.jpeg'):
        filenames.extend(glob.glob(os.path.join(data_dir, ext)))

    # Load an average image and clusters(reference boxes of templates).
    with open(weight_file_path, "rb") as f:
        _, mat_params_dict = pickle.load(f)

    average_image = model.get_data_by_key("average_image")
    clusters = model.get_data_by_key("clusters")
    clusters_h = clusters[:, 3] - clusters[:, 1] + 1
    clusters_w = clusters[:, 2] - clusters[:, 0] + 1
    normal_idx = np.where(clusters[:, 4] == 1)

    # main
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        _results = []

        for filename in filenames:
            fname = filename.split(os.sep)[-1]
            raw_img = cv2.imread(filename)
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            raw_img_f = raw_img.astype(np.float32)

            def _calc_scales():
                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]
                min_scale = min(np.floor(np.log2(np.max(clusters_w[normal_idx] / raw_w))),
                                np.floor(np.log2(np.max(clusters_h[normal_idx] / raw_h))))
                max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
                scales_down = pl.frange(min_scale, 0, 1.)
                scales_up = pl.frange(0.5, max_scale, 0.5)
                scales_pow = np.hstack((scales_down, scales_up))
                scales = np.power(2.0, scales_pow)
                return scales

            scales = _calc_scales()
            start = time.time()

            # initialize output
            bboxes = np.empty(shape=(0, 5))

            # process input at different scales
            for s in scales:
                print("Processing {} at scale {:.4f}".format(fname, s))
                img = cv2.resize(raw_img_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                img = img - average_image
                img = img[np.newaxis, :]

                # we don't run every template on every scale ids of templates to ignore
                tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
                ignoredTids = list(set(range(0, clusters.shape[0])) - set(tids))

                # run through the net
                score_final_tf = sess.run(score_final, feed_dict={x: img})

                # collect scores
                score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
                prob_cls_tf = expit(score_cls_tf)
                prob_cls_tf[0, :, :, ignoredTids] = 0.0

                def _calc_bounding_boxes():
                    # threshold for detection
                    _, fy, fx, fc = np.where(prob_cls_tf > prob_thresh)

                    # interpret heatmap into bounding boxes
                    cy = fy * 8 - 1
                    cx = fx * 8 - 1
                    ch = clusters[fc, 3] - clusters[fc, 1] + 1
                    cw = clusters[fc, 2] - clusters[fc, 0] + 1

                    # extract bounding box refinement
                    Nt = clusters.shape[0]
                    tx = score_reg_tf[0, :, :, 0:Nt]
                    ty = score_reg_tf[0, :, :, Nt:2 * Nt]
                    tw = score_reg_tf[0, :, :, 2 * Nt:3 * Nt]
                    th = score_reg_tf[0, :, :, 3 * Nt:4 * Nt]

                    # refine bounding boxes
                    dcx = cw * tx[fy, fx, fc]
                    dcy = ch * ty[fy, fx, fc]
                    rcx = cx + dcx
                    rcy = cy + dcy
                    rcw = cw * np.exp(tw[fy, fx, fc])
                    rch = ch * np.exp(th[fy, fx, fc])

                    scores = score_cls_tf[0, fy, fx, fc]
                    tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
                    tmp_bboxes = np.vstack((tmp_bboxes / s, scores))
                    tmp_bboxes = tmp_bboxes.transpose()
                    return tmp_bboxes

                tmp_bboxes = _calc_bounding_boxes()
                bboxes = np.vstack((bboxes, tmp_bboxes))  # <class 'tuple'>: (5265, 5)

            print("time {:.2f} secs for {}".format(time.time() - start, fname))

            # non maximum suppression
            # refind_idx = util.nms(bboxes, nms_thresh)
            refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                      tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                      max_output_size=bboxes.shape[0], iou_threshold=nms_thresh)
            refind_idx = sess.run(refind_idx)
            refined_bboxes = bboxes[refind_idx]

            _result = []
            for r in refined_bboxes:
                _score = expit(r[4])
                _r = [int(x) for x in r[:4]]
                print("{} {} {} {} {}".format(_score, _r[0], _r[1], _r[2], _r[3]))
                _result.append([_r[0], _r[1], _r[2], _r[3], _score])
                pass

            overlay_bounding_boxes(raw_img, refined_bboxes, lw)

            if display:
                # plt.axis('off')
                plt.imshow(raw_img)
                plt.show()

            # save image with bounding boxes
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, fname), raw_img)
            _results.append(_result)

            pass

        return _results

    pass


def main():
    argparse = ArgumentParser()
    argparse.add_argument('--weight_file_path', type=str, help='Pretrained weight file.', default="./models/mat2tf.pkl")
    argparse.add_argument('--data_dir', type=str, help='Image data directory.',
                          default="./demo/input")
    argparse.add_argument('--output_dir', type=str, help='Output directory for images with faces detected.',
                          default="./demo/output")
    argparse.add_argument('--prob_thresh', type=float, help='The threshold of detection', default=0.5)
    argparse.add_argument('--nms_thresh', type=float, help='The overlap threshold of nms', default=0.2)
    argparse.add_argument('--line_width', type=int, help='Line width of bounding boxes(0: auto).', default=2)
    argparse.add_argument('--display', type=bool, help='Display each image on window.', default=False)

    args = argparse.parse_args()

    # check arguments
    assert os.path.exists(args.weight_file_path), "weight file: " + args.weight_file_path + " not found."
    assert os.path.exists(args.data_dir), "data directory: " + args.data_dir + " not found."
    assert os.path.exists(args.output_dir), "output directory: " + args.output_dir + " not found."
    assert args.line_width >= 0, "line_width should be >= 0."

    videos_path = "/home/ubuntu/data1.5TB/video/video_deal/show2"
    results_path = "/home/ubuntu/data1.5TB/video/video_deal/face"
    videos_images_path = os.listdir(videos_path)
    for video_images_one in videos_images_path:

        video_images_path = os.path.join(videos_path, video_images_one)
        result_images_path = os.path.join(results_path, video_images_one)
        if not os.path.exists(result_images_path):
            os.makedirs(result_images_path)

        tf.reset_default_graph()
        with tf.Graph().as_default():
            results = evaluate(weight_file_path=args.weight_file_path, data_dir=video_images_path,
                               output_dir=result_images_path, prob_thresh=args.prob_thresh,
                               nms_thresh=args.nms_thresh, lw=args.line_width, display=args.display)

            with open("{}.pkl".format(result_images_path), "wb") as f:
                pickle.dump(results, f)
                pass

            pass
    pass


if __name__ == '__main__':
    main()
