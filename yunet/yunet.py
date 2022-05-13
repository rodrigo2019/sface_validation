# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import os

import cv2 as cv
import numpy as np


class YuNet:
    def __init__(self, *args, input_size=[640, 480], score_threshold=0.6, nms_threshold=0.3, top_k=5000, backend_id=0,
                 target_id=0, min_face_area=None, **kwargs):
        self._input_size = tuple(input_size)  # [w, h]
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._backend_id = backend_id
        self._target_id = target_id
        self._min_face_area = min_face_area

        local_path = os.path.dirname(os.path.realpath(__file__))
        weights_name = "face_detection_yunet_2022mar.onnx"
        weights_path = os.path.join(local_path, "bins", weights_name)
        self._modelPath = weights_path

        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._input_size,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackend(self, backend_id):
        self._backend_id = backend_id
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._input_size,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    def setTarget(self, target_id):
        self._target_id = target_id
        self._model = cv.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._input_size,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return faces[1]

    def detect(self, image):
        h, w, _ = image.shape
        factor = self._input_size[0] / w
        size_y = int(h * factor)
        image_infer = cv.resize(image, (self._input_size[0], size_y))
        self.setInputSize([self._input_size[0], size_y])
        faces = self.infer(image_infer)
        prop_w, prop_h = w / self._input_size[0], h / size_y
        if faces is None:
            return []
        faces_bbox = []
        for face in faces:
            face[0:4] = face[0:4] * np.array([prop_w, prop_h] * 2)
            face[4:14] = face[4:14] * np.array([prop_w, prop_h] * 5)
            w = face[2]
            h = face[3]
            area = w * h
            if self._min_face_area is not None and area < self._min_face_area:
                continue
            faces_bbox.append((image, face))
        return faces_bbox
