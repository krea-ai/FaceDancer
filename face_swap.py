RETINA_CKPT = "RetinaFace-Res50.h5"
ARCFACE_CKPT = "ArcFace-Res50.h5"
FACEDANCER_CKPT = "FaceDancer_config_c_HQ.h5"
# -*- coding: utf-8 -*-
# @Author: netrunner-exe
# @Date:   2022-12-21 12:52:01
# @Last Modified by:   netrunner-exe
# @Last Modified time: 2022-12-21 19:14:34
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import InstanceNormalization

from networks.layers import AdaIN, AdaptiveAttention
from retinaface.models import *
from utils.options import FaceDancerOptions
from utils.swap_func import run_inference

import glob
import os
import shutil
import sys

import cv2
import numpy as np
import proglog
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import subprocess

from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)



from PIL import Image
import requests
import cv2
import numpy as np

MODEL_FOLDER = "/home/erwann/generation-service/safetensor-models"

def face_swap(source, target, RetinaFace,
                  ArcFace, FaceDancer):
    try:

        target = np.array(target)
        source = np.array(source)

        source_h, source_w, _ = source.shape
        source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
        source_lm = get_lm(source_a, source_w, source_h)
        source_aligned = norm_crop(source, source_lm, image_size=112, shrink_factor=1.0)

        source_z = ArcFace.predict(np.expand_dims(source_aligned / 255.0, axis=0))

        blend_mask_base = np.zeros(shape=(256, 256, 1))
        blend_mask_base[77:240, 32:224] = 1
        blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

        im = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)

        detection_scale = (im_w // 640) if (im_w > 640) else 1
        faces = RetinaFace(np.expand_dims(cv2.resize(im,
                                                     (im_w // detection_scale,
                                                      im_h // detection_scale)), axis=0)).numpy()
        total_img = im / 255.0

        for annotation in faces:
            lm_align = get_lm(annotation, im_w, im_h)

            # align the detected face
            M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)

            # face swap
            face_swap = FaceDancer.predict([np.expand_dims((im_aligned - 127.5) / 127.5, axis=0), source_z])
            face_swap = (face_swap[0] + 1) / 2

            # get inverse transformation landmarks
            transformed_lmk = transform_landmark_points(M, lm_align)

            # warp image back
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(face_swap, iM, im_shape, borderValue=0.0)

            # blend swapped face with target image
            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)

            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))

        total_img = np.clip(total_img * 255, 0, 255).astype('uint8')

        # cv2.imwrite(result_img_path, cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB))

        return cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB)
        return total_img

    except Exception as e:
        print('\n', e)
        sys.exit(0)


def facedancer(source_image_url, target_image_url):
    source_image = Image.open(requests.get(source_image_url, stream=True).raw)
    target_image = Image.open(requests.get(target_image_url, stream=True).raw)

    source_image = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    target_image = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)

    retina_path = os.path.join(MODEL_FOLDER, "retinaface", RETINA_CKPT)
    arcface_path = os.path.join(MODEL_FOLDER, "arcface_model", ARCFACE_CKPT)
    facedancer_path = os.path.join(MODEL_FOLDER, "facedancer", FACEDANCER_CKPT)

    print("PATHS***********************8")
    print(retina_path, arcface_path, facedancer_path)

    if len(tf.config.list_physical_devices('GPU')) != 0:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[0], 'GPU')

    print('\nInitializing FaceDancer...')
    RetinaFace = load_model(retina_path, compile=False,
                            custom_objects={"FPN": FPN,
                                            "SSH": SSH,
                                            "BboxHead": BboxHead,
                                            "LandmarkHead": LandmarkHead,
                                            "ClassHead": ClassHead})
    print("loaded retinaface")
    ArcFace = load_model(arcface_path, compile=False)
    print("loaded arcface")

    G = load_model(facedancer_path, compile=False,
                   custom_objects={"AdaIN": AdaIN,
                                   "AdaptiveAttention": AdaptiveAttention,
                                   "InstanceNormalization": InstanceNormalization})
    print("loaded models")
    # G.summary()
    image = face_swap(source_image, target_image, RetinaFace, ArcFace, G)
    cv2.imwrite('result.jpg', image)

if __name__ == "__main__":
    url1 = "https://canvas-generations-v1.s3.us-west-2.amazonaws.com/e13de0cb-c208-487c-9fac-bd90584fc9aa.png"
    url2= "https://canvas-generations-v1.s3.us-west-2.amazonaws.com/bd32cd8f-bf88-49a8-90c2-9b036318d1c5.png"
    facedancer(url1, url2)
