# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os.path
import sys

sys.path.append(os.path.abspath("./ContrastiveUnpairedTranslation"))
sys.path.append(os.path.abspath("./SuperPoint"))

print(sys.path)
from SuperPoint.SuperpointFunctions import *
from SuperPoint.SuperpointFunctions import *
from ContrastiveUnpairedTranslation import cut_funtions, util
import time
import argparse

if __name__ == '__main__':
    theparser = argparse.ArgumentParser(description='runs unsupervised registration pipeline')
    theparser.add_argument('cut_weights_name', type=str)
    theparser.add_argument('sp_weights_name', type=str)
    theparser.add_argument('img1_path', type=str, help='this will be style transferred (default: 480)')
    theparser.add_argument('img2_path', type=str)
    theparser.add_argument('--H', type=int, default=256,help='The height in pixels to resize the images to (default: 480)')
    theparser.add_argument('--W', type=int, default=256, help='The width in pixels to resize the images to (default: 640)')
    theparser.add_argument('--k_best', type=int, default=1000, help='Maximum number of keypoints to keep (default: 1000)')

    args_glob, _ = theparser.parse_known_args()

    print(args_glob)
    img1_file = args_glob.img1_path
    img2_file = args_glob.img2_path
    cut_weights_name = args_glob.cut_weights_name
    sp_weights_name = args_glob.sp_weights_name
    # cut_model = "D:/dev/python/contrastive-unpaired-translation/checkpoints/tissue_test_train_all"
    # sp_model_path = "d:/dev/python/SuperPoint/exper/saved_models/sp_v6/"
    # input_image = "D:/datasets/Image_registration/230123NH-77995rqivU/trainA/p1_wA1_t1_m1010_c0_z1_l1_o0.png"
    image_1 = args_glob.img1_path
    image_2 = args_glob.img2_path

    m = cut_funtions.load_model(cut_weights_name)
    sp_m = load_tensorflow_model(sp_weights_name)
    t = cut_funtions.convert_data_to_tensor(image_2)
    r = cut_funtions.inference(m, t)
    res_img = util.util.tensor2im(r)
    img_size = (256, 256)
    start = time.time()
    i1 = cv2.imread(image_1, cv2.IMREAD_COLOR)
    img1, img1_orig = preprocess_image(i1, img_size)
    img2, img2_orig = preprocess_image(res_img, img_size)
    res = compute_superpoint(img1, img2)
    end = time.time()
    print(end - start)
    print(res)
