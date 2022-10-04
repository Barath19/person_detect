
import torch
import cv2
import rospy

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import random
import time
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from pathlib import Path
from utils.plots import plot_one_box

def detect(weight, img):
    rospy.loginfo("type of image is: {}".format(type(img)))
    rospy.loginfo("type of weight is: {}".format(type(weight)))

    #  source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    
    
    # save_img = not opt.nosave and not source.endswith('.txt')  # save inference iages
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        # ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    # set_logging()
    # device = select_device(opt.device)
    # half = True  # half precision only supported on CUDA
    half = False  # half precision only supported on CUDA

    # Load model
    device='cpu'
    source =img
    imgsz=640
    model = attempt_load(weight, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
         model.half()  # to FP16


    

      # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

     # Run inference
    if device != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    #pytorch image permute
    import numpy as np
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device!= 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, augment=False)[0]

    
    pred = model(img, augment=False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=[0], agnostic=False)

    # Process detections
    results = {}
    if len(pred[0]):
        for *xyxy, conf, cls in pred[0]:
            results['xmin'] = int(xyxy[0])
            results['ymin'] = int(xyxy[1])
            results['xmax'] = int(xyxy[2])
            results['ymax'] = int(xyxy[3])
            results['conf'] = float(conf)
            results['cls'] = int(cls)
            results['name'] = names[int(cls)]
    
    return results
    
        #cv2.imshow(str('zolo'), im0)
        #cv2.waitKey(0)  # 1 millisecond




if __name__=='__main__':
    weight = 'weights/yolov7.pt'
    # img = '/home/zainey/competition/comp_ws/runs/detect/exp4/frame000008.png'
    # img = '/home/zainey/competition/comp_ws/runs/detect/exp3/frame000004.png'
    img = '/home/barath/ros1/workspace/src/people_detection/scripts/images/frame0031.jpg'
    preds = detect(weight, img)

    print(preds) 