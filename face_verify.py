import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
from img2pose.api import image2pose


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one one_polylines on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    for i, (color, label) in enumerate(zip(color, label)):
        if i == 0:
            cv2.line(img, c1, (c1[0], c1[1]+50), color, thickness=tl, lineType=cv2.LINE_AA)
#         c1 = c0[0], c0[1]
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        c1 = c1[0], c1[1]+t_size[1]+tl*10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    img2pose = image2pose(0.9)
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
    else:
        learner.load_state(conf, 'mobilefacenet.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
    print(names)
    # inital camera
    # cap = cv2.VideoCapture('rtsp://admin:HuaWei123@192.168.0.120/LiveMedia/ch1/Media2')
    cap = cv2.VideoCapture('rtsp://admin:1qaz2wsx@192.168.0.64/1')
    # cap.set(3,1280)
    # cap.set(4,720)
    if args.save:
        video_writer = cv2.VideoWriter('face_recog.avi', cv2.VideoWriter_fourcc(*'XVID'), 15, (1280,720))
        # frame rate 6 due to my laptop is quite slow...
    frame_id = 0
    poses, bboxes, align_faces = [], [], []
    while cap.isOpened():
        isSuccess,frame = cap.read()
        if isSuccess:            
            # try:
#                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                # image = Image.fromarray(frame)
                # bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                # bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                # bboxes = bboxes.astype(int)
                # bboxes = bboxes + [-1,-1,1,1] # personal choice  
            frame_id += 1
            
            if frame_id%10 == 0:
                poses, bboxes, align_faces = img2pose(frame)
            if len(poses)>0:
                if frame_id%10 == 0:
                    align_faces = [Image.fromarray(face) for face in align_faces]
                    results, score = learner.infer(conf, align_faces, targets, args.tta)
                # print(score)
                for idx,bbox in enumerate(bboxes):
                    bbox = bbox.astype(int)
                    color = [(0, 200, 80)]
                    if args.score:
                        label = [names[results[idx] + 1] + '_{:.2f}'.format(score[idx])]
                        # frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    else:
                        label = [names[results[idx] + 1]]
                        # frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                    plot_one_box(bbox, frame, color=color, label=label, line_thickness=2)
            # except:
            #     print('detect error')    
                
            cv2.imshow('face Capture', frame)

        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    cap.release()
    if args.save:
        video_writer.release()
    cv2.destroyAllWindows()    