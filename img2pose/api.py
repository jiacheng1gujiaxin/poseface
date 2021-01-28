from .img2pose import img2poseModel
from .model_loader import load_model
from .utils.pose_operations import align_faces

import cv2
import numpy as np
import torch

DEPTH = 18
MAX_SIZE = 1400
MIN_SIZE = 600

class image2pose():
    def __init__(self, threshold):
        threed_points = np.load('/home/ai/Desktop/project/face/img2pose/pose_references/reference_3d_68_points_trans.npy')
        five_points = np.load("/home/ai/Desktop/project/face/img2pose/pose_references/reference_3d_5_points_trans.npy")
        POSE_MEAN = "/home/ai/Desktop/project/face/img2pose/models/WIDER_train_pose_mean_v1.npy"
        POSE_STDDEV = "/home/ai/Desktop/project/face/img2pose/models/WIDER_train_pose_stddev_v1.npy"
        MODEL_PATH = "/home/ai/Desktop/project/face/img2pose/models/img2pose_v1.pth"

        pose_mean = np.load(POSE_MEAN)
        pose_stddev = np.load(POSE_STDDEV)

        img2pose_model = img2poseModel(
            DEPTH, MIN_SIZE, MAX_SIZE, 
            pose_mean=pose_mean, pose_stddev=pose_stddev,
            threed_68_points=threed_points,
        )
        load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True) 
        
        self.five_points = five_points
        self.threshold = threshold
        self.img2pose_model = img2pose_model
        self.img2pose_model.evaluate()

    def __call__(self, img):
        '''
        image: numpy (h, w, c)
        '''
        img_ori = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img/255.).float()

        (_, h, w) = img.shape
        image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])
            
        res = self.img2pose_model.predict([img])[0]

        all_bboxes = res["boxes"].cpu().numpy().astype('float')

        poses = []
        bboxes = []
        for i in range(len(all_bboxes)):
            if res["scores"][i] > self.threshold:
                bbox = all_bboxes[i]
                pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
                pose_pred = pose_pred.squeeze()

                poses.append(pose_pred)  
                bboxes.append(bbox)
    #     im = np.asarray(image)
    #     print(im.shape)
        # print(poses)
    #     print(threed_points)
        # print(img_ori.shape)
        aligned_faces = align_faces(self.five_points, img_ori, poses, face_size=112)
        return poses, bboxes, aligned_faces
