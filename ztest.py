

from model import MobileFaceNet
import torch
from mtcnn import MTCNN


model = MobileFaceNet(512)

model.load_state_dict(torch.load('/home/ai/Desktop/project/pytorch-insightface/model_mobilefacenet.pth'))

print(model.state_dict())

