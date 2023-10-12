import torchvision
from PIL import Image
import torch

# CIFAR10的类别
classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

img_path="./imgs/dog.jpg"
img=Image.open(img_path)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                           torchvision.transforms.ToTensor()])
img=transform(img)
model=torch.load("pth/mynet_0.pth")
img=torch.reshape(img,(1,3,32,32))
model.to("cpu")
output=model(img)
print(classes[output.argmax(1)])