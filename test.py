
import numpy as np
import torch
from model.blocks import Encoder, Decoder


x = np.random.rand(1,18,32,1024) # patch image1 feature
y = np.random.rand(1,18,32,1024)  # patch image2 features
x_torch = torch.Tensor(x)
y_torch = torch.Tensor(y)
x_torch = x_torch.reshape(1,18*32,1024)
y_torch = y_torch.reshape(1,18*32,1024)

#patch position
xpos=torch.Tensor([(i,j) for i in range(0,x.shape[1]) for j in range(0,x.shape[2])])
ypos =torch.Tensor([(i,j) for i in range(0,y.shape[1]) for j in range(0,y.shape[2])])



batch_feature = torch.cat((x_torch, y_torch), dim=0)

encoder = Encoder(x_torch.shape[-1], 16)
decoder = Decoder(x_torch.shape[-1], 16)

result1 = encoder(x_torch,xpos)
result2,result3 = decoder(x_torch,y_torch, xpos,ypos)

print('result1',result1)
print('result2',result2)
print('result3',result3)
