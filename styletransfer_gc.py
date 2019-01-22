

import torch
import torchvision.transforms as transforms 
import torchvision.models as models 
import torch.optim as optim 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
import os 

try:
   os.mkdir('images')
except:
   pass

#Function to load image and transform them into tensors
def loadImage(img_path, max_size=400, shape=None):
	    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image

#Function for converting tensor image to numpy image 
def imgConvert(tensor):
    
    #transfering tensor to cpu and copying it 
    image = tensor.cpu().clone().detach()
    #converting into numpy
    image =image.numpy()
    #removing batch dimention
    image = image.squeeze()
    #reverting the normalization done
    image=image.transpose(1,2,0)
    image=image*np.array((.229,.224,.225))+np.array((.485,.456,.406))
    return image.clip(0,1)


def getFeatures(image,model,layers=None):

    '''
    Creating a mapping that contains 
    layers that are used for style representation 
    and content representation

    Layers for style representation 
    conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    Layers for content representation 
    conv4_2 

    print the model
    0 maps to conv1_1, 5 to conv2_1 .. so on
    '''
    if layers is None:
        layers = {
            '0':'conv1_1',
            '5':'conv2_1',
            '10':'conv3_1',
            '19':'conv4_1',
            '21':'conv4_2',  
            '28':'conv5_1',
        }
    
    features = {}
    x = image 
    for name,layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
        
    return features

def gramMatrix(tensor):

    _,d,w,h = tensor.shape
    tensor = tensor.view(d,h*w)
    gram= torch.mm(tensor,tensor.t())
    return gram
#Loading content and style images
content=loadImage('images/23.png').cuda()
style = loadImage('images/2.jpg', shape=content.shape[-2:]).cuda()
print type(content)

# display the images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
ax1.imshow(imgConvert(content))
ax2.imshow(imgConvert(style))

model = models.vgg19(pretrained=True).features
model.cuda()
#freezing weights
for param in model.parameters():
    param.requires_grad_(False)

#Get content and style feature
content_features = getFeatures(content,model)
style_features = getFeatures(style,model)

#Calculating the gram matrix for each layer of style representation
style_grams= {layer: gramMatrix(style_features[layer]) for layer in style_features}

#Creating a target Image 
targetImage = content.clone().requires_grad_(True).cuda()

style_weights ={
    'conv1_1':1,
    'conv2_1':.5,
    'conv3_1':.5,
    'conv4_1':.3,
    'conv5_1':.1,
}

content_weight = 1
style_weight = 1e7 #increase this value if you're now observing enough style in output image

show_every = 400 

optimizer = optim.Adam([targetImage],lr=.003)
steps = 1600 # more steps, more style transfer

for ii in range(1,steps+1):

    target_features = getFeatures(targetImage,model)
    content_loss = torch.mean((target_features['conv4_2']-content_features['conv4_2'])**2)

    #calculating style loss
    style_loss=0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gramMatrix(target_feature)
        _,d,w,h = target_feature.shape

        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram-style_gram)**2)
        style_loss +=layer_style_loss/(d*h*w)

    total_loss = content_loss*content_weight + style_weight*style_loss
     
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if ii % show_every ==0:
        print('Total lose :',total_loss.item())
        plt.imshow(imgConvert(targetImage))
        plt.savefig('images/step_'+str(ii)+'.png')
        plt.show()



