import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from PIL import Image
import numpy as np
from train import create_model
import json

def main():
    argparser = argparse.ArgumentParser(
        description='Test the Classifier')
    argparser.add_argument(
        'data_directory',
        help='The directory of the image to classify.')
    argparser.add_argument(
        'checkpoint',
        help='The directory for loading checkpoints.')
    argparser.add_argument(
        '--top_k',
        type=int,
        default=5,
        help='Return top KK most likely classes.')
    argparser.add_argument(
        '--category_names',
        default='cat_to_name.json',
        help='mapping of categories to real names.')
    argparser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable gpu')
    args = argparser.parse_args()
    checkpoint_file = args.checkpoint
    image_path = args.data_directory
    topk = args.top_k
    category_names = args.category_names
    gpu_enabled = args.gpu
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    checkpoint = torch.load(checkpoint_file)
    device = 'cpu'
    if gpu_enabled:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print("gpu is not available")
    if checkpoint['device'] != device:
        print("Error! The trained state dict was working in {} environment, but the current environment is {}.".format(checkpoint['device'],device))
    model, classifier_name, hidden_layers = create_model(checkpoint['model_name'], hidden_layers = checkpoint['hidden_layers'])
    model.to(device);
    learning_rate = checkpoint['learning_rate']
    model_classifier_state_dict = checkpoint['model_classifier_state_dict']
    if classifier_name == 'classifier':
        model.classifier.load_state_dict(model_classifier_state_dict)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif classifier_name == 'fc':
        model.fc.load_state_dict(model_classifier_state_dict)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    probs, classes = predict(image_path, model, topk, device)
    for i in range(topk):
        print("flower name: {}, probability: {:.2f}%.".format(cat_to_name[classes[i]], probs[i]*100))
    
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image_tensor = process_image(image).to(device).float()
    with torch.no_grad():
        model.eval()
        log_ps = model.forward(image_tensor.view(1, *image_tensor.shape))
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk, dim=1)
    classes = []
    for claz in top_class[0].to('cpu').numpy():
        for k, v in model.class_to_idx.items():
            if claz == v:
                classes.append(k)
                break
    model.train()
    return top_p[0].to('cpu').numpy(), classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    # image = Image.open('flowers/test/1/image_06743.jpg')
    w, h = image.size
    resize_w = 256
    reseze_h = 256
    if w > h:
        resize_w = round(w * 256 / h)
    elif w < h:
        reseze_h = round(h * 256 / w)
    crop_w = round((resize_w - 224) / 2)
    crop_h = round((reseze_h - 224) / 2)
    image = image.resize((resize_w,reseze_h))
    image = image.crop((crop_w, crop_h, resize_w - crop_w, reseze_h - crop_h))
    np_image = np.array(image) / 255
    np_image = (np_image - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
    np_image = np_image.transpose((2,0,1))
    
    return torch.from_numpy(np_image)

if __name__ == '__main__':

    main()