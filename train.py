import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from torch.nn.modules.container import Sequential
from torchvision import datasets, transforms, models
from workspace_utils import keep_awake, active_session

def main():
    argparser = argparse.ArgumentParser(
        description='Train the Classifier')
    argparser.add_argument(
        'data_directory',
        help='The directory of training data.')
    argparser.add_argument(
        '--save_dir',
        default='checkpoint.pth',
        help='The directory for saving checkpoints.')
    argparser.add_argument(
        '--arch',
        default='densenet121',
        help='The model name.')
    argparser.add_argument(
        '--hidden_units',
        type=int,
        help='The number of hidden units.')
    argparser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='The number of traning epochs.')
    argparser.add_argument(
        '--learning_rate',
        type=float,
        default=0.003,
        help='The learning rate.')
    argparser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable gpu')
    args = argparser.parse_args()
    checkpoint_file = args.save_dir
    data_dir = args.data_directory
    model_name = args.arch
    hidden_units = args.hidden_units
    epochs = args.epochs
    learning_rate = args.learning_rate
    gpu_enabled = args.gpu
    
    # Load training data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    # Biuld training model
    model, classifier_name, hidden_layers = create_model(model_name, hidden_units)
    criterion = nn.NLLLoss()
    if classifier_name == 'classifier':
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif classifier_name == 'fc':
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    device = 'cpu'
    if gpu_enabled:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            print("gpu is not available")
    model.to(device);
    # Start training
    #train_losses, valid_losses = [], []
    for e in keep_awake(range(epochs)):
        running_loss = 0
        for images, labels in trainloader:
        
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
        
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        else:
            # Prints out training loss, validation loss, and validation accuracy as the network trains
            test_loss = 0
            accuracy = 0
        
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model.forward(images)
                    test_loss += criterion(log_ps, labels)
                
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
        
            model.train()
        
            #train_losses.append(running_loss/len(trainloader))
            #valid_losses.append(test_loss/len(validloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                "validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                "validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

    # Do validation on the test set
    with active_session():
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            model.eval()
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                #print(images.shape)
                log_ps = model.forward(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
            model.train()
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
            "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

    # Save the check point
    model_classifier_state_dict = None
    if classifier_name == 'classifier':
        model_classifier_state_dict = model.classifier.state_dict()
    elif classifier_name == 'fc':
        model_classifier_state_dict = model.fc.state_dict()
    checkpoint = {'class_to_idx': train_data.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict,
              'learning_rate' : learning_rate,
              'device' : device,
              'model_name' : model_name,
              'hidden_layers' : hidden_layers,
              'model_classifier_state_dict': model_classifier_state_dict}
    torch.save(checkpoint, checkpoint_file)

def create_model(model_name, hidden_units = None, hidden_layers = None):
    model = None
    if model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    else:
        if model_name != 'densenet121':
            print("Sorry, the model {} is not available in this version, we will use densenet121 instead.".format(model_name))
            print("Please refer to the supported model list: vgg13, vgg16, resnet18, alexnet, densenet161, inception_v3")
        model = models.densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    models_dict = model._modules
    classifier_name = None
    in_features = None
    out_features = 102
    for k in models_dict.keys():
        if k == 'fc' or k == 'classifier':
            classifier_name = k
            break
    linears = models_dict[classifier_name]
    if type(linears) is Linear:
        in_features = linears.in_features
    elif type(linears) is Sequential:
        for linear in linears._modules.values():
            if type(linear) is Linear:
                in_features = linear.in_features
                break
    if hidden_units is not None:
        hidden_layers = [in_features, hidden_units, out_features]
    elif hidden_layers is None:
        hidden_layers = [in_features, 512, 256, 128, out_features]
        if in_features > 1024:
            hidden_layers.insert(1,1024)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_layers = nn.ModuleList()
            for i in range(len(hidden_layers) - 2):
                self.hidden_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))

            self.out = nn.Linear(hidden_layers[-2], hidden_layers[-1])

            self.dropout = nn.Dropout(p=0.1)

        def forward(self, x):
            for fc in self.hidden_layers:
                x = self.dropout(F.relu(fc(x)))

            x = F.log_softmax(self.out(x), dim=1)
            return x
    if classifier_name == 'classifier':
        model.classifier = Classifier()
    elif classifier_name == 'fc':
        model.fc = Classifier()
    return model, classifier_name, hidden_layers

if __name__ == '__main__':
    main()