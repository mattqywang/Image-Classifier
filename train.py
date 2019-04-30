import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
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
    args = argparser.parse_args()
    checkpoint_file = args.save_dir
    # Load training data
    data_dir = args.data_directory
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
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1024, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 102)

            self.dropout = nn.Dropout(p=0.1)

        def forward(self, x):

            x = self.dropout(F.relu(self.fc1(x)))
            x = self.dropout(F.relu(self.fc2(x)))
            x = self.dropout(F.relu(self.fc3(x)))
            x = F.log_softmax(self.fc4(x), dim=1)
            return x
    
    model.classifier = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device);

    # Start training
    epochs = 15
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
    checkpoint = {'class_to_idx': train_data.class_to_idx,
              'optimizer_state_dict': optimizer.state_dict,
              'model_classifier_state_dict': model.classifier.state_dict()}
    torch.save(checkpoint, checkpoint_file)

if __name__ == '__main__':

    main()