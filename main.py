from transformers import ViTModel
from torch import nn
from torchvision import transforms
import torch
from PIL import Image

class BeautyCNN(nn.Module):
    def __init__(self):
        super(BeautyCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.2)
        )
        
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor

        return x

class ViTCNN(nn.Module):
    def __init__(self):
        super(ViTCNN, self).__init__()
        self.vit = ViTModel.from_pretrained('trpakov/vit-face-expression')
        self.fc_vit = nn.Linear(self.vit.config.hidden_size, 256)
        self.fc_vit_dropout = nn.Dropout(0.2)
        
        self.cnn_model = BeautyCNN()
        self.fc_cnn = nn.Linear(64 * 14 * 14, 256)
        self.fc_cnn_dropout = nn.Dropout(0.2)
        
        self.fc_dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256 * 2, 1)

    def forward(self, pixel_values, labels=None):
        # Run ViT and CNN models in parallel
        vit_future = torch.jit.fork(self._process_vit, pixel_values)
        cnn_future = torch.jit.fork(self._process_cnn, pixel_values)
        
        # Wait for the results
        x1 = torch.jit.wait(vit_future)
        x2 = torch.jit.wait(cnn_future)
        
        # Concatenate results and pass through final layers
        x = torch.cat((x1, x2), dim=1)
        values = self.fc2(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(values.view(-1), labels.view(-1))
        return (loss, values) if loss is not None else values

    def _process_vit(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token
        x1 = self.fc_vit(cls_output)
        x1 = self.fc_vit_dropout(x1)
        return x1

    def _process_cnn(self, pixel_values):
        x2 = self.cnn_model(pixel_values)
        x2 = self.fc_cnn(x2)
        x2 = self.fc_cnn_dropout(x2)
        return x2

device = 'cpu'
model = ViTCNN()
model.load_state_dict(torch.load('vit_cnn.weight', map_location=torch.device(device)))
model.eval()

def predict_beauty(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the size expected by the model
        transforms.ToTensor()
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        beauty = model(image_tensor)
        return beauty.item()
    