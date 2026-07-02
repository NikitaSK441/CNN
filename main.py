import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import time
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

image_transforms = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
])

datapath = r'D:\Dataset'
data = ImageFolder(root=datapath, transform = image_transforms)

train_data, test_data = random_split(data,[2000,400])

train_length = 2000
test_length = 400

train_dataloader = DataLoader(train_data,batch_size = 64,shuffle=True)
test_dataloader = DataLoader(test_data,batch_size = 64,shuffle=False)

class ConvolutionalNetwork(nn.Module):
        def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3,16,5,1,2)
                self.conv2 = nn.Conv2d(16,32,5,1,2)
                self.conv3 = nn.Conv2d(32, 64, 5, 1, 2)
                self.fc1 = nn.Linear(64*16*16, 120)
                self.fc2 = nn.Linear(120, 80)
                self.fc3 = nn.Linear(80,64)
                self.fc4 = nn.Linear(64,3)
        def forward(self,X):
                X = F.relu(self.conv1(X))
                X = F.max_pool2d(X,7,2,3) # 7x7 kernel, stride 2 , padding 3
                X = F.relu(self.conv2(X))
                X = F.max_pool2d(X, 7, 2, 3)  # 7x7 kernel, stride 2 , padding 3
                X = F.relu(self.conv3(X))
                X = F.max_pool2d(X, 7, 2, 3)  # 7x7 kernel, stride 2 , padding 3

                X = torch.flatten(X,1)

                X = F.relu(self.fc1(X))
                X = F.relu(self.fc2(X))
                X = F.relu(self.fc3(X))
                X = self.fc4(X)

                return X

torch.manual_seed(42)
model = ConvolutionalNetwork().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.00005)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

start_time = time.time()

epochs = 25

train_losses = []
test_losses = []
train_correct = []
test_correct = []

best_loss = float('inf')
number_batch = 0
for i in range(epochs):
        trn_corr = 0
        tst_corr = 0

        running_train_loss = 0.0
        running_test_loss = 0.0

        model.train()
        for batch, (X_train, y_train) in enumerate(train_dataloader):
                X_train, y_train = X_train.to(device), y_train.to(device)

                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()

                predicted = torch.max(y_pred.data, 1)[1]
                trn_corr += (predicted == y_train).sum().item()

        epoch_train_loss = running_train_loss / len(train_dataloader)
        train_losses.append(epoch_train_loss)
        train_correct.append(trn_corr)

        model.eval()
        with torch.no_grad():
                for batch, (X_test, y_test) in enumerate(test_dataloader):
                        X_test, y_test = X_test.to(device), y_test.to(device)
                        y_val = model(X_test)

                        val_loss = criterion(y_val, y_test)
                        running_test_loss += val_loss.item()

                        predicted = torch.max(y_val.data, 1)[1]
                        tst_corr += (predicted == y_test).sum().item()

        epoch_test_loss = running_test_loss / len(test_dataloader)
        test_losses.append(epoch_test_loss)
        test_correct.append(tst_corr)

        current_val_loss = epoch_test_loss

        if current_val_loss < best_loss:
                best_loss = current_val_loss
                torch.save(model.state_dict(), 'best_model.pt')

        print(f'Epoch: {i}, Batch: {number_batch}, Loss: {loss.item()}')
        number_batch = number_batch + 1
current_time = time.time()
total = current_time - start_time
print("Trained for",total/60,"minutes")

plt.plot(train_losses, label="Testing loss")
plt.plot(test_losses, label="Validation loss")

plt.title("Loss at Epoch")
plt.legend()
plt.show()

train_acc = [t / train_length for t in train_correct]
test_acc = [t / test_length for t in test_correct]

plt.plot(train_acc, label="Testing accuracy")
plt.plot(test_acc, label="Validation accuracy")

plt.title("Accuracy at the end of each epoch")
plt.legend()
plt.show()

model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.eval()

image_name = input("Enter the path to your image: ")
img = Image.open(image_name)

image_transforms_input = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
])

transformed_image = image_transforms_input(img)

image_tensor = transformed_image.unsqueeze(0).to(device)

with torch.no_grad():
        output = model(image_tensor)

predicted_index = torch.argmax(output, dim=1).item()

class_names = data.classes
predicted_class_name = class_names[predicted_index]

print(f"Class index: {predicted_index}")
print(f"Class name:  {predicted_class_name}")