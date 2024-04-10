import torch
import converter
import dataset
from PIL import Image
from dataset import fixed_size_tensor
import model 
from model import CRNN
import matplotlib.pyplot as plt


model_path = r'CRNN\best_model3.pth'
img_path = r'CRNN\cropped_lps\cropped_lps\32634.jpg'
alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

model = CRNN(32,1,37,256)
#if torch.cuda.is_available():
#    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


converter = converter.strLabelConverter(alphabet)

image = Image.open(img_path)
plt.imshow(image)
plt.axis('off')
plt.title("original_image")
plt.show()


image = fixed_size_tensor(img_path)

#if torch.cuda.is_available():
#    image = image.cuda()
image = image.view(1, *image.size())
image = image.unsqueeze(0)
print(image.size())
model.eval()
preds = model(image)
_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))

