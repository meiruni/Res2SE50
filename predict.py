
import torch
from utils import load_predict_data,Config
from model import *

def predict(config, model, file_path):
    """
    预测
    """
    model.load_state_dict(torch.load(config.save_path), False)
    model.eval()
    with torch.no_grad():
        img_tensor = load_predict_data(config, file_path)
        label_predict = model(img_tensor)
        label_predict = torch.argmax(label_predict, 1)
        print(config.id2class[label_predict.data.item()])

if __name__ == '__main__':
    config = Config()
    model = Res2SE50(config.num_epochs).to(config.device)
    test_img_path = './DataPrep/data/NWPU-RESISC45/valid-NWPU-RESISC45/8/circular_farmland_007.jpg'
    predict(config, model, test_img_path)
