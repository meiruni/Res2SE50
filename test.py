import numpy as np
import torch
import pandas as pd
from train import evaluate
def test(config, model, one_iter):
    model.load_state_dict(torch.load(config.save_path), False)
    model.eval()
    loss, acc, f1, report, confusion = evaluate(model, one_iter, config, output_dict=True)
    msg = 'Loss:{0:>5.2}, Acc:{1:>6.2%}, Dev F1 :{2:>6.2%}'
    print(msg.format(loss, acc, f1))
    print("Precision, Recall and F1-Score")
    print("Test Confusion")
    # # 计算每个类别的样本数量
    # class_counts = np.sum(confusion, axis=1)
    # # 将混淆矩阵的每个元素除以该类别的样本数量，得到该类别的分类准确率
    # normalized_confusion = confusion / class_counts[:, np.newaxis]
    # # 打印归一化后的混淆矩阵
    # print(normalized_confusion)
    pd.DataFrame(confusion).to_csv(config.dataset_name+'confusion_matrix.csv', index=False, header=False)
    # 打印混淆矩阵
    print(confusion)

    file = pd.DataFrame(report)
    file = file.T
    print(file)
    file.to_csv(config.evaluate_path, mode='a')