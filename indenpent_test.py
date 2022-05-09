import math

# from model import *
import torch

from DataSet import *
from torch.utils.data import DataLoader
# from model_en import *
from model_layernorm import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ROC(predict_label, test_data_label):
    # ROC Summary of this function goes here

    TruePositive = 0
    TrueNegative = 0
    FalsePositive = 0
    FalseNegative = 0

    # for i,seq_len in enumerate(len(seqlen)):
    #     start_index = sum(seqlen[:i])

    for index in range(len(predict_label)):
        if (test_data_label[index] == 1 and predict_label[index] == 1):
            TruePositive = TruePositive + 1
        if (test_data_label[index] == 0 and predict_label[index] == 0):
            TrueNegative = TrueNegative + 1
        if (test_data_label[index] == 0 and predict_label[index] == 1):
            FalsePositive = FalsePositive + 1
        if (test_data_label[index] == 1 and predict_label[index] == 0):
            FalseNegative = FalseNegative + 1

    print("TruePositive = ", TruePositive)
    print("TrueNegative = ", TrueNegative)
    print("FalsePositive = ", FalsePositive)
    print("FalseNegative = ", FalseNegative)

    ACC = (TruePositive + TrueNegative) / float(TruePositive +
                                                TrueNegative + FalsePositive + FalseNegative + 1e-5)
    SN = TruePositive / float(TruePositive + FalseNegative + 1e-5)

    Spec = TrueNegative / float(TrueNegative + FalsePositive + 1e-5)

    MCC = (TruePositive * TrueNegative - FalsePositive * FalseNegative) / \
          math.sqrt((TruePositive + FalseNegative) * (TrueNegative + FalsePositive) *
                    (TruePositive + FalsePositive) * (TrueNegative + FalseNegative) + 1e-5)

    return (ACC, SN, Spec, MCC)


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, num_epochs=1000):
    best_mcc = 0
    best_mcc = 0
    best_acc = 0
    best_sn = 0
    best_spec = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        ##每个epoch有两个训练阶段

        acc_history = []
        mcc_history = []
        y_train_all = torch.tensor([])
        pre_lab_train_all = torch.tensor([])
        # seqlen = []

        for step, (x, y) in enumerate(train_dataloader):
            # y = torch.LongTensor(y)
            model.train()
            x = torch.FloatTensor(x).to(device)
            # y = torch.FloatTensor(y).to(device)
            output = model(x)

            pre_lab = torch.argmax(output, 1).cpu()
            y = y.long().to(device)
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # seqlen.append(seq_len)

            pre_lab_train_all = torch.cat((pre_lab_train_all, pre_lab), 1)

            y = y.cpu()
            y_train_all = torch.cat((y_train_all, y), 1)
            # print('------{}--------'.format(step))
            step += 1

        train_ROC_acc, train_ROC_sn, train_ROC_spec, train_ROC_mcc = ROC(y_train_all[0], pre_lab_train_all[0])

        print('Train acc:{:.4f}\tsn:{:.4f}\tspec:{:.4f}\tmcc:{:.4f}'.format(train_ROC_acc, train_ROC_sn, train_ROC_spec,
                                                                            train_ROC_mcc))

        # 测试部分
        y_test_all = torch.tensor([])
        pre_lab_test_all = torch.tensor([])

        model.eval()
        with torch.no_grad():
            for x, y in test_dataloader:
                x = torch.FloatTensor(x).to(device)
                # x = x.to(device)

                # y = y.to(device)

                output = model(x)
                pre_lab = torch.argmax(output, 1).cpu()
                y = y.long().to(device)
                loss = criterion(output, y)

                y = y.cpu()
                y_test_all = torch.cat((y_test_all, y), 1)
                pre_lab_test_all = torch.cat((pre_lab_test_all, pre_lab), 1)

        test_ROC_acc, test_ROC_sn, test_ROC_spec, test_ROC_mcc = ROC(y_test_all[0], pre_lab_test_all[0])

        print('Test acc:{:.4f}\tsn:{:.4f}\tspec:{:.4f}\tmcc:{:.4f}'.format(test_ROC_acc, test_ROC_sn, test_ROC_spec,
                                                                           test_ROC_mcc))
        if test_ROC_mcc > best_mcc:
            torch.save(model, 'tr1_cnn4_cnn.pth')
            best_mcc = test_ROC_mcc
            best_acc = test_ROC_acc
            best_sn = test_ROC_sn
            best_spec = test_ROC_spec
            best_epoch = epoch
    print('No.{:.4f}\tBest acc:{:.4f}\tsn:{:.4f}\tspec:{:.4f}\tmcc:{:.4f}'.format(best_epoch, best_acc, best_sn,
                                                                                  best_spec, best_mcc))

def indenpent_test(dataset,model):
    data = dataset
    model = model

    dataloader = DataLoader(data,1)

    y_test_all = torch.tensor([])
    pre_lab_test_all = torch.tensor([])

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = torch.FloatTensor(x).to(device)
            # x = x.to(device)

            # y = y.to(device)

            output = model(x)
            pre_lab = torch.argmax(output, 1).cpu()
            y = y.long().to(device)
            #loss = criterion(output, y)

            y = y.cpu()
            y_test_all = torch.cat((y_test_all, y), 1)
            pre_lab_test_all = torch.cat((pre_lab_test_all, pre_lab), 1)

    test_ROC_acc, test_ROC_sn, test_ROC_spec, test_ROC_mcc = ROC(y_test_all[0], pre_lab_test_all[0])

    return test_ROC_acc, test_ROC_sn, test_ROC_spec, test_ROC_mcc



if __name__ == '__main__':


    test_dataset = Dataset('./data/test_add_norm.dat', './data/test_label.dat')

    model = torch.load('model.pth',map_location=torch.device('cpu'))

    test_ROC_acc, test_ROC_sn, test_ROC_spec, test_ROC_mcc = indenpent_test(test_dataset,model)
    print('acc:{:.4f}\nsn:{:.4f}\nspec:{:.4f}\nmcc:{:.4f}'.format(test_ROC_acc, test_ROC_sn, test_ROC_spec, test_ROC_mcc))

