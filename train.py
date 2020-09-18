from preprocess import *
from model import *
from function import *
import torch
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import f1_score

writer = SummaryWriter('runs/localCNN')

def train_net(epoch, model, data_trainer, criterion, optimizer):
    running_loss = 0.0
    correct = 0
    train_loss = 0.0

    pred_axis = torch.zeros(1, 1, dtype=torch.long).to('cuda')  # 网络预测值
    true_axis = torch.zeros(1, 1, dtype=torch.long).to('cuda')  # 真实值


    model.train()
    prefetcher = data_prefetcher(data_trainer)
    data, label = prefetcher.next()
    batch_idx = 0
    while data is not None:
        batch_idx += 1

        if torch.any(torch.isnan(data)) | torch.any(torch.isnan(label)):
            break

        optimizer.zero_grad()
        output = model(data)
        _, pred_index = output.topk(2, largest=True, sorted=False)
        pred = torch.zeros_like(output)
        for i in range(len(pred_index)):
            pred[i,pred_index[i,0]] = 1 
            pred[i,pred_index[i,1]] = 1 

        pred = onehot2num(pred).long()
        labe = onehot2num(label).long()
        pred_axis = torch.cat((pred_axis, pred), 0)
        true_axis = torch.cat((true_axis, labe), 0)


        correct += pred.eq(labe).sum().item()

        loss = criterion(output, label.float())
        train_loss += loss.item()
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), 15)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20)
        optimizer.step()
        # scheduler.step()
        running_loss += loss.item()
        if  batch_idx  % 1 == 0:
            writer.add_scalar('training loss',
                              running_loss / 10,
                              (epoch - 1) * len(data_trainer) + batch_idx)
            print('Train epoch:{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                          len(data_trainer.dataset),
                                                                          100. * batch_idx / len(data_trainer),
                                                                          running_loss / 1))
            running_loss = 0.0
        data, label = prefetcher.next()
    print("Training Finished")
    #true_list = true_list.squeeze().long().to('cpu').tolist()
    #pred_list = pred_list.squeeze().long().to('cpu').tolist()
    
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss / len(data_trainer.dataset), correct, len(data_trainer.dataset),
        100. * correct / len(data_trainer.dataset)))

def test_net(epoch, model, test_loader):
    test_loss = 0.0
    correct = 0

    pred_axis = torch.zeros(1, 1, dtype=torch.long).to('cuda')  # 网络预测值
    true_axis = torch.zeros(1, 1, dtype=torch.long).to('cuda')  # 真实值

    model.eval()
    with torch.no_grad():

        prefetcher = data_prefetcher(test_loader)
        data, label = prefetcher.next()
       
        batch_idx = 0
        while data is not None:
            batch_idx += 1

            output = model(data)  #batch * class

            _, pred_index = output.topk(2, largest=True, sorted=False)
            pred = torch.zeros_like(output)
            for i in range(len(pred_index)):
                pred[i,pred_index[i,0]] = 1 
                pred[i,pred_index[i,1]] = 1 

            pred = onehot2num(pred).long()
            labe = onehot2num(label).long()
            correct += pred.eq(labe).sum().item()

            test_loss += criterion(output, label.float()) # 将一批的损失相加

            pred_axis = torch.cat((pred_axis, pred), 0)
            true_axis = torch.cat((true_axis, labe), 0)
            '''
            if (batch_idx+1) % 50 == 0:
                print('[{}/{} ({:.0f}%)]'.format(batch_idx*len(data), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))
            '''
            data, label = prefetcher.next()

    test_loss /= len(test_loader.dataset)
    writer.add_scalar('valid loss', test_loss, epoch)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    #MAE_error, max_error, std_error = MAE(epoch, pred_list[:,1:], true_list[:,1:], plt_flag=True, data_name=name_array[:,1:])
    true_axis = true_axis.squeeze().long().to('cpu').tolist()
    pred_axis = pred_axis.squeeze().long().to('cpu').tolist()
    F1_score = f1_score(true_axis, pred_axis, average='macro')

    writer.add_scalar(' valid F1 Score', F1_score, epoch )
    
    valid_F1_score_list.append(F1_score)

    print('\nvalid_F1_score = {:.4f}\n'.format(F1_score))




if __name__ == '__main__':
    learning_rate = 0.003
    epoch = 2000
    torch.backends.cudnn.benchmark = True
    Frozen_Layers = False
    
    train = pd.read_csv('data\All__train_data.csv').to_numpy()
    train_data = torch.from_numpy(train[:,:240])
    train_label = torch.from_numpy(train[:,240:])
    train_set = TensorDataset(train_data, train_label)

    valid = pd.read_csv('data\All__valid_data.csv').to_numpy()
    valid_data = torch.from_numpy(valid[:, :240])
    valid_label = torch.from_numpy(valid[:, 240:])
    valid_set = TensorDataset(valid_data, valid_label)
    
    valid_F1_score_list = []

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    model = Deep_CNN()
    #model = ResNet_TO_LSTM()
    
    for i in model.modules():
        weight_init(i)
    device = torch.device('cuda')
    #criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 3.774, 14.49, 1]))
    criterion = mycriterion
    #criterion.to(device)### 要想使用GPU，此处必须使用cuda()

    if Frozen_Layers == True:
        for k, v in model.named_parameters():
            if 'lstm' not in k and 'fc' not in k:
                v.requires_grad = False
        print('CNN FROZENED!')
        for k, v in model.named_parameters():
            if v.requires_grad == True:
                print('training:',k)

    bias_list = (param for name, param in model.named_parameters() if 'bias' in name and param.requires_grad == True)
    others_list = (param for name, param in model.named_parameters() if 'bias' not in name and param.requires_grad == True)
    parameters = [{'params': bias_list, 'weight_decay': 0},
                  {'params': others_list}]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    model.to(device)

    #加载LSTM预训练参数
    #pretrained_params = torch.load('best_model.pkl')
    #model.load_state_dict(pretrained_params, strict=False)
    #model.load_state_dict(torch.load('bestModel5500A.pkl'),strict=False)

    max_f1 = 0

    for i in range(1, epoch+1):

        train_net(i, model, train_loader, criterion, optimizer)

        test_net(i, model, valid_loader)
        if valid_F1_score_list[-1] > max_f1:
            max_f1 = valid_F1_score_list[-1]
            torch.save(model.state_dict(), 'bestModel5500A.pkl')  # save only the parameters

"""    model.load_state_dict(torch.load('best_deep_cnn_classification.pkl'))
    test_set = load_data_GPU('test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=10, pin_memory=True)
    test_net(1, model,test_loader, test=True)"""