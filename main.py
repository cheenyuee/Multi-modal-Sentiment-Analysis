import time
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
# 导入实现的数据加载模块
from model.utils import load_data, save_data
# 导入实现的model
from model.Multimodal_model import MultimodalModel


def train(args, train_dataloader, dev_dataloader):
    model = MultimodalModel(args).to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    num_epochs = args.epoch
    best_accuracy = 0.0
    for epoch in range(1, num_epochs + 1):
        train_l_sum, train_acc_sum, n = 0., 0., 0
        for step, bach in enumerate(train_dataloader):
            ids, bach_text, bach_img, y = bach
            bach_text = bach_text.to(device=args.device)
            bach_img = bach_img.to(device=args.device)
            y = y.to(device=args.device)
            y_hat_text, y_hat_img, y_hat = model(bach_text=bach_text, bach_img=bach_img)
            ensemble(y_hat_text, y_hat_img, y_hat)  #
            # 训练阶段使用三个损失
            if y_hat is not None:
                loss = loss_func(y_hat, y.long()).sum() + \
                       0.2 * loss_func(y_hat_text, y.long()).sum() + \
                       0.2 * loss_func(y_hat_img, y.long()).sum()
            elif y_hat_text is not None and y_hat_img is None:
                loss = loss_func(y_hat_text, y.long()).sum()
            elif y_hat_text is None and y_hat_img is not None:
                loss = loss_func(y_hat_img, y.long()).sum()
            else:
                continue
            # 梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
            # print('step %d, loss %.4f, train acc %.3f' % (step, train_l_sum / n, train_acc_sum / n))
        print('epoch %d, loss %.4f, train acc %.3f' % (epoch, train_l_sum / n, train_acc_sum / n))
        # 计算验证集准确率
        accuracy = evaluate(args, model, dev_dataloader, epoch)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            time_now_ = time.strftime('/%Y-%m-%d-%H-%M-%S-', time.localtime(time.time()))
            # 保存检查点
            torch.save(model.state_dict(), args.checkpoints_dir + time_now_ + 'best' + '-checkpoint.pth')
    torch.save(model.state_dict(), args.checkpoints_dir + '/final_checkpoint.pth')


def evaluate(args, model, dev_dataloader, epoch=None):
    # 计算验证集准确率
    dev_acc_sum_text, dev_acc_sum_img, dev_acc_sum, dev_acc_sum_ensemble, n = 0., 0., 0., 0., 0
    for step, bach in enumerate(dev_dataloader):
        ids, bach_text, bach_img, y = bach
        bach_text = bach_text.to(device=args.device)
        bach_img = bach_img.to(device=args.device)
        y = y.to(device=args.device)
        y_hat_text, y_hat_img, y_hat = model(bach_text=bach_text, bach_img=bach_img)
        dev_acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        dev_acc_sum_text += (y_hat_text.argmax(dim=1) == y).float().sum().item()
        dev_acc_sum_img += (y_hat_img.argmax(dim=1) == y).float().sum().item()
        dev_acc_sum_ensemble += (ensemble(y_hat_text, y_hat_img, y_hat) == y).float().sum().item()
        n += y.shape[0]
    if epoch:
        print('epoch %d, dev acc %.4f(text), dev acc %.4f(image), dev acc %.4f(fusion),    dev acc %.4f(ensemble)'
              % (epoch, dev_acc_sum_text / n, dev_acc_sum_img / n, dev_acc_sum / n, dev_acc_sum_ensemble / n))
    else:
        print('dev acc %.4f(text), dev acc %.4f(image), dev acc %.4f(fusion),    dev acc %.4f(ensemble)'
              % (dev_acc_sum_text / n, dev_acc_sum_img / n, dev_acc_sum / n, dev_acc_sum_ensemble / n))
    return dev_acc_sum / n


def ensemble(y_hat_text, y_hat_img, y_hat_multi):
    result = y_hat_multi.argmax(dim=1)
    tmp1 = y_hat_text.argmax(dim=1)
    tmp2 = y_hat_img.argmax(dim=1)
    for i in range(len(result)):
        if tmp1[i] == tmp2[i] and tmp1[i] != result[i]:
            result[i] = tmp1[i]
    return result


def test(args, test_dataloader, dev_dataloader):
    model = MultimodalModel(args).to(device=args.device)
    model.load_state_dict(torch.load(args.checkpoints_dir + '/final_checkpoint.pth'))
    # 计算验证集准确率
    evaluate(args, model, dev_dataloader)
    # 测试集预测
    print('predicting...')
    predict_list = []
    # print(test_dataloader.dataset.label_dict_str)
    for step, bach in enumerate(test_dataloader):
        ids, bach_text, bach_img, _ = bach
        bach_text = bach_text.to(device=args.device)
        bach_img = bach_img.to(device=args.device)
        y_hat_text, y_hat_img, y_hat = model(bach_text=bach_text, bach_img=bach_img)
        predict_y = y_hat.argmax(dim=1)  # 使用主分类器
        # predict_y = ensemble(y_hat_text, y_hat_img, y_hat)  # ensemble主分类器和辅助分类器集成
        for i in range(len(ids)):
            item_id = ids[i]
            tag = test_dataloader.dataset.label_dict_str[int(predict_y[i])]
            predict_dict = {
                'guid': item_id,
                'tag': tag,
            }
            predict_list.append(predict_dict)
    save_data(args.test_output_file, predict_list)


if __name__ == '__main__':
    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('beginning--------------------------' + time_now + '---------------------------')
    # 命令参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true', help='Whether to run training.')
    parser.add_argument('--do_test', action='store_true', help='Whether to run testing.')

    # 路径类参数
    parser.add_argument('-checkpoints_dir', '--checkpoints_dir',
                        type=str, default='./checkpoints', help='output_dir')
    parser.add_argument('-test_output_file', '--test_output_file',
                        type=str, default='./test_with_label.txt', help='test_output_file')

    parser.add_argument('-train_file', '--train_file',
                        type=str, default='./dataset/train.json', help='train_file')
    parser.add_argument('-dev_file', '--dev_file',
                        type=str, default='./dataset/dev.json', help='dev_file')
    parser.add_argument('-test_file', '--test_file',
                        type=str, default='./dataset/test.json', help='test_file')
    parser.add_argument('-pretrained_model', '--pretrained_model',
                        type=str, default='roberta-base', help='pretrained_model')
    # 训练类参数
    parser.add_argument("-lr", "--lr",
                        type=float, default=1e-5, help='learning rate')
    parser.add_argument("-dropout", "--dropout",
                        type=float, default=0.0, help='dropout')
    parser.add_argument("-epoch", "--epoch",
                        type=int, default=10, help='epoch')
    parser.add_argument("-batch_size", "--batch_size",
                        type=int, default=4, help='bach size')
    # 模型类参数
    parser.add_argument("--img_size", "--img_size",
                        type=int, default=384, help='image size')
    parser.add_argument("--text_size", "--text_size",
                        type=int, default=64, help='text size')

    arguments = parser.parse_args()

    # cuda
    arguments.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:' + str(arguments.device))

    # 加载数据集
    print('data loading...')

    if arguments.do_train:
        train_set, dev_set = load_data(arguments)
        train_dataloader_ = DataLoader(train_set, shuffle=True, batch_size=arguments.batch_size)
        eval_dataloader_ = DataLoader(dev_set, shuffle=True, batch_size=arguments.batch_size)
        print('model training...')
        train(arguments, train_dataloader_, eval_dataloader_)
    if arguments.do_test:
        test_set, dev_set = load_data(arguments)
        test_dataloader_ = DataLoader(test_set, shuffle=False, batch_size=arguments.batch_size)
        eval_dataloader_ = DataLoader(dev_set, shuffle=False, batch_size=arguments.batch_size)
        print('model testing...')
        test(arguments, test_dataloader_, eval_dataloader_)

    if arguments.do_train is False and arguments.do_test is False:
        print("you didn't train or test.")
        print("you can train or test with argument --do_train or --do_test.")

    time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('ending-----------------------------' + time_now + '---------------------------')
