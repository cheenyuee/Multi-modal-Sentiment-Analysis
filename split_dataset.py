import json
import math
import argparse
import chardet
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-train_file', '--train_file',
                    type=str, default='./dataset/train.json', help='train_file')
parser.add_argument('-test_file', '--test_file',
                    type=str, default='./dataset/test.json', help='test_file')
parser.add_argument('-dev_file', '--dev_file',
                    type=str, default='./dataset/dev.json', help='dev_file')
parser.add_argument('-dev_size', '--dev_size',
                    type=float, default=0.1, help='dev_size')
parser.add_argument('-random_state', '--random_state',
                    type=int, default=6, help='random_state')
arguments = parser.parse_args()

path_train = './dataset/train.txt'
path_test = './dataset/test_without_label.txt'
path_data = './dataset/data/'

df_train_dev = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)
df_train, df_dev = train_test_split(df_train_dev, test_size=arguments.dev_size, random_state=arguments.random_state)

print('total negative:' + str(len(df_train_dev.loc[df_train_dev["tag"] == 'negative'])))
print('total neutral:' + str(len(df_train_dev.loc[df_train_dev["tag"] == 'neutral'])))
print('total positive:' + str(len(df_train_dev.loc[df_train_dev["tag"] == 'positive'])))

print('train negative:' + str(len(df_train.loc[df_train["tag"] == 'negative'])))
print('train neutral:' + str(len(df_train.loc[df_train["tag"] == 'neutral'])))
print('train positive:' + str(len(df_train.loc[df_train["tag"] == 'positive'])))

print('dev negative:' + str(len(df_dev.loc[df_dev["tag"] == 'negative'])))
print('dev neutral:' + str(len(df_dev.loc[df_dev["tag"] == 'neutral'])))
print('dev positive:' + str(len(df_dev.loc[df_dev["tag"] == 'positive'])))

print('test:' + str(len(df_test)))


def get_text(file, encoding):
    text = ''
    with open(file, encoding=encoding) as fp:
        for line in fp.readlines():
            line = line.strip('\n')
            text += line
    return text


def transform(values):
    dataset = []
    for i in range(len(values)):
        guid = str(int(values[i][0]))
        tag = values[i][1]
        if type(tag) != str and math.isnan(tag):
            tag = None
        # print(tag)
        file = path_data + guid + '.txt'
        with open(file, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            if encoding == "GB2312":
                encoding = "GBK"

        text = ''
        try:
            text = get_text(file, encoding)
        except UnicodeDecodeError:
            try:
                text = get_text(file, 'ANSI')
            except UnicodeDecodeError:
                print('UnicodeDecodeError')
        dataset.append({
            'guid': guid,
            'text': text,
            'label': tag,
            'img': path_data + guid + '.jpg',
        })
    return dataset


train_set = transform(df_train.values)
dev_set = transform(df_dev.values)
test_set = transform(df_test.values)

with open(arguments.train_file, 'w', encoding="utf-8") as f:
    json.dump(train_set, f)

with open(arguments.dev_file, 'w', encoding="utf-8") as f:
    json.dump(dev_set, f)

with open(arguments.test_file, "w", encoding="utf-8") as f:
    json.dump(test_set, f)
