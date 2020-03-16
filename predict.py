import os
import sys
import json
import torch
import logging
from train import Model


logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):
    logger.info('Loading the model.')
    model_info = {}
    with open(os.path.join(model_dir, 'model_info.pth'), 'rb') as f:
        model_info = torch.load(f)
    logger.debug('model_info: {}'.format(model_info))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))
    model = torch.nn.DataParallel(Model(input_size=model_info['input_size'],
                                        hidden_size=model_info['hidden_size'],
                                        num_layers=model_info['num_layers'],
                                        num_classes=model_info['num_classes']))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    input_size = model_info['input_size']
    window_size = model_info['window_size']
    num_candidates = model_info['num_candidates']
    return {'model': model.to(device),
            'window_size': window_size,
            'input_size': input_size,
            'num_candidates': num_candidates}


def input_fn(request_body, request_content_type):
    # logger.debug('Deserializing the input data.')
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("{} not supported by script!".format(request_content_type))


def predict_fn(input_data, model_info):
    # logger.info('Predict next template on this pattern series.')
    line = input_data['line']
    num_candidates = model_info['num_candidates']
    input_size = model_info['input_size']
    window_size = model_info['window_size']
    model = model_info['model']

    # logger.debug(f'line: {line}')
    # logger.debug(f'num_candidates: {num_candidates}')
    # logger.debug(f'input_size: {input_size}')
    # logger.debug(f'window_size: {window_size}')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # logger.debug('Current device: {}'.format(device))

    predict_cnt = 0
    anomaly_cnt = 0
    predict_list = [0] * len(line)
    for i in range(len(line) - window_size):
        seq = line[i:i + window_size]
        label = line[i + window_size]
        seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
        label = torch.tensor(label).view(-1).to(device)
        output = model(seq)
        predict = torch.argsort(output, 1)[0][-num_candidates:]
        if label not in predict:
            anomaly_cnt += 1
            predict_list[i + window_size] = 1
        predict_cnt += 1
    return {'anomaly_cnt': anomaly_cnt, 'predict_cnt': predict_cnt, 'predict_list': predict_list}


if __name__ == '__main__':
    ##############
    # Load Model #
    ##############
    model_dir = './model'
    model_info = model_fn(model_dir)

    ###########
    # predict #
    ###########
    test_abnormal_list = []
    with open('test_abnormal', 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            request = json.dumps({'line': line})
            input_data = input_fn(request, 'application/json')
            response = predict_fn(input_data, model_info)
            test_abnormal_list.append(response)

    test_normal_list = []
    with open('test_normal', 'r') as f:
        for line in f.readlines():
            line = list(map(lambda n: n - 1, map(int, line.strip().split())))
            request = json.dumps({'line': line})
            input_data = input_fn(request, 'application/json')
            response = predict_fn(input_data, model_info)
            test_normal_list.append(response)

    ##############
    # Evaluation #
    ##############
    thres = 22
    abnormal_has_anomaly = [1 if t['anomaly_cnt'] > thres else 0 for t in test_abnormal_list]
    abnormal_cnt_anomaly = [t['anomaly_cnt'] for t in test_abnormal_list]
    abnormal_predict = []
    for test_abnormal in test_abnormal_list:
        abnormal_predict += test_abnormal['predict_list']

    normal_has_anomaly = [1 if t['anomaly_cnt'] > thres else 0 for t in test_normal_list]
    normal_cnt_anomaly = [t['anomaly_cnt'] for t in test_normal_list]
    normal_predict = []
    for test_normal in test_normal_list:
        normal_predict += test_normal['predict_list']

    ground_truth = [1]*len(abnormal_has_anomaly) + [0]*len(normal_has_anomaly)
    predict = abnormal_has_anomaly + normal_has_anomaly
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    accu = 0
    for p, t in zip(predict, ground_truth):
        if p == t:
            accu += 1

        if p==1 and t==1:
            TP += 1
        elif p==1 and t==0:
            FP += 1
        elif p==0 and t==1:
            FN += 1
        else:
            TN +=1

    print(f'thres: {thres}')
    print(f'TP: {TP}')
    print(f'FP: {FP}')
    print(f'FP: {TN}')
    print(f'FP: {FN}')

    accuracy = accu / len(predict)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)

    print(f'accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {F1}')
