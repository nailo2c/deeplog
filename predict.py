import sys
import json
import logging
from deeplog.deeplog import model_fn, input_fn, predict_fn


logging.basicConfig(level=logging.WARNING,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

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

        if p == 1 and t == 1:
            TP += 1
        elif p == 1 and t == 0:
            FP += 1
        elif p == 0 and t == 1:
            FN += 1
        else:
            TN += 1

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
