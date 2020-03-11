import os
import torch
import logging
from train import Model

logging.basicConfig(level=logging.DEBUG,
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
    logger.info('Deserializing the input data.')
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError("{} not supported by script!".format(request_content_type))


def predict_fn(input_data, model_info):
    logger.info('Predict next template on this pattern series.')
    line = input_data['line']
    num_candidates = model_info['num_candidates']
    input_size = model_info['input_size']
    window_size = model_info['window_size']
    model = model_info['model']

    logger.info(line)
    logger.debug(num_candidates)
    logger.debug(input_size)
    logger.debug(window_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('Current device: {}'.format(device))

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
        predict_cnt +=1
    return {'anomaly_cnt': anomaly_cnt, 'predict_cnt': predict_cnt, 'predict_list': predict_list}


if __name__ == '__main__':
    model_info = model_fn(model_dir)
    for event_id in deeplog_df['EventId']:
        request = json.dumps({'line': event_id})
        input_data = input_fn(request, 'application/json')
        response = predict_fn(input_data, model_info)
