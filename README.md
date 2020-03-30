# deeplog

PyTorch implements "DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning"

## Preprocessing

First, install PyTorch, boto3 and the auto-parser from same author of DeepLog.

```bash
pip install -r requirements.txt
```

Then we use open data from logpai's loghub

```python
python preprocess.py
```

## Train

```python
python train.py --num-class 1143 --num-candidates 114 --epochs 35 --window-size 3 --local True
```

## Predict

```python
python predict.py
```

## Result

| Accuracy  | 0.90404 |
|-----------|---------|
| Precision | 0.51054 |
| Recall    | 0.87681 |
| F1        | 0.64533 |
