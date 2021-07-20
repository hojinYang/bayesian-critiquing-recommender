import math
import time
import torch
import numpy as np
from collections import OrderedDict


class Evaluator:
    def __init__(self):
        return 

    def evaluate(self, model, dataset, test_batch_size):
        model.eval()

        model.before_evaluate()
        
        preds, ys = model.predict(dataset, test_batch_size)
        return np.sqrt((np.sum((preds - ys)**2)) / len(ys))