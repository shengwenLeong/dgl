import time
import numpy as np
import torch
import torch.nn.functional as F


def inference(model, data, epochs, device):
    if hasattr(data, 'features'):
        x = torch.tensor(data.features, dtype=torch.float, device=device)
    else:
        x = data
    model = model.to(device)
    model.eval()
    infer_time = []
    inference_time = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
           
        t_end = time.perf_counter()
        infer_time.append(t_end-t_start)
        print(t_end-t_start)
    print('Avg time :{:.3f}'.format(np.mean(infer_time)))

    return inference_time, infer_time
