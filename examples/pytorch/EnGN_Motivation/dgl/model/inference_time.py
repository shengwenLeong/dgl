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
    stage1 = []
    stage2 = []
    stage3 = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if epoch > 3:
            t_start = time.perf_counter()
        with torch.no_grad():
            _, s1, s2, s3 = model(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if epoch > 3:    
            t_end = time.perf_counter()
            infer_time.append((t_end-t_start)*1000000)
            stage1.append(s1)
            stage2.append(s2)
            stage3.append(s3)
            #print(t_end-t_start)
    print('Avg time :{:.3f}'.format(np.mean(infer_time)))

    return infer_time, stage1, stage2, stage3
