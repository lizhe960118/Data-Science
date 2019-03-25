import numpy as np
import torch

x = np.array([[0.1,0.12,0.48, 0.32],[0.04,0.14,0.18, 0.2],[0.11, 0.17, 0.99, 0.87]])
x = torch.from_numpy(x)

score, labels = x.max(1)
print(score, labels)

ids = score > 0.2 # anchors
print(ids)

ids = ids.nonzero().squeeze()
print(ids)
print(type(ids))

numpy_scores = score.numpy().astype(np.float)
rank_ids = np.argsort(numpy_scores)[::-1]
print(rank_ids)

if len(rank_ids) > 2:
    choose_ids = rank_ids[:2][::-1].astype(np.int)
    print(choose_ids)
    choose_ids = torch.from_numpy(choose_ids)
    ids = choose_ids
    print(ids)
    print(ids.shape)