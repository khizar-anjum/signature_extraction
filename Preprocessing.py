from PIL import Image
import numpy as np
from torch.tensor import Tensor


def list_to_tensor(image_list):
    for i, img in enumerate(image_list):
        image_list[i] = Image.fromarray(img).resize([220,155])
    X_arr = np.stack(image_list,axis=0)
    X_arr = X_arr / 255.0
    return Tensor(X_arr).view(len(image_list), 1, 220, 155)
        
