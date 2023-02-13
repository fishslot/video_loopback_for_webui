from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import numpy as np
from .fastdvdnet.models import FastDVDnet as FastDVDnet_model
from .fastdvdnet.fastdvdnet import denoise_seq_fastdvdnet
from .utils import get_image_paths

# sed -i "22c from skimage.metrics import peak_signal_noise_ratio as compare_psnr" ./fastdvdnet/utils.py


class FastDVDNet:
    def __init__(
            self, alpha, noise_sigma=60,
            model_path=Path(__file__).parent/'fastdvdnet'/'model.pth',
            temporal_size=5):
        assert torch.cuda.is_available()
        print("Loading FastDVDNet: ", model_path.absolute(), model_path.exists())
        self.device = torch.device('cuda')
        self.alpha = alpha
        self.noise_sigma = noise_sigma / 255
        self.temporal_size = temporal_size
        device_ids = [0]
        model_temp = FastDVDnet_model(num_input_frames=temporal_size)
        model_temp = nn.DataParallel(model_temp, device_ids=device_ids).cuda()
        model_temp.load_state_dict(torch.load(model_path))
        model_temp.eval()
        self.model = model_temp

    def process(self, input_path):
        image_paths = get_image_paths(input_path)
        seq_images = [Image.open(p) for p in image_paths]
        with torch.no_grad():
            # process data
            seq = torch.from_numpy(
                np.stack(
                    seq_images, axis=0
                ).transpose(0, 3, 1, 2).astype(np.float32)/255.
            ).to(self.device)
            # Add noise
            noise = torch.empty_like(seq).normal_(
                mean=0, std=self.noise_sigma).to(self.device)
            seqn = seq + noise
            noisestd = torch.FloatTensor([self.noise_sigma]).to(self.device)
            denframes = denoise_seq_fastdvdnet(
                seq=seqn,
                noise_std=noisestd,
                temp_psz=self.temporal_size,
                model_temporal=self.model
            )
        denframes = denframes.data.cpu().numpy()
        denframes = (denframes * 255.).clip(0, 255)\
            .astype(np.uint8).transpose(0, 2, 3, 1)
        for o_img, p_img, p in zip(seq_images, denframes, image_paths):
            p_img = Image.fromarray(p_img)
            Image.blend(o_img, p_img, self.alpha).save(p)
