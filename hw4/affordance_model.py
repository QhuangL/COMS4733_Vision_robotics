from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        rgb = data['rgb']
        shape = np.array(rgb).shape
        (img_h, img_w) = (shape[0], shape[1])
        center_point = data['center_point']
        angle = data['angle']

        kps = KeypointsOnImage([
            Keypoint(x=center_point[0], y=center_point[1]),
        ], shape=shape)

        seq = iaa.Sequential([
            iaa.Rotate(-angle.item())
        ])

        image_aug, kps_aug = seq(image=np.array(rgb), keypoints=kps)
        x_aug = kps_aug[0].x
        y_aug = kps_aug[0].y

        target = get_gaussian_scoremap((img_h, img_w), np.array([x_aug, y_aug]))
        target = np.expand_dims(target, axis=0)

        input = torch.from_numpy(image_aug).permute(2, 0, 1).type(torch.float32)
        target = torch.from_numpy(target).type(torch.float32)

        # ======================================================================
        return {
            'input': input,
            'target': target
        }


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)
        self.n_past = n_past_actions

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
        self, 
        rgb_obs: np.ndarray,  
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        # TODO: (problem 2) complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        # coord, angle = None, None
        rotated_rgb_list = []
        rgb_shape = rgb_obs.shape
        n = 8
        angle_unit = 180/n
        for q in range(n):
            seq = iaa.Sequential([iaa.Rotate(-q * angle_unit)])
            rgb_rot = seq(image=rgb_obs)
            rotated_rgb_list.append(rgb_rot)
        input_value = torch.from_numpy(np.stack(rotated_rgb_list)).permute(0, 3, 1, 2).type(torch.float32).to(device)
        input_value.to(device)

        # predict
        predicted = self.predict(input_value)
        # print(predicted.shape)

        nth_prediction = torch.topk(predicted.flatten(), 1)[0][-1].item()
        # print(torch.topk(predicted.flatten(), 1))
        # print('nth_', nth_prediction)
        index = ((predicted == nth_prediction).nonzero())[0]
        # print('index', index)
        coord = (index[3].item(), index[2].item())
        # print('coord', coord)
        angle = index[0].item() * -22.5
        # print('pred', predicted.shape)
        bin = index[0].item()

        # ===============================================================================

        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        sup_seq = iaa.Affine(rotate=-angle)
        supression_map = get_gaussian_scoremap(rgb_obs.shape[:2], keypoint=np.asarray(coord), sigma=4)
        rotated_map = sup_seq(image=supression_map)
        self.past_actions.append((index[0].item(), coord))
        for max_coord in list(self.past_actions):
            bin = max_coord[0]
            if bin == index[0].item():
                print('failed grasp, try again')
                new_pred = predicted
                # print(new_pred.shape)
                new_pred[bin] -= torch.from_numpy(rotated_map)
                # print(new_pred.shape)

                h_prediction = torch.topk((new_pred.flatten()), 1)[0][-1].item()
                index = ((new_pred == h_prediction).nonzero())[0]
                coord = (index[3].item(), index[2].item())
                angle = index[0].item() * -22.5


            # supress past actions and select next-best action
        # ===============================================================================

        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        vis_list = []
        left = []
        right = []

        for i in range(8):
            input = input_value.detach().numpy()[i, ...]
            target = predicted.detach().numpy()[i, ...]
            vis_image = self.visualize(input / 255, target)
            vis_image[127, ...] = 127
            vis_list.append(vis_image)
            if i * -22.5 == angle:
                draw_grasp(vis_image, coord, 0.0)
                chosen_input = np.moveaxis(input, 0, -1)
        #     if i % 2 ==0:
        #         left.append(vis_image)
        #     else:
        #         right.append(vis_image)
        # vis_img = np.concatenate([left, right], axis=0)
        # print(vis_img.shape)
        vis_img = np.concatenate([
            np.concatenate([vis_list[0], vis_list[1]], axis=1),
            np.concatenate([vis_list[2], vis_list[3]], axis=1),
            np.concatenate([vis_list[4], vis_list[5]], axis=1),
            np.concatenate([vis_list[6], vis_list[7]], axis=1),
        ], axis=0)
        # print(vis_img.shape)
        # ===============================================================================

        kps = KeypointsOnImage([Keypoint(x=coord[0], y=coord[1])], shape=chosen_input.shape)

        seq = iaa.Sequential([iaa.Rotate(-angle)])

        image_aug, kps_aug = seq(image=chosen_input, keypoints=kps)
        xy = kps_aug[0].xy

        coord_aug = (int(xy[0]), int(xy[1]))
        # print(coord_aug)



        return coord_aug, -angle, vis_img