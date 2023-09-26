from typing import Dict, List, Tuple

import os
import argparse
from functools import lru_cache
from random import seed
import json

import numpy as np
from skimage.io import imsave
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from image import read_rgb
import affordance_model
import action_regression_model
from common import save_chkpt, load_chkpt


@lru_cache(maxsize=128)
def read_rgb_cached(file_path):
    return read_rgb(file_path)


class RGBDataset(Dataset):
    def __init__(self, labels_dir: str):
        super().__init__()
        labels_path = os.path.join(labels_dir, 'labels.json')
        labels = json.load(open(labels_path, 'r'))
        label_pairs = list()
        for key, value in labels.items():
            for i, label in enumerate(value):
                label_pairs.append((
                    '{}_{}'.format(key, i),
                    label
                ))
        self.labels_dir = labels_dir
        self.label_pairs = label_pairs

    def __len__(self) -> int:
        return len(self.label_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Read dataset labels.
        return:
        {
            'rgb': np.ndarray (H,W,3), torch.uint8, range [0,255]
            'center_point': np.ndarray (2,), np.float32, range [0,127]
            'angle': np.ndarray (,), np.float32, range [0, 180]
        }
        """
        key, label = self.label_pairs[idx]
        img_path = os.path.join(self.labels_dir, '{}_rgb.png'.format(key))
        rgb = read_rgb_cached(img_path)
        data = {
            'rgb': rgb,
            'center_point': np.array(label[:2], dtype=np.float32),
            'angle': np.array(label[2], dtype=np.float32)
        }
        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = torch.from_numpy(value)
        return data_torch


def get_finger_points(
        center_point: np.ndarray,
        angle: float, width: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the pick position and angle in pixel space,
    return the position of left and right fingers of the gripper
    given the gripper width.
    """
    center_coord = np.array(center_point, dtype=np.float32)
    rad = angle / 180 * np.pi
    direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
    left_coord = center_coord - direction * width
    right_coord = center_coord + direction * width
    return left_coord, right_coord


def get_center_angle(
        left_coord: np.ndarray,
        right_coord: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert the pixel coordinate of left and right fingers to
    gripper center and angle.
    return:
    center_coord: np.ndarray([x,y], dtype=np.float32)
    angle: float
    """
    # TODO: complete this function
    # Why do we need this function?
    # Hint: read get_finger_points
    # Hint: it's a hack
    # ===============================================================================
    center_coord = np.array([(left_coord[0] + right_coord[0]) / 2, (left_coord[1] + right_coord[1]) / 2])
    x, y = right_coord - center_coord
    angle = np.arctan2(y, x) * 180. / np.pi

    return center_coord, angle


class AugmentedDataset(Dataset):
    def __init__(self, rgb_dataset: RGBDataset):
        super().__init__()
        angle_delta = 180 / 8
        self.rgb_dataset = rgb_dataset
        self.aug_pipeline = iaa.Sometimes(0.7, iaa.Affine(
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-angle_delta / 2, angle_delta / 2),
        ))

    def __len__(self) -> int:
        return len(self.rgb_dataset)

    def __getitem__(self, idx: int
                    ) -> Dict[str, torch.Tensor]:
        """
        The output format should be exactly the same as RGBDataset.__getitem__
        """
        data_torch = self.rgb_dataset[idx]
        # TODO: complete this method
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # Hint: use get_finger_points and get_center_angle
        data_torch = self.rgb_dataset[idx]
        rgb = data_torch['rgb']
        center_point = data_torch['center_point']
        angle = data_torch['angle']
        center_point_x, center_point_y = center_point[0].item(), center_point[1].item()
        left_coord, right_coord = get_finger_points(np.array([center_point_x, center_point_y]), angle.item())
        kps = KeypointsOnImage([
            Keypoint(x=left_coord[0], y=left_coord[1]),
            Keypoint(x=right_coord[0], y=right_coord[1])
        ], shape=rgb.shape)
        seq = iaa.Sequential([
            self.aug_pipeline
        ])
        image_aug, kps_aug = seq(image=np.array(rgb), keypoints=kps)
        left_aug_xy = kps_aug[0].xy

        right_coord_aug_xy = kps_aug[1].xy

        center_coord_aug, angle_aug = get_center_angle(np.array([left_aug_xy[0], left_aug_xy[1]]),
                                                       np.array([right_coord_aug_xy[0], right_coord_aug_xy[1]]))

        data = {
            'rgb': image_aug,
            'center_point': np.array(center_coord_aug, dtype=np.float32),
            'angle': np.array(angle_aug, dtype=np.float32)
        }
        data_torch = dict()
        for key, value in data.items():
            data_torch[key] = torch.from_numpy(value)
        # ===============================================================================
        return data_torch


def train(model, train_loader, criterion, optimizer, epoch, device):
    """
        Loop over each sample in the dataloader. Do forward + backward + optimize procedure and print mean IoU on train set.
        :param model (torch.nn.module object): miniUNet model object
        :param train_loader (torch.utils.data.DataLoader object): train dataloader
        :param criterion (torch.nn.module object): Pytorch criterion object
        :param optimizer (torch.optim.Optimizer object): Pytorch optimizer object
        :param epoch (int): current epoch number
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.train()
    epoch_loss = []
    for cur_step, sample_batched in enumerate(train_loader):
        data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Stats
        epoch_loss.append(loss.item())
        # Output stats every [steps] iteration
        if cur_step % 50 == 0:
            global_step = (epoch - 1) * len(train_loader) + cur_step
            print('step#', global_step, 'training loss', np.asarray(epoch_loss).mean())

    return np.asarray(epoch_loss).mean()


def test(model, test_loader, criterion, device, save_dir=None):
    """
        Similar to train(), but no need to backward and optimize.
        :param model (torch.nn.module object): miniUNet model object
        :param test_loader (torch.utils.data.DataLoader object): test dataloader
        :param criterion (torch.nn.module object): Pytorch criterion object
        :return mean_epoch_loss (float): mean loss across this epoch
        :return mean_iou (float): mean iou across this epoch
    """
    model.eval()
    with torch.no_grad():
        epoch_loss = []
        for i, sample_batched in enumerate(test_loader):
            data, target = sample_batched['input'].to(device), sample_batched['target'].to(device)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Stats
            epoch_loss.append(loss.item())

    return np.asarray(epoch_loss).mean()


def save_prediction(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dump_dir: str,
        BATCH_SIZE: int
) -> None:
    print(f"Saving predictions in directory {dump_dir}")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    model.eval()
    with torch.no_grad():
        for batch_ID, sample_batched in enumerate(dataloader):
            print(batch_ID)
            input = sample_batched['input'].numpy()
            target = sample_batched['target'].numpy()

            data = sample_batched['input'].to(model.device)
            output = model.predict(data)
            pred = output.detach().to('cpu').numpy()

            for i in range(len(output)):
                vis_img = model.visualize(
                    input=input[i]/255, output=pred[i], target=target[i])
                idx = batch_ID * BATCH_SIZE + i
                fname = os.path.join(dump_dir, '{:03d}.png'.format(idx))
                imsave(fname, vis_img)


def main():
    parser = argparse.ArgumentParser(description='Model training script')
    parser.add_argument('-m', '--model', default='affordance',
                        help='which model to train: "affordance" or "action_regression"')
    parser.add_argument('-a', '--augmentation', action='store_true',
                        help='flag to enable data augmentation')
    args = parser.parse_args()

    if args.model == 'affordance':
        model_class = affordance_model.AffordanceModel
        dataset_class = affordance_model.AffordanceDataset
        max_epochs = 101
        model_dir = 'data/affordance'
    else:
        model_class = action_regression_model.ActionRegressionModel
        dataset_class = action_regression_model.ActionRegressionDataset
        max_epochs = 201
        model_dir = 'data/action_regression'
    chkpt_path = os.path.join(model_dir, 'best.ckpt')
    dump_dir = os.path.join(model_dir, 'training_vis')

    seed(0)
    torch.manual_seed(0)
    ia.seed(0)

    # Check if GPU is being detected
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset_dir = './data/labels'
    raw_dataset = RGBDataset(dataset_dir)
    train_raw_dataset, test_raw_dataset = random_split(
        raw_dataset, [int(0.9 * len(raw_dataset)), len(raw_dataset) - int(0.9 * len(raw_dataset))])
    if args.augmentation:
        train_raw_dataset = AugmentedDataset(train_raw_dataset)
    train_dataset = dataset_class(train_raw_dataset)
    test_dataset = dataset_class(test_raw_dataset)
    print(f"Train dataset: {len(train_dataset)}; Test dataset: {len(test_dataset)}")

    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("Loading model")
    model = model_class(pretrained=True)
    model.to(device)

    criterion = model.get_criterion()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    epoch = 1
    best_loss = float('inf')
    while epoch <= max_epochs:
        print('Start epoch', epoch)
        train_loss = train(model, train_loader, criterion, optimizer, epoch, device)
        test_loss = test(model, test_loader, criterion, device)
        lr_scheduler.step(test_loss)
        print('Epoch (', epoch, '/', max_epochs, ')')
        print('---------------------------------')
        print('Train loss: %0.4f' % (train_loss))
        print('Test loss: %0.4f' % (test_loss))
        print('---------------------------------')
        # Save checkpoint if is best
        if epoch % 5 == 0 and test_loss < best_loss:
            best_loss = test_loss
            save_chkpt(model, epoch, test_loss, chkpt_path)
            save_prediction(model, test_loader, dump_dir, BATCH_SIZE)
        epoch += 1


if __name__ == "__main__":
    main()
