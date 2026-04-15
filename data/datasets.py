# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import transforms
from torch.utils.data import Dataset

from PIL import Image
import io
import torch
from .dct import DCT_base_Rec_Module
import random

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

HF_DATASET_MARKERS = ("dataset_dict.json", "dataset_info.json", "state.json")

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def is_hf_dataset_path(root):
    if root is None or not os.path.isdir(root):
        return False
    return any(os.path.exists(os.path.join(root, marker)) for marker in HF_DATASET_MARKERS)


def _load_hf_dataset(root, is_train):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise ImportError("Please install `datasets` to load Hugging Face datasets: pip install datasets") from exc

    dataset = load_from_disk(root)
    if hasattr(dataset, "keys"):
        if is_train:
            split_candidates = ("train", "training")
        else:
            split_candidates = ("test", "validation", "eval", "val", "dev", "train")

        for split in split_candidates:
            if split in dataset:
                return dataset[split]

        available = ", ".join(dataset.keys())
        raise ValueError(f"Cannot find a suitable split in Hugging Face dataset. Available splits: {available}")

    return dataset


def _get_first_existing(sample, candidates):
    for key in candidates:
        if key in sample:
            return sample[key]
    raise KeyError(f"Cannot find any of these columns in Hugging Face dataset sample: {candidates}")


def _label_to_int(label):
    if isinstance(label, torch.Tensor):
        label = label.item()
    if isinstance(label, str):
        label_lower = label.lower()
        if label_lower in ("real", "0_real", "human", "authentic", "original"):
            return 0
        if label_lower in ("fake", "1_fake", "ai", "aigc", "generated", "synthetic"):
            return 1
        return int(label)
    return int(label)


def _open_image(image):
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, str):
        return Image.open(image).convert('RGB')
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert('RGB')
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert('RGB')
        if image.get("path") is not None:
            return Image.open(image["path"]).convert('RGB')
    return Image.fromarray(image).convert('RGB')


def _append_folder_images(data_list, folder_path, real_dir, fake_dir):
    for image_path in os.listdir(os.path.join(folder_path, real_dir)):
        data_list.append({"image_path": os.path.join(folder_path, real_dir, image_path), "label" : 0})

    for image_path in os.listdir(os.path.join(folder_path, fake_dir)):
        data_list.append({"image_path": os.path.join(folder_path, fake_dir, image_path), "label" : 1})


class TrainDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else (args.eval_data_path or args.data_path)

        self.data_list = []
        self.hf_dataset = None

        if is_hf_dataset_path(root):
            self.hf_dataset = _load_hf_dataset(root, is_train=is_train)
            self.dct = DCT_base_Rec_Module()
            return

        if'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root

            if '0_real' not in os.listdir(file_path):
                for folder_name in os.listdir(file_path):
                
                    assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                    for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                 
                    for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
            
            else:
                for image_path in os.listdir(os.path.join(file_path, '0_real')):
                    self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                    self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
        else:

            for filename in os.listdir(root):

                file_path = os.path.join(root, filename)

                if '0_real' not in os.listdir(file_path):
                    for folder_name in os.listdir(file_path):
                    
                        assert os.listdir(os.path.join(file_path, folder_name)) == ['0_real', '1_fake']

                        for image_path in os.listdir(os.path.join(file_path, folder_name, '0_real')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '0_real', image_path), "label" : 0})
                    
                        for image_path in os.listdir(os.path.join(file_path, folder_name, '1_fake')):
                            self.data_list.append({"image_path": os.path.join(file_path, folder_name, '1_fake', image_path), "label" : 1})
                
                else:
                    for image_path in os.listdir(os.path.join(file_path, '0_real')):
                        self.data_list.append({"image_path": os.path.join(file_path, '0_real', image_path), "label" : 0})
                    for image_path in os.listdir(os.path.join(file_path, '1_fake')):
                        self.data_list.append({"image_path": os.path.join(file_path, '1_fake', image_path), "label" : 1})
                
        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        if self.hf_dataset is not None:
            return len(self.hf_dataset)
        return len(self.data_list)

    def __getitem__(self, index):
        if self.hf_dataset is not None:
            sample = self.hf_dataset[index]
            image_value = _get_first_existing(sample, ("image", "img", "jpg", "png", "file", "path", "image_path"))
            targets = _label_to_int(_get_first_existing(sample, ("label", "labels", "target", "class")))
            image_id = sample.get("image_path", sample.get("path", index))
        else:
            sample = self.data_list[index]
            image_value, targets = sample['image_path'], sample['label']
            image_id = image_value

        try:
            image = _open_image(image_value)
        except:
            print(f'image error: {image_id}')
            return self.__getitem__(random.randint(0, len(self) - 1))


        image = transform_before(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except:
            print(f'image error: {image_id}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self) - 1))

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        


        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

    

class TestDataset(Dataset):
    def __init__(self, is_train, args):
        
        root = args.data_path if is_train else (args.eval_data_path or args.data_path)

        self.data_list = []
        self.hf_dataset = None

        if is_hf_dataset_path(root):
            self.hf_dataset = _load_hf_dataset(root, is_train=is_train)
            self.dct = DCT_base_Rec_Module()
            return

        file_path = root
        entries = os.listdir(file_path)

        if 'real' in entries and 'fake' in entries:
            _append_folder_images(self.data_list, file_path, 'real', 'fake')
        elif '0_real' in entries and '1_fake' in entries:
            _append_folder_images(self.data_list, file_path, '0_real', '1_fake')
        else:
            for folder_name in entries:
                folder_path = os.path.join(file_path, folder_name)
                if not os.path.isdir(folder_path):
                    continue

                folder_entries = os.listdir(folder_path)
                if 'real' in folder_entries and 'fake' in folder_entries:
                    real_dir, fake_dir = 'real', 'fake'
                else:
                    assert '0_real' in folder_entries and '1_fake' in folder_entries
                    real_dir, fake_dir = '0_real', '1_fake'

                _append_folder_images(self.data_list, folder_path, real_dir, fake_dir)


        self.dct = DCT_base_Rec_Module()


    def __len__(self):
        if self.hf_dataset is not None:
            return len(self.hf_dataset)
        return len(self.data_list)

    def __getitem__(self, index):
        if self.hf_dataset is not None:
            sample = self.hf_dataset[index]
            image_value = _get_first_existing(sample, ("image", "img", "jpg", "png", "file", "path", "image_path"))
            targets = _label_to_int(_get_first_existing(sample, ("label", "labels", "target", "class")))
        else:
            sample = self.data_list[index]
            image_value, targets = sample['image_path'], sample['label']

        image = _open_image(image_value)

        image = transform_before_test(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)


        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin) 
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1) 
        x_maxmax1 = transform_train(x_maxmax1)
        
        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))

