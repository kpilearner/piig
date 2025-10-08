"""
Data loading utilities for RGB-Infrared paired dataset
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob


class RGBInfraredDataset(Dataset):
    """
    Dataset for paired RGB and infrared images.

    Expected directory structure:
        data_root/
            rgb/
                img_001.png
                img_002.png
                ...
            infrared/
                img_001.png
                img_002.png
                ...
    """

    def __init__(
        self,
        data_root,
        rgb_dir='rgb',
        infrared_dir='infrared',
        image_size=512,
        mode='train',
        rgb_transform=None,
        ir_transform=None
    ):
        """
        Args:
            data_root: Root directory containing rgb and infrared subdirectories
            rgb_dir: Name of RGB subdirectory
            infrared_dir: Name of infrared subdirectory
            image_size: Size to resize images to
            mode: 'train' or 'val'
            rgb_transform: Optional custom transform for RGB images
            ir_transform: Optional custom transform for infrared images
        """
        self.data_root = data_root
        self.image_size = image_size
        self.mode = mode

        # Get image paths
        rgb_path = os.path.join(data_root, rgb_dir)
        ir_path = os.path.join(data_root, infrared_dir)

        # Find all RGB images
        rgb_files = sorted(glob.glob(os.path.join(rgb_path, '*.*')))
        ir_files = sorted(glob.glob(os.path.join(ir_path, '*.*')))

        # Ensure paired data
        assert len(rgb_files) == len(ir_files), \
            f"Mismatch: {len(rgb_files)} RGB images, {len(ir_files)} infrared images"

        self.rgb_files = rgb_files
        self.ir_files = ir_files

        # Define transforms
        if rgb_transform is None:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = rgb_transform

        if ir_transform is None:
            self.ir_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.ir_transform = ir_transform

        print(f"Loaded {len(self.rgb_files)} {mode} samples from {data_root}")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # Load images
        rgb_img = Image.open(self.rgb_files[idx]).convert('RGB')
        ir_img = Image.open(self.ir_files[idx]).convert('L')  # Grayscale

        # Apply transforms
        rgb_tensor = self.rgb_transform(rgb_img)
        ir_tensor = self.ir_transform(ir_img)

        # Get image ID (filename without extension)
        image_id = os.path.splitext(os.path.basename(self.rgb_files[idx]))[0]

        return {
            'rgb': rgb_tensor,
            'infrared': ir_tensor,
            'image_id': image_id
        }


def create_dataloaders(
    train_root,
    val_root=None,
    batch_size=8,
    num_workers=4,
    image_size=512,
    **dataset_kwargs
):
    """
    Create training and validation dataloaders.

    Args:
        train_root: Root directory for training data
        val_root: Root directory for validation data (if None, use train_root)
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Image size
        **dataset_kwargs: Additional arguments for RGBInfraredDataset

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader (or None if val_root is None)
    """
    # Create training dataset
    train_dataset = RGBInfraredDataset(
        data_root=train_root,
        image_size=image_size,
        mode='train',
        **dataset_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Create validation dataset if specified
    val_loader = None
    if val_root is not None:
        val_dataset = RGBInfraredDataset(
            data_root=val_root,
            image_size=image_size,
            mode='val',
            **dataset_kwargs
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader


class RGBInfraredMaskDataset(RGBInfraredDataset):
    """
    Extended dataset that also includes inpainting masks for FLUX training.

    Expected directory structure:
        data_root/
            rgb/
            infrared/
            mask/  (optional, will generate random masks if not provided)
    """

    def __init__(
        self,
        data_root,
        rgb_dir='rgb',
        infrared_dir='infrared',
        mask_dir='mask',
        image_size=512,
        mode='train',
        random_mask_ratio=0.5,
        **kwargs
    ):
        """
        Args:
            random_mask_ratio: Probability of generating random mask if no mask file exists
        """
        super().__init__(
            data_root=data_root,
            rgb_dir=rgb_dir,
            infrared_dir=infrared_dir,
            image_size=image_size,
            mode=mode,
            **kwargs
        )

        self.random_mask_ratio = random_mask_ratio

        # Check if mask directory exists
        mask_path = os.path.join(data_root, mask_dir)
        if os.path.exists(mask_path):
            self.mask_files = sorted(glob.glob(os.path.join(mask_path, '*.*')))
            assert len(self.mask_files) == len(self.rgb_files), \
                f"Mismatch: {len(self.mask_files)} masks, {len(self.rgb_files)} RGB images"
        else:
            print(f"Mask directory not found at {mask_path}. Will generate random masks.")
            self.mask_files = None

        # Mask transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def _generate_random_mask(self, size):
        """Generate random rectangular mask"""
        mask = torch.ones(1, size, size)

        # Random rectangle
        h, w = size, size
        mask_h = torch.randint(h // 4, h // 2, (1,)).item()
        mask_w = torch.randint(w // 4, w // 2, (1,)).item()
        y = torch.randint(0, h - mask_h, (1,)).item()
        x = torch.randint(0, w - mask_w, (1,)).item()

        mask[:, y:y+mask_h, x:x+mask_w] = 0

        return mask

    def __getitem__(self, idx):
        # Get base data from parent class
        data = super().__getitem__(idx)

        # Load or generate mask
        if self.mask_files is not None:
            mask_img = Image.open(self.mask_files[idx]).convert('L')
            mask_tensor = self.mask_transform(mask_img)
            mask_tensor = (mask_tensor > 0.5).float()  # Binarize
        else:
            mask_tensor = self._generate_random_mask(self.image_size)

        data['mask'] = mask_tensor

        return data


if __name__ == '__main__':
    # Test dataset
    print("Testing RGBInfraredDataset...")

    # NOTE: Update this path to your actual data directory
    data_root = './data/train'

    if os.path.exists(data_root):
        dataset = RGBInfraredDataset(
            data_root=data_root,
            image_size=256,
            mode='train'
        )

        print(f"Dataset size: {len(dataset)}")

        # Test loading a sample
        sample = dataset[0]
        print(f"RGB shape: {sample['rgb'].shape}")
        print(f"Infrared shape: {sample['infrared'].shape}")
        print(f"Image ID: {sample['image_id']}")

        # Test dataloader
        train_loader, val_loader = create_dataloaders(
            train_root=data_root,
            batch_size=4,
            num_workers=0,
            image_size=256
        )

        batch = next(iter(train_loader))
        print(f"\nBatch RGB shape: {batch['rgb'].shape}")
        print(f"Batch Infrared shape: {batch['infrared'].shape}")

        print("\nDataset test successful!")
    else:
        print(f"Data directory not found at {data_root}")
        print("Please create the directory structure or update the path.")
