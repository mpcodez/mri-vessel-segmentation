import numpy as np
import cv2
from keras_preprocessing import image


def random_flip(img, mask, probability=0.5):
    """Randomly flips image and mask horizontally and/or vertically."""
    if np.random.random() < probability:
        img = image.flip_axis(img, 1)
        mask = image.flip_axis(mask, 1)
    if np.random.random() < probability:
        img = image.flip_axis(img, 0)
        mask = image.flip_axis(mask, 0)
    return img, mask


def random_rotate(img, mask, rotate_limit=(-20, 20), probability=0.5):
    """Randomly rotates image and mask within the given limits."""
    if np.random.random() < probability:
        theta = np.random.uniform(*rotate_limit)
        img = image.apply_affine_transform(img, theta=theta)
        mask = image.apply_affine_transform(mask, theta=theta)
    return img, mask


def random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), probability=0.5):
    """Randomly shifts image and mask within the given limits."""
    if np.random.random() < probability:
        wshift = np.random.uniform(*w_limit)
        hshift = np.random.uniform(*h_limit)
        img = image.apply_affine_transform(img, tx=wshift * img.shape[1], ty=hshift * img.shape[0])
        mask = image.apply_affine_transform(mask, tx=wshift * mask.shape[1], ty=hshift * mask.shape[0])
    return img, mask


def random_zoom(img, mask, zoom_range=(0.8, 1.2), probability=0.5):
    """Randomly zooms into or out of image and mask."""
    if np.random.random() < probability:
        zx, zy = np.random.uniform(*zoom_range, 2)
        img = image.apply_affine_transform(img, zx=zx, zy=zy)
        mask = image.apply_affine_transform(mask, zx=zx, zy=zy)
    return img, mask


def random_shear(img, mask, intensity_range=(-0.5, 0.5), probability=0.5):
    """Randomly shears image and mask."""
    if np.random.random() < probability:
        shear = np.random.uniform(*intensity_range)
        img = image.apply_affine_transform(img, shear=shear)
        mask = image.apply_affine_transform(mask, shear=shear)
    return img, mask


def random_gray(img, probability=0.5):
    """Randomly converts image to grayscale."""
    if np.random.random() < probability:
        gray = np.dot(img[..., :3], [0.114, 0.587, 0.299])
        img = np.stack([gray] * 3, axis=-1)
    return img


def random_contrast(img, limit=(-0.3, 0.3), probability=0.5):
    """Randomly adjusts contrast of image."""
    if np.random.random() < probability:
        alpha = 1.0 + np.random.uniform(*limit)
        mean_intensity = np.mean(img, axis=(0, 1), keepdims=True)
        img = alpha * (img - mean_intensity) + mean_intensity
        img = np.clip(img, 0, 1)
    return img


def random_brightness(img, limit=(-0.3, 0.3), probability=0.5):
    """Randomly adjusts brightness of image."""
    if np.random.random() < probability:
        alpha = 1.0 + np.random.uniform(*limit)
        img = np.clip(alpha * img, 0, 1)
    return img


def random_saturation(img, limit=(-0.3, 0.3), probability=0.5):
    """Randomly adjusts saturation of image."""
    if np.random.random() < probability:
        alpha = 1.0 + np.random.uniform(*limit)
        gray = np.dot(img[..., :3], [0.114, 0.587, 0.299])[:, :, np.newaxis]
        img = np.clip(alpha * img + (1 - alpha) * gray, 0, 1)
    return img


def random_crop(img, mask, probability=0.1):
    """Randomly crops and resizes image and mask back to original size."""
    if np.random.random() < probability:
        h, w = img.shape[:2]
        crop_x = np.random.randint(0, w // 2)
        crop_y = np.random.randint(0, h // 2)
        crop_w = np.random.randint(w // 2, w)
        crop_h = np.random.randint(h // 2, h)

        img = cv2.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h), interpolation=cv2.INTER_LINEAR)
        mask = mask[..., np.newaxis]
    return img, mask


def random_augmentation(img, mask):
    """Applies a series of random augmentations to image and mask."""
    img = random_brightness(img, limit=(-0.1, 0.1), probability=0.05)
    img = random_contrast(img, limit=(-0.1, 0.1), probability=0.05)
    img = random_saturation(img, limit=(-0.1, 0.1), probability=0.05)
    img, mask = random_rotate(img, mask, rotate_limit=(-10, 10), probability=0.05)
    img, mask = random_shear(img, mask, intensity_range=(-5, 5), probability=0.05)
    img, mask = random_flip(img, mask, probability=0.5)
    img, mask = random_shift(img, mask, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1), probability=0.05)
    img, mask = random_zoom(img, mask, zoom_range=(0.9, 1.1), probability=0.05)
    return img, mask
