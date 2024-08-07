import os
import numpy as np
from PIL import Image
import random


def add_white_noise(image, box):
    """
    Add white noise to a specific region in the image to make it white.

    Args:
    image: PIL Image
    box: list, [x1, y1, x2, y2], region to cover with white noise

    Returns:
    Image with the specified region covered with white noise
    """
    image_array = np.array(image)
    x1, y1, x2, y2 = box
    noise = np.full((y2 - y1, x2 - x1, 3), 255, dtype=np.uint8)
    image_array[y1:y2, x1:x2] = noise
    return Image.fromarray(image_array)


def gen_samples(stop_value, img_dir, anno_landmarks_src, anno_src, save_dir):
    """
    Generate positive and negative samples by adding white noise to faces.

    Args:
    face_size: int, size of the face images (12, 24, 48) - not used here
    stop_value: int, total number of images to generate
    """
    sample_img_dir = os.path.join(save_dir, "image")

    if not os.path.exists(sample_img_dir):
        os.makedirs(sample_img_dir)

    anno_filename = os.path.join(save_dir, "bbox.txt")

    try:
        anno_file = open(anno_filename, 'w')

        positive_count = 0
        negative_count = 0

        with open(anno_landmarks_src) as f:
            landmarks_list = f.readlines()

        with open(anno_src) as f:
            anno_list = f.readlines()

        for i, (anno_line, landmarks) in enumerate(zip(anno_list, landmarks_list)):
            if i < 182639:
                continue

            landmarks = landmarks.split()
            strs = anno_line.split()
            img_name = strs[0].strip()
            img = Image.open(os.path.join(img_dir, img_name))

            x, y, w, h = float(strs[1].strip()), float(strs[2].strip()), float(strs[3].strip()), float(strs[4].strip())

            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            if max(w, h) < 40 or x1 < 0 or x2 > img.width or y1 < 0 or y2 > img.height:
                continue

            if positive_count < stop_value and negative_count < stop_value:
                image_id = "{:06d}.jpg".format(positive_count + negative_count)
                if random.random() < 0.5:
                    img.save(os.path.join(sample_img_dir, image_id))
                    anno_file.write(f"{image_id} 1 {int(x1)} {int(y1)} {int(w)} {int(h)}\n")
                    anno_file.flush()
                    positive_count += 1
                else:
                    noisy_img = add_white_noise(img, [x1, y1, x2, y2])
                    noisy_img.save(os.path.join(sample_img_dir, image_id))
                    anno_file.write(f"{image_id} 0 {int(x1)} {int(y1)} {int(w)} {int(h)}\n")
                    anno_file.flush()
                    negative_count += 1

            if positive_count + negative_count >= stop_value:
                break

    except Exception as ex:
        print(ex)
    finally:
        anno_file.close()


# Example usage
stop_value = 4000
save_dir = '../negative_data'
DATA_ROOT = '/Users/jhzhu/Downloads/software/pan.baidu/CelebA'
img_dir = os.path.join(DATA_ROOT, 'test')
anno_src = os.path.join(DATA_ROOT, 'Anno/list_bbox_celeba.txt')
anno_landmarks_src = os.path.join(DATA_ROOT, 'Anno/list_landmarks_celeba.txt')

gen_samples(stop_value, img_dir, anno_landmarks_src, anno_src, save_dir)
