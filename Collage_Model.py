!pip install torch
!pip install torchvision 
!pip install ftfy regex tqdm
!pip install git+https://github.com/openai/CLIP.git

import os
import cv2
import clip
import math
import copy
import torch
from PIL import Image
from typing import Union
from torchvision.datasets import CIFAR100
import numpy as np
from matplotlib import pyplot as plt

# The description of the target image, e.g. a duck in a lake
TEXT = ['a duck in a lake']
# The path of background components
BACKGROUND_PATH = ['./lake.jpg', './sky.jpg', './table.jpeg']
# The path of Characters components
CHARACTER_PATH = ['./yellow_duck.png', './monkey.png']
background_list = []
character_list = []
blending_list = []
RANDOM_TIMES = 1500

class IMG_VAR:
    def __init__(self, img:Union[str, np.ndarray]) -> object:
        """
        Create an image object for many variations.
            img: an image file path or a numpy array
        """
        if type(img) != str:
            self.img = img
        else:
            self.img = cv2.imread(img)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
    
    def show_img(self):
        """
        Show the object image.
        """
        plt.imshow(self.img)
    
    def resize(self, scale:float):
        """
        Reszie the image.
            scale: the scale for resize
        """
        self.img = cv2.resize(self.img, (int(self.width * scale), int(self.height * scale)))
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
    
    def rotate(self, angle:int, scale:float=1.0):
        """
        Rotate the image.
            angle: the angle for rotate, the value must between 0 and 360
            scale: the scale for resize, default is 1.0
        """
        center = (self.width / 2, self.height / 2) # the center of the graph
        new_width = int((abs(math.cos(math.radians(angle))) * self.width + abs(math.sin(math.radians(angle))) * self.height) * scale) # cosθ * width + sinθ * height
        new_height = int((abs(math.cos(math.radians(angle))) * self.height + abs(math.sin(math.radians(angle))) * self.width) * scale) # sinθ * width + cosθ * height
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=scale)
        rotate_matrix +=  [[0, 0, (new_width - self.width) / 2], [0, 0, (new_height - self.height) / 2]] # translation
        self.img = cv2.warpAffine(src=self.img, M=rotate_matrix, dsize=(new_width, new_height))
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
    
    def crop(self, width_bg:int, width_ed:int, height_bg:int, height_ed:int):
        """
        Crop the image.
            width_st: the beginning position of the width
            width_ed: the end position of the width
            height_st: the beginning position of the height
            height_ed: the end position of the height
        """
        self.img = self.img[height_bg:height_ed, width_bg:width_ed]
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
    
    def flip(self):
        """
        Flip the image.
        """
        self.img = cv2.flip(self.img, 0)

def load_image(path:str) -> torch.Tensor:
    """
    Show and load the image as torch tensor.
        path: the path of the image
    """
    Image.open(path).show()
    image = preprocess(Image.open(path)).unsqueeze(0).to(device)
    return image

def create_white_img(height:int, width:int) -> np.ndarray:
    """
    Create a white image.
        height: the height of the image
        width: the width of the image
    """
    white_img = np.zeros([height, width, 3], dtype=np.uint8)
    white_img.fill(255)
    return white_img

def put_text(img, text, position=(0, 0), scale=1, color=(0, 0, 0), thick=3, font:str=None):
    """
    Put the text to the image.
        img: the target image
        text: the text would be put to the image
        position: the bottom-left corner position of the text in the image
        scale: text scale 
        color: BGR
        thick: the thickness of the text
    """
    if not font:
        font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, position, font, scale, color, thick, cv2.LINE_AA)

def random_midpoint(img1:np.ndarray, img2:np.ndarray, specific:str=None, random_num:int=1) -> np.ndarray:
    """
    Create a random mid point according to the size of img1 and img2.\ 
    The mid point is for anchoring the img2 to the img1.
        img1: larger image
        img2: smaller image
        specific: top_left, bottom_left, top_right, bottom_right
        random_num: the number of random mid point
    """
    img1_rows, img1_cols = img1.shape[0], img1.shape[1] # large image
    img2_rows, img2_cols = img2.shape[0], img2.shape[1] # small image
    min_row, max_row = img2_rows//2, img1_rows + img2_rows//2 - img2_rows
    min_col, max_col = img2_cols//2, img1_cols + img2_cols//2 - img2_cols
    specific_position = {'top_left':[min_row, min_col], 'bottom_left':[max_row, min_col], 'top_right':[min_row, max_col], 'bottom_right':[max_row, max_col]}
    if specific:
        return [specific_position[specific]]
    random_midpoint = np.random.randint([min_row, min_col], [max_row, max_col], size= [random_num, 2])
    return random_midpoint

def blending_img(img1:np.ndarray, img2:np.ndarray, middle_point:Union[list, np.ndarray]) -> np.ndarray:
    """
    Blend the two images.\ 
    Anchor the img2 to the img1.
        img1: larger image
        img2: smaller image
        middle_point: the position where the img2 will be anchored
    """
    rows, cols = img2.shape[0], img2.shape[1]
    roi = img1[middle_point[0] - rows//2: middle_point[0] - rows//2 + rows  , middle_point[1] - cols//2: middle_point[1] - cols//2 + cols]
    
    img2gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

    mask_inverse = cv2.bitwise_not(mask)
    img1_background = cv2.bitwise_and(roi, roi, mask=mask_inverse)
    img2_figure = cv2.bitwise_and(img2, img2, mask=mask)
    tmp_blending_img = cv2.add(img1_background, img2_figure)
    blending_img = img1.copy()
    blending_img[middle_point[0] - rows//2: middle_point[0] - rows//2 + rows  , middle_point[1] - cols//2: middle_point[1] - cols//2 + cols] = tmp_blending_img
    return blending_img

def create_variation(images:IMG_VAR, scale:float, angle:int, flip:bool):
    """
    Variation the IMG_VAR object, include: resizing, rotatation, flipping.\ 
    The variation result will be appended to the variation attribute of the object.
        images: IMG_VAR object
        scale: the scale for resizing
        angle: the angle for rotation
        flip: True or False, 1 or 0
    """
    tmp = copy.deepcopy(images)
    tmp.resize(scale)
    tmp.rotate(angle)
    if flip:
        tmp.flip()
    images.variation = tmp.img

def collage_image(background_list:list, character_list:list, background_size:list=[600, 800], var:bool=False, resize_degree:float=1.0, random:bool=False, random_size:list=[1], *args, **kwargs):
    """
    Collage images. The components will be chosen from the background_list and the character_list.
        background_list: background list
        character_list: character list
        background_size: if no background_list, it will create a white background according to the background_size, default is 600 x 800
        var: vary the characters or not
        resize_degree: if var is True, characters will be resized between 0.1 and resize_degree scale
        random: random pick the background and characters or not (background will only pick once, the characters will base on random_size)
        random_size: it is a list value. If the list contains only one value, it will pick the specific numbers, otherwise the picked numbers will randomly decide among the random_size 
    """
    # create background
    if not BACKGROUND_PATH:
        background = create_white_img(background_size[0], background_size[1])
        background_list.append(IMG_VAR(background))
    else:
        for b_path in BACKGROUND_PATH:
            background_list.append(IMG_VAR(b_path))
    # create characters
    for ch_path in CHARACTER_PATH:
        character_list.append(IMG_VAR(ch_path))
    # create variation
    if var:
        for images in character_list:
            variation = []
            images.size_list = [math.ceil(i * 10 * resize_degree) / 10  for i in np.random.random(RANDOM_TIMES)] # ceil the number one decimal place, so the minimum will be 0.1
            images.angle_list = np.random.randint(0, 360, RANDOM_TIMES)
            images.flip_list = np.random.randint(0, 2, RANDOM_TIMES)
            for i in range(RANDOM_TIMES):
                tmp = copy.deepcopy(images)
                tmp.resize(images.size_list[i])
                tmp.rotate(images.angle_list[i])
                if images.flip_list[i]:
                    tmp.flip()
                variation.append(tmp.img)
            images.variation = variation
    # build collage images
    for images in character_list:
        if not var and not random:      
            images.mid_point = random_midpoint(background_list[0].img, images.img, random_num=RANDOM_TIMES)
        else:
            images.mid_point = np.zeros([RANDOM_TIMES, 2], dtype=int)
    for i in range(RANDOM_TIMES):
        if random:
            tmp_blending = np.random.choice(background_list, size=1)[0].img.copy()
            characters = np.random.choice(character_list, size=np.random.choice(random_size), replace=False)
        else:
            tmp_blending = background_list[0].img.copy()
            characters = character_list
        for images in characters:
            if not var and not random:
                tmp_blending = blending_img(tmp_blending, images.img, images.mid_point[i])
            elif var:
                images.mid_point[i] = random_midpoint(tmp_blending, images.variation[i], random_num=1)
                tmp_blending = blending_img(tmp_blending, images.variation[i], images.mid_point[i])
            else:
                images.mid_point[i] = random_midpoint(tmp_blending, images.img, random_num=1)
                tmp_blending = blending_img(tmp_blending, images.img, images.mid_point[i])
        blending_list.append(tmp_blending)

def show_result(n:int):
    """
    Show the n index blending result.
    """
    for text, prob in zip(TEXT, probs[n]):
        print(f'{text}: {round(prob * 100, 2)}')
    plt.imshow(blending_list[n])

# method 1
def predict_category_1(image_input, text_input, top_n=5):
    with torch.no_grad():
        text_features = model.encode_text(text_input) # convert the texts to text representation vectors (n, 512)
        image_features = model.encode_image(image_input) # convert the texts to image representation vectors (n, 512)
    text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True) # norm is L2 norm  
    image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True) # norm is L2 norm
    similarity = (100 * image_features_norm @ text_features_norm.T).softmax(dim=-1) # equal to 'torch.matmul(image_features_norm, text_features_norm.T)'
    values, indices = similarity[0].topk(top_n)

    print("\nTop 5 Categories:")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>13s}: {100 * value.item():.2f}%")

# method 2
def predict_category_2(image_input, text_input, top_n=5):
    result = model(image_input, text_input)
    values, indices = result[0][0].softmax(dim=-1).topk(top_n)
    print("\nTop 5 Categories:")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>13s}: {100 * value.item():.2f}%")

def text_setting(additional_text):
    # add categories
    cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    cifar100.classes.extend(additional_text)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    return text_inputs, cifar100

if __name__ == "__main__":
    # create collage image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    collage_image(background_list, character_list, random=True)

    # cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
    # text_inputs, cifar100 = text_setting(['printer', 'projector', 'mickey', 'robotic arm'])
    # image_input = load_image('./drive/MyDrive/Colab Notebooks/test.png')
    # predict_category_2(image_input, text_inputs)

    image = torch.cat([preprocess(Image.fromarray(blending_list[i])).unsqueeze(0).to(device) for i in range(RANDOM_TIMES)])
    text = clip.tokenize(TEXT).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        logits_per_image, logits_per_text = model(image, text)  
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    for txt, img in zip(TEXT, np.argmax(probs, axis =0)):
        print(f'Maximum Probability of {txt} is {img} image')
    show_result()

    # img = blending_list[1183]
    # logo = IMG_VAR('./epson_logo.png')
    # logo.resize(0.5)
    # mid = random_midpoint(img, logo.img, specific='bottom_left')
    # img2 = blending_img(img, logo.img, mid[0])