from PIL import Image
import numpy as np
from Crypto.Util.number import *
from tqdm import tqdm
import cv2

n = 5
r = 3
path = "test2.png"

def read_image(path):
    img = Image.open(path)
    img_array = np.asarray(img)
    # print(img_array.shape)
    
    return img_array.flatten(), img_array.shape


def polynomial(img, n, r):
    num_pixels = img.shape[0]
    coef = np.random.randint(low = 0, high = 257, size = (num_pixels, r - 1))
    gen_imgs = []
    for i in range(1, n + 1):
        base = np.array([i ** j for j in range(1, r)])
        base = np.matmul(coef, base) 
        img_ = img + base
        img_ = img_ % 257
        gen_imgs.append(img_)

    return np.array(gen_imgs)

def lagrange(x, y, num_points, x_test):
    l = np.zeros(shape=(num_points, ))
    r = np.zeros(shape=(num_points, ))

    for k in range(num_points):
        l[k] = 1
        r[k] = 1
        for k_ in range(num_points):
            if k != k_:
                l[k] = l[k]*(x_test-x[k_]) % 257
                r[k] = r[k]*(x[k]-x[k_]) % 257
            else:
                pass
        
        r_ = inverse(int(r[k]), 257) # 采用模逆元运算
        l[k] = l[k] * r_ % 257
    L = 0
    for i in range(num_points):
        L += y[i]*l[i]
    return L

def decode(imgs, index, r, n):
    assert imgs.shape[0] >= r
    x = np.array(index)
    dim = imgs.shape[1]
    # print("dimension: {}".format(dim))
    img = []
    for i in tqdm(range(dim)):
        y = imgs[:, i]
        pixel = lagrange(x, y, r, 0) % 257
        img.append(pixel)
    return np.array(img)
    
if __name__ == "__main__":
    img_flattened, shape = read_image(path)
    gen_imgs = polynomial(img_flattened, n = n, r = r)
    to_save = gen_imgs.reshape(n, *shape)
    for i, img in enumerate(to_save):
        img = img.astype(np.uint16)
        cv2.imwrite("test2_{}.png".format(i + 1), img.astype(np.uint16)) 
        # Image.fromarray(img.astype(np.uint8)).save("test2_{}.png".format(i + 1)) #原方法


    index = [1, 3, 5]
    images = []
    for i in index:
        img = cv2.imread('test2_{}.png'.format(i), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR) # 读取3通道16位图片
        # img = Image.open('test2_{}.png'.format(i)) #原方法
        img_array = np.asarray(img)
        read_img = img_array.flatten()
        images.append(read_img)
    images = np.array(images)

    origin_img = decode(images, index, r = r, n = n)
    # print(origin_img - img_flattened)
    origin_img = origin_img.reshape(*shape)

    try:
        Image.fromarray(origin_img.astype(np.uint8)).save("test2_origin.png")
    except OSError:
        print("Image dimension too large!")