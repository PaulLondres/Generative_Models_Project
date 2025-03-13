import numpy as np
import scipy.linalg
import scipy.signal
# import cv2
import time
import tqdm
import os
import torch
from PIL import Image

#Anisotropic Deblurring
class Deblurring2D():
    def mat_by_img(self, M, v):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim)).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, self.img_dim,
                        self.img_dim), M).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel1, kernel2, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        #build 1D conv matrix - kernel1
        H_small1 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel1.shape[0]//2, i + kernel1.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small1[i, j] = kernel1[j - i + kernel1.shape[0]//2]
        #build 1D conv matrix - kernel2
        H_small2 = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel2.shape[0]//2, i + kernel2.shape[0]//2):
                if j < 0 or j >= img_dim: continue
                H_small2[i, j] = kernel2[j - i + kernel2.shape[0]//2]
        #get the svd of the 1D conv
        self.U_small1, self.singulars_small1, self.V_small1 = torch.svd(H_small1, some=False)
        self.U_small2, self.singulars_small2, self.V_small2 = torch.svd(H_small2, some=False)
        ZERO = 3e-2
        self.singulars_small1[self.singulars_small1 < ZERO] = 0
        self.singulars_small2[self.singulars_small2 < ZERO] = 0
        #calculate the singular values of the big matrix
        self._singulars = torch.matmul(self.singulars_small1.reshape(img_dim, 1), self.singulars_small2.reshape(1, img_dim)).reshape(img_dim**2)
        #sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(descending=True) #, stable=True)
        # Save matrices for direct application
        self.H1 = H_small1
        self.H2 = H_small2

    def apply_H(self, image):
        """Applies the blurring matrix H to an image."""
        image_vector = image.reshape(image.shape[0] * self.channels, self.img_dim, self.img_dim)
        blurred = torch.matmul(self.H1, image_vector)  # Apply horizontal blur
        blurred = torch.matmul(blurred, self.H2)  # Apply vertical blur
        return blurred.reshape(image.shape)

    def V(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small1, temp)
        out = self.img_by_mat(out, self.V_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Vt(self, vec):
        #multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        #invert the permutation
        temp = torch.zeros(vec.shape[0], self.img_dim**2, self.channels, device=vec.device)
        temp[:, self._perm, :] = vec.clone().reshape(vec.shape[0], self.img_dim**2, self.channels)
        temp = temp.permute(0, 2, 1)
        #multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small1, temp)
        out = self.img_by_mat(out, self.U_small2.transpose(0, 1)).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        #multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small1.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small2).reshape(vec.shape[0], self.channels, -1)
        #permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)
    

def read_image_pillow(file_path):
    """Reads an image in grayscale using Pillow."""
    image = Image.open(file_path).convert("L")  # "L" mode converts to grayscale
    return np.array(image)

def convert_to_uint8(image_float):
    """Converts a floating-point image (0-255) to uint8 (0-255)."""
    image_clipped = np.clip(image_float, 0, 255)  # Ensure values are within [0,255]
    image_uint8 = np.round(image_clipped).astype(np.uint8)  # Round and convert
    return image_uint8

def save_image_pillow(image_array, save_path):
    """Saves a NumPy array as a grayscale image using Pillow."""
    image = Image.fromarray(image_array)  # Convert NumPy array to PIL image
    image.save(save_path)



# if main file
if __name__ == "__main__":
    image_size = 256
    device = "cuda"
    sigma = 2       # Adjust for medical relevance
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
    kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
    pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
    kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(device)
    deblur = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), 1, 256, device)

    # downsample images with torch
    downsample = torch.nn.AvgPool2d(4)
    # open images
    files = os.listdir("images")

    for file in tqdm.tqdm(files[:10000]):
        image = read_image_pillow(f"images/{file}")
        image = torch.Tensor(image).to("cuda").unsqueeze(0)
        downsampled_image = downsample(image)
        save_image_pillow(downsampled_image.squeeze(0).cpu().numpy().astype(np.uint8), f"downsampled_images/{file}")
        blurred_image = deblur.apply_H(downsampled_image)
        save_image_pillow(convert_to_uint8(blurred_image.squeeze(0).cpu().numpy()), f"blurred_images/{file}")

