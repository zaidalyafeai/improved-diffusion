import torch
import torchvision.transforms.functional as TF
import random, math

def roll_minmax(low, high):
    roll = random.random()
    return roll*(high-low) + low

class RandomResizedProtectedCropLazy(torch.nn.Module):
    def __init__(self, size, min_area, max_area=1, interpolation=TF.InterpolationMode.BILINEAR):
        super().__init__()
        self.size = size
        self.min_area = min_area
        self.max_area = max_area
        self.interpolation = interpolation

    def get_params(self, img, safebox, return_n=True):
        width, height = TF.get_image_size(img)
        area = height * width

        left_s, top_s, right_s, bottom_s = safebox
        protected_space_h = right_s - left_s
        protected_space_v = bottom_s - top_s

        protected_edgesize = max(protected_space_h, protected_space_v)
        protected_area = (protected_edgesize) * (protected_edgesize)

        min_area = max(self.min_area, protected_area / area)
        max_area = self.max_area

        roll = random.random()

        target_area = area * roll_minmax(min_area, max_area)

        target_edgesize = math.sqrt(target_area)

        ok = False
        n = 0
        while not ok:
            doleft = random.random() < 0.5
            if doleft:
                cropbox_left = roll_minmax(0, left_s)
                cropbox_right = cropbox_left + target_edgesize
                ok_h = right_s < cropbox_right < width
            else:
                cropbox_right = roll_minmax(right_s, width)
                cropbox_left = cropbox_right - target_edgesize
                ok_h = 0 < cropbox_left < left_s

            dotop = random.random() < 0.5
            if dotop:
                cropbox_top = roll_minmax(0, top_s)
                cropbox_bottom = cropbox_top + target_edgesize
                ok_v = bottom_s < cropbox_bottom < height
            else:
                cropbox_bottom = roll_minmax(bottom_s, height)
                cropbox_top = cropbox_bottom - target_edgesize
                ok_v = 0 < cropbox_top < top_s

            ok = ok_h and ok_v
            n+=1

            if n > 10000:
                print('struggling w/ image, returning uncropped')
                print(f"safebox: {safebox}")
                print(f"target_edgesize: {target_edgesize}")
                print(f"protected_area: {protected_area}")
                cropbox_left, cropbox_top, cropbox_right, cropbox_bottom = (0, 0, width, height)
                break

        if return_n:
            return (cropbox_left, cropbox_top, cropbox_right, cropbox_bottom), n

        return (cropbox_left, cropbox_top, cropbox_right, cropbox_bottom)

    def forward(self, img, safebox):
        cropbox = self.get_params(img, safebox, return_n=False)
        i, j = cropbox[1], cropbox[0]
        h, w = cropbox[2] - j, cropbox[3] - i
        # display(img.crop(cropbox).resize((self.size, self.size)))
        return TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
