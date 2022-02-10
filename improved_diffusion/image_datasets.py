import string, os, random
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as T

import tokenizers


def make_char_level_tokenizer():
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<unk>"))
    trainer = tokenizers.trainers.BpeTrainer(special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"])
    tokenizer.train_from_iterator([[c] for c in string.printable], trainer)
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        "<s> $0 </s>", special_tokens=[("<s>", 0), ("</s>", 1)]
    )
    return tokenizer


def load_tokenizer(tokenizer_path  = "tokenizer_file", max_seq_len=64, char_level=False):
    if char_level:
        tokenizer = make_char_level_tokenizer()
    else:
        tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(max_seq_len)
    tokenizer.enable_padding()
    return tokenizer


def tokenize(tokenizer, txt):
    return [t.ids for t in tokenizer.encode_batch(txt)]


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,
    txt=False, monochrome=False, offset=0, min_filesize=0,
    txt_pdrop=0., txt_drop_string='<mask><mask><mask><mask>',
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files, image_file_to_text_file, file_sizes = _list_image_files_recursively(data_dir, txt=txt, min_filesize=min_filesize)
    print(f"found {len(all_files)} images, {len(image_file_to_text_file)} texts")
    all_files = all_files[offset:]

    n_texts = sum(1 for k in file_sizes.keys() if k.endswith('.txt'))  # sanity check
    n_nonempty_texts = sum(file_sizes[k] > 0 for k in file_sizes.keys() if k.endswith('.txt'))
    n_empty_texts = n_texts - n_nonempty_texts

    if n_texts > 0:
        frac_empty = n_empty_texts/n_texts
        frac_nonempty = n_nonempty_texts/n_texts

        print(f"of {n_texts} texts, {n_empty_texts} ({frac_empty:.1%}) are empty, {n_nonempty_texts} ({frac_nonempty:.1%}) are nonempty")

    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for vpath in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        image_file_to_text_file=image_file_to_text_file,
        txt=txt,
        monochrome=monochrome,
        file_sizes=file_sizes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        txt_pdrop=txt_pdrop,
        txt_drop_string=txt_drop_string,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True,
        )
    while True:
        yield from loader


def load_superres_data(data_dir, batch_size, large_size, small_size, class_cond=False, txt=False, monochrome=False,
                       deterministic=False, offset=0, colorize=False,
                       blur_prob=0., blur_sigma_min=0.4, blur_sigma_max=0.6,
                       min_filesize=0,
                       txt_pdrop=0., txt_drop_string='<mask><mask><mask><mask>'
                       ):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=class_cond,
        txt=txt,
        monochrome=monochrome,
        deterministic=deterministic,
        offset=offset,
        min_filesize=min_filesize,
        txt_pdrop=txt_pdrop,
        txt_drop_string=txt_drop_string,
    )
    for large_batch, model_kwargs in data:
        blurrer = T.RandomApply(transforms=[T.GaussianBlur(5, sigma=(blur_sigma_min, blur_sigma_max))], p=blur_prob)

        model_kwargs["low_res"] = F.interpolate(large_batch, small_size, mode="area")
        if colorize:
            model_kwargs["low_res"] = model_kwargs["low_res"].mean(dim=1, keepdim=True)
        if blur_prob > 0:
            model_kwargs["low_res"] = blurrer(model_kwargs["low_res"])

        yield large_batch, model_kwargs


def _list_image_files_recursively(data_dir, txt=False, min_filesize=0):
    results = []
    image_file_to_text_file = {}
    file_sizes = {}
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            if min_filesize > 0:
                filesize = os.path.getsize(full_path)
                if filesize < min_filesize:
                    continue
                file_sizes[full_path] = filesize
            results.append(full_path)
            if txt:
                prefix, _, ext = full_path.rpartition(".")
                path_txt = prefix + ".txt"
                # print(f'made path_txt={repr(path_txt)} from {repr(entry)}')
                if bf.exists(path_txt):
                    image_file_to_text_file[full_path] = path_txt
                    filesize = os.path.getsize(path_txt)
                    file_sizes[path_txt] = filesize
                else:
                    pass
                    # raise ValueError(path_txt)
        elif bf.isdir(full_path):
            next_results, next_map, next_file_sizes = _list_image_files_recursively(full_path, txt=txt)
            results.extend(next_results)
            image_file_to_text_file.update(next_map)
            file_sizes.update(next_file_sizes)
    return results, image_file_to_text_file, file_sizes


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths,
                 classes=None,
                 image_file_to_text_file=None,
                 txt=False,
                 monochrome=False,
                 file_sizes=None,
                 shard=0, num_shards=1,
                 txt_pdrop=0.,
                 txt_drop_string='<mask><mask><mask><mask>',
                 empty_string_to_drop_string=False,  # unconditional != no text
                 ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.txt = txt
        self.monochrome = monochrome
        self.file_sizes = file_sizes
        self.txt_pdrop = txt_pdrop
        self.txt_drop_string = txt_drop_string
        self.empty_string_to_drop_string = empty_string_to_drop_string

        if self.txt:
            self.local_images = [p for p in self.local_images if p in image_file_to_text_file]
            self.local_texts = [image_file_to_text_file[p] for p in self.local_images]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        mode = "L" if self.monochrome else "RGB"
        arr = np.array(pil_image.convert(mode))
        if self.monochrome:
            arr = np.expand_dims(arr, 2)
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.txt:
            path_txt = self.local_texts[idx]
            if (self.txt_pdrop > 0) and (random.random() < self.txt_pdrop):
                text = self.txt_drop_string
            else:
                with bf.BlobFile(path_txt, "r") as f:
                    text = f.read()
            if (len(text) == 0) and self.empty_string_to_drop_string:
                text = self.txt_drop_string
            out_dict['txt'] = text
        return np.transpose(arr, [2, 0, 1]), out_dict
