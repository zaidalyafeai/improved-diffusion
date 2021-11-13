from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import tokenizers


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False,
    txt=False,
    tokenizer_path  = "tokenizer_file",
    max_seq_len     = 130,
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
    all_files, image_file_to_text_file = _list_image_files_recursively(data_dir, txt=txt)
    print(f"found {len(all_files)} images, {len(image_file_to_text_file)} texts")

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
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir, txt=False):
    results = []
    image_file_to_text_file = {}
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
            if txt:
                prefix, _, ext = entry.rpartition(".")
                path_txt = prefix + ".txt"
                if bf.exists(path_txt):
                    image_file_to_text_file[full_path] = path_txt
        elif bf.isdir(full_path):
            next_results, next_map = _list_image_files_recursively(full_path)
            results.extend(next_results)
            image_file_to_text_file.update(next_map)
    return results, image_file_to_text_file


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths,
                 classes=None,
                 image_file_to_text_file=None,
                 txt=False,
                 tokenizer_path="tokenizer_file",
                 max_seq_len=130,
                 shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.txt = txt
        self.max_seq_len = max_seq_len

        if self.txt:
            self.local_images = [p for p in self.local_images if p in image_file_to_text_file]
            self.local_texts = [image_file_to_text_file[p] for p in self.local_images]

            self.tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
            self.tokenizer.enable_truncation(max_seq_len)
            self.tokenizer.enable_padding()

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

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        if self.txt:
            path_txt = self.local_texts[idx]
            with bf.BlobFile(path_txt, "r") as f:
                text = f.read()
            out_dict['txt'] = text
        return np.transpose(arr, [2, 0, 1]), out_dict

    def tokenize(self, txt):
        return [t.ids for t in self.tokenizer.encode_batch(txt)]
