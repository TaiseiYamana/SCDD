from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits

class ImageCLEF(ImageList):
    image_list = {
        "C": "image_list/c.txt",
        "I": "image_list/i.txt",
        "P": "image_list/p.txt",
        "B": "image_list/b.txt"
    }
    CLASSES = ['aeroplane', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car', 'dog', 'horse', 'monitor', 'motorbike', 'people']

    def __init__(self, root: str, task: str, indexs: Optional[Callable] = None, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(ImageCLEF, self).__init__(root, ImageCLEF.CLASSES, data_list_file=data_list_file, **kwargs)
    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())