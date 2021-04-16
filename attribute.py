import io
import hashlib
import numpy as np
import yaml
import mc
import json
import os.path as osp
from easydict import EasyDict as ED

try:
    import linklink as link
except:
    import spring.linklink as link

import torch
import torchvision.datasets
import torchvision.transforms

# from linklink_utils import *
from PIL import Image
import numpy as np
from petrel_client.client import Client

__all__ = ['Gender', 'Age', 'Liveness']

class ImageNetDataset:
    """
    ImageNet Dataset.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_type (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader (:obj:`str`): reader type 'pil' or 'ks'

    Metafile example::
        "n01440764/n01440764_10026.JPEG 0\n"
    """
    
    def __init__(self, cfg, root_dir, meta_file, eval=False, bucket_name=None, image_reader_type='pil'):
        
        self.root_dir = root_dir
        self.meta_file = meta_file
        self.eval = eval
        self.transform = self._build_transform()
        self.image_reader = pil_loader
        self.initialized = False
        self.bucket_name = bucket_name
        
        with open(meta_file) as f:
            lines = f.readlines()
        
        self.num = len(lines)
        self.metas = []
        for line in lines:
            filename, label = line.rstrip().split()
            self.metas.append({'filename': filename, 'label': label})
        
        # if self.bucket_name is None:
        #     self._init_memcached()
        # else:
        #     self._init_ceph()
        self._init_ceph()
        if 'global' in cfg['tasks']:
            self.cls_task_name = list(cfg['tasks']['global'].keys())[0]
            self.cls_task_dim = cfg['tasks']['global'][self.cls_task_name][1]

        elif 'event' in cfg['tasks']:
            self.cls_task_name = list(cfg['tasks']['event'].keys())[0]
            self.cls_task_dim = cfg['tasks']['event'][self.cls_task_name][1]

        else:
            raise NotImplementedError

    def __len__(self):
        return self.num
    
    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True
            
    def _init_ceph(self):
        if not self.initialized:
            conf_path = '~/petreloss.conf'
            self.mclient = Client(conf_path, mc_key_cb=trim_key)
            self.initialized = True
    
    def image_get(self, filename):
        # if self.bucket_name is None:
        #     filename = filename
        #     value = mc.pyvector()
        #     self.mclient.Get(filename, value)
        #     value = mc.ConvertBuffer(value)
        #     filebytes = np.frombuffer(value.tobytes(), dtype=np.uint8)
        # else:
        #     filename = 's3://'+self.bucket_name + filename
        #     filebytes = self.mclient.Get(filename)
        
        if self.bucket_name is None:
            filename = filename
            filebytes = self.mclient.Get(filename)
        else:
            filename = 's3://'+self.bucket_name + filename
            filebytes = self.mclient.Get(filename)
        
        img = self.image_reader(filebytes, filename)
        return img
    
    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = osp.join(self.root_dir, curr_meta['filename'])
        label = int(curr_meta['label'])
        try:
            img = self.image_get(filename)
            
            if self.transform is not None:
                img = self.transform(img)
            
            item = {
                'image': img,
                'gt': {'global': {
                    self.cls_task_name: label
                }},
                'image_id': idx,
                'filename': filename
            }
            return item
        except Exception as e:
            print(e)
            print("[ERROR] Image loading failed! Location: {}".format(filename))
            return self[idx - 1]
    
    def _build_transform(self):
        if self.eval:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform


class AttributeGenderDataset(ImageNetDataset):
    
    def __init__(self, cfg, root_dir, meta_file, eval=False, bucket_name=None, image_reader_type='pil', attr_name="gender"):
    
        self.root_dir = root_dir
        self.meta_file = meta_file
        self.eval = eval
        self.transform = self._build_transform()
        self.image_reader = pil_loader
        self.initialized = False
        self.bucket_name = bucket_name

        self.metas = []
        with open(meta_file) as f:
            for line in f.readlines():
                self.metas.append(json.loads(line))
        self.num = len(self.metas)

        self._init_ceph()
        
        self.attr_name = attr_name
        self.cls_task_name = list(cfg['tasks']['global'].keys())[0]
        self.cls_task_dim = cfg['tasks']['global'][self.cls_task_name][1]

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = self.root_dir + curr_meta['image_path']
        try:
            # add root_dir to filename
            img = self.image_get(filename)
    
            label = list(curr_meta['attribute'][self.attr_name].values())
            label = np.argmax(label)
            
            if self.transform is not None:
                img = self.transform(img)
            
            item = {
                'image': img,
                'gt': {'global': {
                    self.cls_task_name: label
                }},
                'image_id': idx,
                'filename': filename
            }
            return item
        except Exception as e:
            print(e)
            print("[ERROR] Image loading failed! Location: {}".format(filename))
            return self[idx - 1]


class AttributeAgeDataset(AttributeGenderDataset):
    def __init__(self, cfg, root_dir, meta_file, eval=False, bucket_name=None, image_reader_type='pil', attr_name="age"):
        super(AttributeAgeDataset, self).__init__(cfg, root_dir,
                                                  meta_file, eval,
                                                  bucket_name, image_reader_type,
                                                  attr_name)


def pil_loader(img_bytes, filepath):
    buff = io.BytesIO(img_bytes)
    with Image.open(buff) as img:
        img = img.convert('RGB')
    return img


def all_gather(data, group_id, task_size):
    gather_data = []
    for _ in range(task_size):
        gather_data.append(torch.zeros_like(data))
    link.gather(gather_data, data, 0, group_idx=group_id)
    return gather_data


def trim_key(key):
    key = key[9:]
    if len(key) >= 250:
        key = str(key).encode('utf-8')
        m = hashlib.md5()
        m.update(key)
        return "md5://{}".format(m.hexdigest())
    else:
        return key


def Gender():
    cfg = yaml.safe_load("""
tasks:
    global:
        gender_attri_cls: ['CSELoss', 2]
kwargs:
    root_dir: /attribute/
    meta_file: /mnt/lustre/heyinan/gvm_list/gender_train_major.json
    bucket_name: gvm
val_kwargs:
    root_dir: /attribute/
    meta_file: /mnt/lustre/heyinan/gvm_list/gender_test.json
    bucket_name: gvm
    eval: True
batch_size:
    train: 32
    val: 64
task_scheduler:
    weight: 1
    warmup_weight: 1
    warmup_steps: 0
    x: [0]
    y: [1]
""")
    return AttributeGenderDataset(
        cfg=ED(cfg),
        root_dir='/attribute/',
        meta_file='/mnt/lustre/heyinan/gvm_list/gender_test.json',
        bucket_name='gvm',
        eval=True,
    )


def Age():
    cfg = yaml.safe_load("""
tasks:
    global:
        age_attri_cls: ['CSELoss', 3]
kwargs:
    root_dir: /attribute/
    meta_file:  /mnt/lustre/heyinan/gvm_list/age_train_major.json
    bucket_name: gvm
val_kwargs:
    root_dir: /attribute/
    meta_file:  /mnt/lustre/heyinan/gvm_list/age_test.json
    bucket_name: gvm
    eval: True
batch_size:
    train: 32
    val: 64
task_scheduler:
    weight: 1
    warmup_weight: 1
    warmup_steps: 0
    x: [0]
    y: [1]
""")
    return AttributeAgeDataset(
        cfg=ED(cfg),
        root_dir='/attribute/',
        meta_file='/mnt/lustre/heyinan/gvm_list/gender_test.json',
        bucket_name='gvm',
        eval=True,
    )


def Liveness():
    cfg = yaml.safe_load("""
tasks:
    global:
        liveness_cls: ['CSELoss', 2]
kwargs:
    root_dir: /
    meta_file: /mnt/lustre/heyinan/gvm_list/liveness_4M.train
    bucket_name: gvm
val_kwargs:
    root_dir: /
    meta_file: /mnt/lustre/heyinan/gvm_list/liveness.test
    bucket_name: gvm
    eval: True
batch_size:
    train: 32
    val: 64
task_scheduler:
    weight: 1
    warmup_weight: 1
    warmup_steps: 0
    x: [0]
    y: [1]
""")
    return ImageNetDataset(
        cfg=ED(cfg),
        root_dir='/attribute/',
        meta_file='/mnt/lustre/heyinan/gvm_list/gender_test.json',
        bucket_name='gvm',
        eval=True,
    )


