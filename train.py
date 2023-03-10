import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
# import torchvision
from torchvision import datasets
from torchvision.models import resnet50
from torchvision.transforms import ToTensor
import random
import torchvision.transforms.functional as TF

print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.__version__)

TRANSFORMS_CONFIG={"jitter":False, "erase":False, "rotation":True, "hflip":True, "vflip":True}
BATCH_SIZE_TRAIN_LOADER = 64
BATCH_SIZE_TEST_LOADER = 64
TRAIN_EPOCHS = 200
INPUT_RESIZE = 224
FEATURE_VEC_SIZE = 256
LEARNING_RATE = 0.001
# DATA './Data/Sketchy/256x256' 12500 Images
DATA = "./Data/Sketchy/256x256"
DATA_STYLIZED = "./Data/Sketchy/256x256_stylized"
#Set stylized to true if want to train on stylized data, else false
STYLIZED = True

#Random Transforms Parameters
colorJitter = T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
random_rotation_angle_range_min=2
random_rotation_angle_range_max=15
#Erase coord limits
erase_coord_min = 56
erase_coord_max = 168


# transform_randoms = transforms.Compose(
#     [
#         T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#         torchvision.transforms.Resize((INPUT_RESIZE, INPUT_RESIZE)),
#         #T.RandomRotation(degrees=(10, 180)),
#         #T.RandAugment(),
#         transforms.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

transform_norandoms = transforms.Compose(
    [
        torchvision.transforms.Resize((INPUT_RESIZE, INPUT_RESIZE)),
        transforms.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def is_matching(path_image, path_sketch):
    splitted_image = path_image.split(".")[0]
    splitted_sketch = path_sketch.split("-")[0]
    if splitted_image == splitted_sketch:
        return True
    else:
        return False

def is_matching_stylized(path_image, path_sketch):
    # print("Is matching_stylized method")
    # print(f"img path: {path_image}")
    # print(f"path sketch: {path_sketch}")
    splitted_image = path_image.split("_stylized")[0]
    splitted_sketch = path_sketch.split("-")[0]
    # print(f"splitted img: {splitted_image}")
    # print(f"splitted sketch: {splitted_sketch}")
    if splitted_image == splitted_sketch:
        return True
    else:
        return False

def match_images_to_sketches(imgs, sketches):
    # Create a dictionary that maps image names to sketches
    sketch_dict = {sketch.split('_')[1]: sketch for sketch in sketches}
    
    # Create an empty dictionary to store the matches
    matches = []
    
    # Loop through the images and look up the corresponding sketch in the dictionary
    for img in imgs:
        img_name = img.split('_')[1]
        img_name += "-1.png"
        # print(f"img_splitted: {img_name}")
        sketch = sketch_dict.get(img_name)
        if sketch is not None:
            matches.append((img,sketch))
    # print(f"sketch_dict: {sketch_dict}")
    
    # Return the dictionary of matches
    return matches



class ImgSketchDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, train=True, trainsplit_decimal=0.9, rnd_aug=True, stylized=False):
        if not (trainsplit_decimal > 0.0 and trainsplit_decimal < 1.0):
            raise ValueError("trainsplit_decimal must be in range ]0.0,1.0[")

        super().__init__(root, transform, target_transform)
        self.trainsplit_decimal = trainsplit_decimal
        self.train = train
        self.rnd_aug = rnd_aug
        self.stylized = stylized
        # imgs_dir e.g. QuickDraw_images_two
        print(os.listdir(self.root)[0])
        if (os.listdir(self.root)[0] == "photo" and os.listdir(self.root)[1] == "sketch"):
            self.imgs_dirs = os.path.join(root, os.listdir(self.root)[0])
            self.sketches_dirs = os.path.join(root, os.listdir(self.root)[1])
        elif(os.listdir(self.root)[0] == "sketch" and os.listdir(self.root)[1] == "photo"):
            self.imgs_dirs = os.path.join(root, os.listdir(self.root)[1])
            self.sketches_dirs = os.path.join(root, os.listdir(self.root)[0])
        else:
            raise ValueError("Dir structures must be /dataset/sketch and /dataset/photo")
        # sketches_dir e.g. QuickDraw_sketches_two
        
        print("IMGS DIR")
        print(self.imgs_dirs)
        print("SKETCHES DIR")
        print(self.sketches_dirs)
        self.samples = self.make_samples(self.imgs_dirs, self.sketches_dirs, multiple_sketches_per_img=False)

    def __getitem__(self, index: int):
        path_img, path_sketch, label = self.samples[index]
        img = self.loader(path_img)
        sketch = self.loader(path_sketch)

        if self.transform is not None and self.target_transform is not None:
            img = self.transform(img)
            sketch = self.target_transform(sketch)
            #Apply random transforms if trainset
            if self.rnd_aug == True:
                img, sketch = self.apply_random_transforms(img, sketch)
        else: raise ValueError("transform and target_transform cannot be none!")
    
        return sketch, img, path_sketch, path_img, label

    # return list of tuples with path to imgs,sketches like [(airplane1img.jpeg, airplane1sketch.jpeg),...]
    # needs paths like './data/QuickDraw_images_two/airplane/image_00001.jpg' to load with loader
    # If more sketches than imgs => only #images pairs
    # Sketch and Image have to contain same folder structure (i.e. airplanes, alarmclocks... folders both in images and sketches folders)
    # Given trainsplit_decimal == 0.9 IF Trainset => first 90% Images, first 90% Sketches, IF Testset => last 10% images, last 10%sketches
    # TODO Make only 1 sketch per 1 image pairs, no multiple sketches for each img -> Why? Because of ClipLoss (Use multiple_sketches_per_img=False)
    def make_samples(self, imgs_dirs, sketches_dirs, multiple_sketches_per_img=False):
        samples = []
        # nth_loop = 0
        #i = 0
        for dir in os.listdir(imgs_dirs):
            #i+=1
            # ./data/QuickDraw_images_two/airplane
            path_to_imgs = os.path.join(imgs_dirs, dir)
            # ./data/QuickDraw_sketches_two/airplane
            path_to_sketches = os.path.join(sketches_dirs, dir)
            #print(i)

            imgs = os.listdir(path_to_imgs)
            imgs.sort()
            sketches = os.listdir(path_to_sketches)
            sketches.sort()

            dataset_size_imgs = int(self.trainsplit_decimal * len(imgs))
            if self.train == True:
                imgs = imgs[:dataset_size_imgs]
                # sketches = sketches[:dataset_size_imgs]
            else:
                imgs = imgs[dataset_size_imgs:]
                # Cut off not needed sketches
                sketches = sketches[dataset_size_imgs:]
            # print(f"IMGSLEN of dir:{dir} is: {len(imgs)}")
            # print(f"SKETCHESLEN of dir:{dir} is: {len(sketches)}")
            # makes (img,sketch) pairs in form [(img1,sketch1-1),(img1,sketch1-2),(img2,sketch2-1),...,(img4,sketch4-3)]
            if multiple_sketches_per_img == True:
                index = 0
                for img in imgs:
                    at_least_one_match = False
                    while index < len(sketches):
                        if is_matching(img, sketches[index]):
                            imgPath = os.path.join(path_to_imgs, img)
                            sketchPath = os.path.join(path_to_sketches, sketches[index])
                            samples.append((imgPath, sketchPath, dir))
                            at_least_one_match = True
                            index += 1
                        elif at_least_one_match == False:
                            index += 1
                        else:
                            break
            # alternatively makes (img,sketch) pairs in form [(img1,sketch1-1), (img2,sketch2-1),...,(img4,sketch4-1)]
            else:
                if(self.stylized == False):
                    index = 0
                    counter = 0
                    # if(dir=="bell"):
                    #     for img in imgs:
                    #         print(f"img: {img}")
                    #     for sketch in sketches:
                    #         print(f"sketch: {sketch}")
                    for img in imgs:
                        while index < len(sketches):
                            if is_matching(img, sketches[index]):
                                imgPath = os.path.join(path_to_imgs, img)
                                sketchPath = os.path.join(path_to_sketches, sketches[index])
                                #samples = [(img,sketch,dir)]
                                samples.append((imgPath, sketchPath, dir))
                                index += 1
                                counter += 1
                                # print("IMG PATH")
                                # print(imgPath)
                                # print("SKETCH PATH")
                                # print(sketchPath)
                                break
                            else:
                                index += 1
                    # print(f"samplecounter: {counter}")
                elif(self.stylized == True):
                    #matches=[(img,sketch),(img,sketch),(img,sketch)]
                    matches = match_images_to_sketches(imgs, sketches)
                    for match in matches:
                        img, sketch = match
                        imgPath = os.path.join(path_to_imgs, img)
                        sketchPath = os.path.join(path_to_sketches, sketch)
                        samples.append((imgPath, sketchPath, dir))
                    
        print(len(samples))
        return samples

    # def apply_random_transforms(self, input, rotation_angle, erase_coord_i, erase_coord_j):
    #     # coords corners [56,56], [56,168], [168,56], [168,168]
    #     output = TF.erase(input,erase_coord_i, erase_coord_j,10,15,255)
    #     output = TF.rotate(output, rotation_angle)
    #     output = TF.hflip(output)

    #     return output

    def apply_random_transforms(self, img, sketch):
        #Apply random color jitter
        if random.random() > 0.5 and TRANSFORMS_CONFIG["jitter"]:
            img = colorJitter(img)
        if random.random() > 0.5 and TRANSFORMS_CONFIG["erase"]:
            erase_coord_i = random.randint(erase_coord_min, erase_coord_max)
            erase_coord_j = random.randint(erase_coord_min, erase_coord_max)
            img = TF.erase(img, erase_coord_i, erase_coord_j,10,15,255)
            sketch = TF.erase(sketch, erase_coord_i, erase_coord_j,10,15,255)
        #Apply random rotation
        if random.random() > 0.5 and TRANSFORMS_CONFIG["rotation"]:
            # coords corners [56,56], [56,168], [168,56], [168,168]
            rotation_angle = random.randint(random_rotation_angle_range_min, random_rotation_angle_range_max)
            img = TF.rotate(img, rotation_angle)
            sketch = TF.rotate(sketch, rotation_angle)
            centerCrop = T.CenterCrop(185)
            img = centerCrop(img)
            sketch = centerCrop(sketch)
            resize = T.Resize((INPUT_RESIZE, INPUT_RESIZE))
            img = resize(img)
            sketch = resize(sketch)
        #Apply horizontal flip
        if random.random() > 0.5 and TRANSFORMS_CONFIG["hflip"]:
            img = TF.hflip(img)
            sketch = TF.hflip(sketch)
        # # #Apply vertical flip
        if random.random() > 0.5 and TRANSFORMS_CONFIG["vflip"]:
            img = TF.vflip(img)
            sketch = TF.vflip(sketch)
        return img, sketch

                
                
class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    # What is logit_scale?
    def forward(self, image_features, text_features, logit_scale=1.0):
        device = image_features.device

        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return total_loss


class SiameseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.model = nn.Sequential(*list(self.model.modules())[:-1]) # strips off last linear layer
        self.sketch_model = nn.Sequential(*(list(resnet50(pretrained=True).children())[:-1]))
        self.sketch_model_fc = nn.Linear(2048, FEATURE_VEC_SIZE)
        self.image_model = nn.Sequential(*(list(resnet50(pretrained=True).children())[:-1]))
        self.image_model_fc = nn.Linear(2048, FEATURE_VEC_SIZE)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss = ClipLoss()

    def forward(self, inputs):
        results = {}
        if "sketch" in inputs:
            # print(x.shape)
            x = self.sketch_model(inputs["sketch"])
            x = torch.squeeze(x)
            # print(x.shape)
            x = self.sketch_model_fc(x)
            results["sketch"] = x
        if "image" in inputs:
            # print(x.shape)
            x = self.image_model(inputs["image"])
            x = torch.squeeze(x)
            # print(x.shape)
            x = self.image_model_fc(x)
            results["image"] = x
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):

        sketches = train_batch[0]
        images = train_batch[1]

        feature_vecs = self({"sketch": sketches, "image": images})

        # normalized features
        sketch_features = feature_vecs["sketch"] / feature_vecs["sketch"].norm(dim=-1, keepdim=True)
        image_features = feature_vecs["image"] / feature_vecs["image"].norm(dim=-1, keepdim=True)

        scale = self.logit_scale.exp()

        loss = self.loss.forward(sketch_features, image_features, scale)

        cos = nn.CosineSimilarity(dim=1)
        similarity = cos(sketch_features, image_features)
        mean_cosSim_ofBatch = torch.mean(similarity)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/scale", scale, on_step=False, on_epoch=True)
        self.log("cosineSim/TrainBatch", mean_cosSim_ofBatch, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):

        sketches = val_batch[0]
        images = val_batch[1]

        feature_vecs = self({"sketch": sketches, "image": images})

        # normalized features
        sketch_features = feature_vecs["sketch"] / feature_vecs["sketch"].norm(dim=-1, keepdim=True)
        image_features = feature_vecs["image"] / feature_vecs["image"].norm(dim=-1, keepdim=True)

        scale = self.logit_scale.exp()

        loss = self.loss.forward(sketch_features, image_features, scale)

        cos = nn.CosineSimilarity(dim=1)
        similarity = cos(sketch_features, image_features)
        mean_cosSim_ofBatch = torch.mean(similarity)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("cosineSim/ValBatch", mean_cosSim_ofBatch, prog_bar=True, on_step=False, on_epoch=True)
        return {"loss": loss}


import argparse
import os
import re
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-o", "--output_path", help="verbose output")
    args = parser.parse_args()
    return args

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger


class MyPrintingCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        # i=0
        # for param in pl_module.parameters():
        #     print(param)
        #     if i>2:
        #         break
        #     i+=1

root = DATA
train_set = ImgSketchDataset(root, transform_norandoms, transform_norandoms, train=True, trainsplit_decimal=0.9, rnd_aug=True)
# print(train_set.__len__)
test_set = ImgSketchDataset(root, transform_norandoms, transform_norandoms, train=False, trainsplit_decimal=0.9, rnd_aug=False)
train_set_stylized = ImgSketchDataset(DATA_STYLIZED, transform_norandoms, transform_norandoms, train=True, trainsplit_decimal=0.9, rnd_aug=True, stylized=True)
print(train_set_stylized.__len__)

# TODO Trainset batch size? Not sure what best size is, but for CLIPLOSS it should be pretty big
train_loader = DataLoader(train_set, BATCH_SIZE_TRAIN_LOADER, shuffle=True, num_workers=4)
val_loader = DataLoader(test_set, BATCH_SIZE_TEST_LOADER, shuffle=True, num_workers=4)
train_loader_stylized = DataLoader(train_set_stylized, BATCH_SIZE_TRAIN_LOADER, shuffle=True, num_workers=4)
# for i,batch in enumerate(train_loader_stylized):
#     print(batch[2][2])
#     print(batch[3][2])
#     print(batch[4][2])
#     if(i>8):
#         break
    

# exit(1)

transforms_str = str(TRANSFORMS_CONFIG)
transforms_str = transforms_str.replace('{', '-')
transforms_str = transforms_str.replace('}', '-')
if(STYLIZED == True):
    logName = f"Stylized={STYLIZED},Transforms={transforms_str},Batchsize={BATCH_SIZE_TRAIN_LOADER},LR={LEARNING_RATE},FEATURE_VEC_SIZE={FEATURE_VEC_SIZE}"
else:
    logName = f"Transforms={transforms_str},Batchsize={BATCH_SIZE_TRAIN_LOADER},LR={LEARNING_RATE},FEATURE_VEC_SIZE={FEATURE_VEC_SIZE}"
logger = TensorBoardLogger("train_logs", name=logName)

# import torch.utils.data as data_utils
# indices = torch.arange(0,100)
# overfit_trainset = data_utils.Subset(train_set, indices)
# print(len(overfit_trainset))
# overfit_loader = DataLoader(overfit_trainset, BATCH_SIZE_TRAIN_LOADER, shuffle=True, num_workers=16)
def main():
    args = parse_args()
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    
    #every_n_epochs=10
    checkpoint_name = logName+",{epoch}"
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=args.output_path, save_top_k=-1, filename=checkpoint_name)
   

    # model
    model = SiameseModel()
    

    #model = SiameseModel.load_from_checkpoint("output/epoch=19-step=440.ckpt")

    # training
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback,MyPrintingCallback()],
        accelerator="gpu",
        devices=-1,
        max_epochs=TRAIN_EPOCHS,
        gradient_clip_val=0.5
    )
    # trainer.fit(model, train_loader, val_loader)
    if(STYLIZED == True):
        print("Training started on stylized data")
        trainer.fit(model, train_loader_stylized, val_loader)
    else:
        print("Training started on regular data")
        trainer.fit(model, train_loader, val_loader)

    return 0


if __name__ == "__main__":
    sys.exit(main())
