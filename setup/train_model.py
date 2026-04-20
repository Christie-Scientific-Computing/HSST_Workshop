"""
Script for training the updated model used in Part 3.
Only train on abdo images for now
"""
import os
import numpy as np
from pathlib import Path
import torch
import albumentations as A
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

structure_names = ['Body', 'Brainstem', 'Mandible', 'Parotids', 'Spinalcord']
datapath = Path('/config/teaching/HSST_Workshop/data/HnN_data/')
im_size = 192 ##Flare data is 256

## -------------- PRE-PROCESSING ------------
def getFiles(targetdir:Path):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def erode_nask(arr):
    holder= np.zeros_like(arr)
    struct = generate_binary_structure(2, 2)
    for val in np.unique(arr):
        if val == 0: continue
        msk = np.where(arr==val, 1, 0)
        msk = binary_erosion(msk, structure=struct, iterations=1)
        holder[msk] = val
    return holder

def load_images_and_masks(root_dir: Path):
    fnames = sorted(getFiles(root_dir / 'ims'))
    
    ims = np.zeros((len(fnames), im_size, im_size), dtype=np.float32)
    masks = np.zeros((len(fnames), im_size, im_size), dtype=np.int8)
    
    for fdx, fname in enumerate(fnames):
        ims[fdx] = np.load(root_dir / 'ims' / fname)
        masks[fdx] = erode_nask(np.load(root_dir / 'masks' / fname))


    return ims, masks 

def window_level(data, window=350, level=50):
    low_edge  = level - (window//2)
    high_edge = level + (window//2)
    windowed_data = (np.clip(data, low_edge, high_edge) - low_edge)/window
    return windowed_data

def load_data():
    train_path = datapath / 'train'
    test_path = datapath / 'test'
    train_ims, train_masks = load_images_and_masks(train_path)
    test_ims, test_masks = load_images_and_masks(test_path)
    train_ims = window_level(train_ims)
    test_ims = window_level(test_ims)
    return (train_ims, train_masks), (test_ims, test_masks)

#### -------------------------------

class DataSet(torch.utils.data.Dataset):
    def __init__(self, image_array, mask_array, transform):
        super().__init__()
        self.image_array = image_array
        self.mask_array = mask_array
        self.transform = transform
    
    def __len__(self):
        return self.image_array.shape[0]

    def __getitem__(self, idx):
        image = self.image_array[idx]
        mask = self.mask_array[idx]
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image[None], mask

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.FPN(
            "resnet18",
            in_channels=1, 
            classes=len(structure_names)+1,
            encoder_weights='imagenet')
        
        self.loss_fcn = smp.losses.DiceLoss("multiclass", from_logits=True)
        self.optimizer = torch.optim.Adam

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=1e-4)
        return {"optimizer": optimizer, "reduce_on_plateau": True}
    
    def training_step(self, batch, batch_idx):
        img, mask = batch
        msk_pred = self(img)
        loss = self.loss_fcn(msk_pred, mask.long())
        self.log("loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        msk_pred = self(img)
        loss = self.loss_fcn(msk_pred, mask.long())
        self.log("val_loss", loss)
        return loss


def train():
    train_data, test_data = load_data()
    train_transforms = A.Compose([
        A.Rotate(45),
        A.GaussianBlur(blur_limit=2, sigma_limit=2, p=1),
        #A.CenterCrop(height=120, width=120, p=1),
        #A.Resize(im_size, im_size),
        #A.GridElasticDeform(num_grid_xy=[10, 10], magnitude=5, p=1),
        A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))
    ])
    test_transforms = A.Compose([
        A.GaussianBlur(blur_limit=2, sigma_limit=2, p=1),
        # A.CenterCrop(height=64, width=64, p=1),
        # A.Resize(im_size, im_size),
        A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))
    ])

    train_dataset = DataSet(*train_data, train_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
    
    test_dataset = DataSet(*test_data, test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    if plot_train_input:
        train_img, train_msk = next(iter(train_dataloader))
        train_msk = prep_mask_for_plot(train_msk.numpy())
        #train_msk = train_msk.numpy()
        fig, ax = plt.subplots()
        ax.imshow(train_img.numpy()[7,...].squeeze(), cmap='Greys_r')
        ax.imshow(train_msk[7,...].squeeze(), alpha=0.5, cmap='jet', vmax=5)
        ax.invert_yaxis()
        plt.show()
        #fig.savefig('./tmp/train_im.png')
        exit()

    checkpoint_callback = ModelCheckpoint(
        dirpath="HnN_checkpoints_erodeOneIter",          # Folder to save checkpoints
        filename="model-{epoch:02d}-{val_loss:.2f}",  # Naming pattern
        save_top_k=3,                   # Keep only the best 3 models
        monitor="val_loss",             # Metric to monitor
        mode="min",                     # "min" for loss, "max" for accuracy
        save_last=True,                  # Always save the last epoch
        save_weights_only=True
    )
    # Train
    model = Model()
    trainer = pl.Trainer(max_epochs=50, callbacks=[checkpoint_callback], logger=CSVLogger("logs", "test"))
    trainer.fit(
        model, train_dataloader, test_dataloader)
    trainer.save_checkpoint("HnN_erodeOneIter.ckpt")

def test():
    ## Test model on original data to check degradation
    model = Model.load_from_checkpoint('./HnN_new_model.ckpt')
    _, test_data = load_data()
    test_transforms = A.Compose([
        # A.GaussianBlur(blur_limit=1, sigma_limit=2, p=1),
        # A.CenterCrop(height=128, width=128, p=1),
        # A.Resize(256, 256),
        A.Normalize(mean=(np.mean([0.485, 0.456, 0.406])), std=(np.mean([0.229, 0.224, 0.225])))
    ])
    
    test_dataset = DataSet(*test_data, test_transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

    for img, mask in test_dataloader:
        print(img.shape, mask.shape)
        pred = model.forward(img)
        pred = np.argmax(pred.detach().numpy(), axis=1)
        pred = prep_mask_for_plot(pred)
        mask = prep_mask_for_plot(mask.numpy())

        fig, ax = plt.subplots()
        ax.imshow(img.numpy()[7,...].squeeze(), cmap='gray')
        #ax.imshow(mask[3,...].squeeze(), cmap='jet', alpha=0.5)
        ax.imshow(pred[7,...].squeeze(), alpha=0.5, cmap='jet', vmax=5)
        ax.invert_yaxis()
        plt.show()
        #fig.savefig('./tmp/train_im.png')
        exit()


def prep_mask_for_plot(mask):
    #mask = mask.astype(np.float32)
    print(np.unique(mask))
    mask = np.where(mask == 0, np.nan, mask)
    return np.where(mask == 1, np.nan, mask)

if __name__ == '__main__':
    plot_train_input = False
    #train()    
    
    test()