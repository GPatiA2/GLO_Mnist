from torch.utils.data import Dataset

class IndexedDataset(Dataset):

    def __init__(self, imgs):

        self.imgs = imgs

    def __getitem__(self, index):

        img = self.imgs[index]

        return (img, index)
    
    def __len__(self):
            
        return len(self.imgs)