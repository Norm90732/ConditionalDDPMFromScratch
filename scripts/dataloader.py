import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = datasets.Flowers102(root='./data', split='train', download=True,transform=transform)
val_set = datasets.Flowers102(root='./data', split='val', download=True,transform=transform)
test_set = datasets.Flowers102(root='./data', split='test', download=True,transform=transform)

train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)
