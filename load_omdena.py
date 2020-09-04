import torch
from hub import dataset

# Load data
ds = dataset.load("abhinav/aerial-omdena")

# Transform into pytorch
ds = ds.to_pytorch()
ds = torch.utils.data.DataLoader(
    ds, batch_size=2, collate_fn=ds.collate_fn
)

# Iterate over the data
for batch in ds:
    print(batch["image_lat"])
    print(batch["image_lon"])
    print(batch["cluster_lat"])
    print(batch["cluster_lon"])
    print(batch["cons_pc"])
    print(batch["nightlights"])
    print(batch["nightlights_bin"])
    print(batch["image"])
