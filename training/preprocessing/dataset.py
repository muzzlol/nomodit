from datasets import load_dataset

train_ds = load_dataset("grammarly/coedit", split="train")
val_ds = load_dataset("grammarly/coedit", split="validation")

print(train_ds)
print(train_ds[0])


print(val_ds)
print(val_ds[0])


# For interactive exploration in a notebook, you could use:
# help(ds)  # For full documentation
# ds?       # In Jupyter notebook for quick reference

