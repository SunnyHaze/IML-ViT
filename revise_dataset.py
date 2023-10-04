import utils.misc as misc
import iml_vit_model
import torch

target_path= r"G:\Checkpoints\IML-ViT\output_dir_13\checkpoint-144.pth"
target_path=r"D:\workspace\IML-ViT\checkpoints\checkpoint-144.pth"
output_dir = r"D:\workspace\IML-ViT\checkpoints"


model = iml_vit_model.iml_vit_model(
    vit_pretrain_path=r"D:\workspace\IML-ViT\pretrained-weights\mae_pretrain_vit_base.pth"
)

mydict = torch.load(target_path)
my_model = mydict['model']
missing, unexpected = model.load_state_dict(my_model, strict=False)
print(missing)
print(unexpected)
# print(model)
import os
mydict['model'] = model.state_dict()
torch.save(mydict['optimizer'], os.path.join(output_dir, "optimizer_checkpoint-144.pth"))

print(mydict.keys())
