import torch

# test cuda availability
print(torch.cuda.is_available())
# print available cuda devices
print(torch.cuda.device_count())
# print available cuda device names
print(torch.cuda.get_device_name(0))