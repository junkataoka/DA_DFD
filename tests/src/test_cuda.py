import torch
# test cuda availability
is_cuda = torch.cuda.is_available()
print("Cuda availability: ", torch.cuda.is_available())
if is_cuda:
    # print available cuda devices
    device_count = torch.cuda.device_count()
    print("Available number of GPUs: ", device_count)
    for i in range(device_count):
        # print available cuda device names
        print("Device name: ", torch.cuda.get_device_name(i))