import torch
print (torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"{i}: {torch.cuda.get_device_name(i)}")