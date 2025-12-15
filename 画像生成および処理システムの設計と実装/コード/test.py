import torch,sys
print("torch.__version__:", getattr(torch,'__version__', None))
print("torch.version.cuda:", getattr(torch.version,'cuda', None))
print("torch.cuda.is_available():", torch.cuda.is_available() if hasattr(torch,'cuda') else None)
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    try:
        print("device name:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("get_device_name failed:", e)
