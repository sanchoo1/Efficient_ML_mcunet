import torch
from torchsummary import summary
from mcunet.model_zoo import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for model_id in ("mcunet-in2", "mcunet-in4"):
    model, resolution, _ = build_model(model_id, pretrained=True)
    model = model.to(device)
    print(f"\n===== {model_id} (输入：{resolution}×{resolution}) =====")

    summary(model, input_size=(3, resolution, resolution), device=str(device))
