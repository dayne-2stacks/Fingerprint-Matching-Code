from src.model.ngm import Net

model=Net()

for n, p in model.named_parameters():
    print(n, p.shape)