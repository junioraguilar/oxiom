import torch 
from ultralytics.nn.tasks import DetectionModel 
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel']) 
print("Ajuste de compatibilidade aplicado com sucesso") 
