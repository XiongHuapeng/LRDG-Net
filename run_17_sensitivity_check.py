import torch
from paper_experiments.sensitivity.config import MODEL
from paper_experiments.sensitivity.src.models.lrdg_net import LRDGNet
if __name__ == "__main__":
    m=LRDGNet(MODEL.geometry_in_channels,MODEL.degradation_in_channels,MODEL.geometry_hidden,MODEL.degradation_hidden,MODEL.instance_channels,MODEL.classifier_hidden,MODEL.dropout,MODEL.num_classes)
    y=m(torch.randn(20,12),torch.randn(20,4),torch.tensor([0]*8+[1]*12),2)
    n=sum(p.numel() for p in m.parameters())
    assert n==10898, n
    print("LRDG-Net params:",n,"logits:",tuple(y.shape))
