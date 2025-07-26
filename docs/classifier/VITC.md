# VITC

::: vitm.models.classifier.VITC
    options:
      show_source: true
      show_signature: true

```python
    vit = VisionTransformer(...)  # pretrained or custom ViT   
    model = VITC(vit)    
    x = torch.randn(8, 3, 224, 224)    
    logits = model(x) 
```