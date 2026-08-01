import torch
import timm

def main():
    print("Creating model...")
    timm_kwargs = {
        'img_size': 224, 'patch_size': 14, 'depth': 24, 'num_heads': 24,
        'init_values': 1e-5, 'embed_dim': 1536, 'mlp_ratio': 2.66667*2,
        'num_classes': 0, 'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 'dynamic_img_size': True
    }
    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=False, **timm_kwargs)
    
    print("\nBlocks attribute:")
    print(type(model.blocks))
    print(len(model.blocks))
    
    print("\nNorm attribute:")
    print(hasattr(model, 'norm'))
    
    print("\nForward Head attribute:")
    print(hasattr(model, 'forward_head'))

if __name__ == "__main__":
    main()
