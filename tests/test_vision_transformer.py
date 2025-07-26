import torch
import torch.nn as nn
from vitm.models.models import VisionTransformer

def test_vision_transformer():
    image_size = (224, 224)
    num_classes = 10
    norm_layer = nn.LayerNorm
    vision_transformer = VisionTransformer(
            image_size,
            patch_size=4,
            num_classes=num_classes,
            num_heads=8,
            mlp_ratio=0.8,
            norm_layer=norm_layer,
            embed_norm_layer=norm_layer,
            final_norm_layer=norm_layer,
            parallel_block=True,
            )
    x = torch.randn(8, 3, 224, 224)
    out = vision_transformer(x)
    print(out.shape)

def test_vision_transformer_parallel(device_id, world_size, path_to_config="models/sv22b.yaml"):
    from vitm.models.sv22b import setup, create_mesh
    import torch.distributed as dist

    setup(device_id, world_size)

    print(f"[Rank {device_id}] setup complete", flush=True)

    # Config
    image_size = (224, 224)
    num_classes = 10
    norm_layer = nn.LayerNorm

    torch.manual_seed(0)

    vision_transformer = VisionTransformer(
        image_size,
        patch_size=4,
        num_classes=num_classes,
        num_heads=8,
        mlp_ratio=0.8,
        norm_layer=norm_layer,
        embed_norm_layer=norm_layer,
        final_norm_layer=norm_layer,
        parallel_block=True
    )

    rows = world_size // 1
    cols = 1
    mesh_shape = (rows, cols)
    mesh = create_mesh(device="cuda", mesh_shape=mesh_shape, shard_along_dims=None)

    print(f"[Rank {device_id}] mesh created: {mesh}", flush=True)

    vision_transformer.parallelize_blocks(mesh)
    vision_transformer = vision_transformer.to(device_id)
    vision_transformer.train()

    x = torch.randn(8, 3, 224, 224).to(f"cuda:{device_id}")

    out = vision_transformer(x)

    print(f"[Rank {device_id}] forward done. Output shape: {out.shape}", flush=True)

    dist.destroy_process_group()

    return out


def test_vision_transformer_parallel_run():
    import os
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    out = test_vision_transformer_parallel(local_rank, world_size)
    print(f"[Rank {local_rank}] final output shape: {out.shape}", flush=True)


if __name__ == "__main__":
    test_vision_transformer_parallel_run()
