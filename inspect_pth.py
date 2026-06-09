import torch
import sys

def inspect_checkpoint(path):
    print(f"Loading checkpoint {path}...")
    try:
        # Map location to CPU to avoid OOM
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        # Determine structure
        if isinstance(checkpoint, dict):
            print("Checkpoint is a Dictionary.")
            print("Top-level keys:", list(checkpoint.keys()))
            
            # Find the state dict
            state_dict = None
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
                
            print(f"\nFound {len(state_dict)} tensors in state_dict.")
            
            # Print first 20 keys and their shapes to infer architecture
            print("\nFirst 20 keys and shapes:")
            for i, (k, v) in enumerate(state_dict.items()):
                if i >= 20:
                    break
                print(f"{k}: {v.shape if hasattr(v, 'shape') else type(v)}")
                
        else:
            print("Checkpoint is not a dictionary. Type:", type(checkpoint))
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

if __name__ == "__main__":
    inspect_checkpoint("fmow_finetune.pth")
