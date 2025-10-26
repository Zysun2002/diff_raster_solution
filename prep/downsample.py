from pathlib import Path
from PIL import Image

# subfolder-level


def run(subfold_path):
    size_map = {256: 128, 128: 64, 64: 32, 32: 16}

    for orig_size, target_size in size_map.items():
        infile = subfold_path / f"{orig_size}_image.png"
        if not infile.exists():
            # print(f"Skipping {infile.name} (not found)")
            continue

        # Open image
        img = Image.open(infile)

        # Resize with anti-aliasing
        img_down = img.resize((target_size, target_size), resample=Image.LANCZOS)
        # other options: NEAREST, BILINEAR, BICUBIC


        # Save output
        outfile = subfold_path / f"aa_{target_size}.png"
        img_down.save(outfile)
    
    pass


# if __name__ == "__main__":
#     run(subfold = Path("/workspace/diffvg/diffAaSolution/data/bow"),
#          )
    

def batch(fold_path):
    
    # one-level only
    for subfold_path in fold_path.glob("*/"):
        run(subfold_path)

if __name__ == "__main__":
    
    fold_path = Path("/workspace/diffvg/diffAaSolution/data")
    batch(fold_path)