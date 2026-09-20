import os
import glob
from PIL import Image, ImageDraw, ImageFont

asset_dir = "frontend/public/assets"
files = glob.glob(os.path.join(asset_dir, "snapshot_duo_elite_*m.png"))

for filepath in files:
    img = Image.open(filepath).convert("RGBA")
    
    # Sample the exact background color from a safe spot (top-left)
    bg_color = img.getpixel((10, 10))
    
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # Carefully erase the old title. The black plot borders start around y=75. 
    # We will only erase y=0 to y=65 to be absolutely safe and avoid erasing the border!
    draw.rectangle([w * 0.34, 0, w * 0.66, 68], fill=bg_color)
    
    depth = filepath.split("_")[-1].replace(".png", "")
    text = f"Tri-Breed Engine Reconstruction ({depth} Depth)"
    
    try:
        font = ImageFont.truetype("arialbd.ttf", 52)  # slightly larger to match Matplotlib 13pt
    except:
        font = ImageFont.load_default()
        
    try:
        text_bbox = font.getbbox(text)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
        
    # Center horizontally in the middle third
    x = (w/3) + ((w/3) - tw) / 2
    # Place vertically so it aligns with the other matplotlib titles (~ y=15)
    y = 15 
    
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)
    img.save(filepath)
    print(f"Flawlessly fixed {filepath}")
