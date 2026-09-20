import os
from PIL import Image, ImageDraw, ImageFont
import glob

asset_dir = "frontend/public/assets"
files = glob.glob(os.path.join(asset_dir, "snapshot_duo_elite_*m.png"))

for filepath in files:
    img = Image.open(filepath).convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # Coordinates of the middle title (empirically calculated based on 4023x911 size)
    # The image is 3 subplots. Middle one is from ~1341 to 2682.
    # Title is centered above the middle plot. Let's cover y=0 to y=80, x=1350 to 2700.
    draw.rectangle([w/3 + 50, 0, 2*w/3 - 50, 95], fill=(255, 255, 255, 255))
    
    # Write the new text
    depth = filepath.split("_")[-1].replace(".png", "")
    text = f"Tri-Breed Engine Reconstruction ({depth} Depth)"
    
    # Use default font, scaled up, or just Arial if available
    try:
        font = ImageFont.truetype("arialbd.ttf", 46)
    except:
        font = ImageFont.load_default()
        
    # Get text size to center it
    try:
        text_bbox = font.getbbox(text)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    except AttributeError:
        tw, th = draw.textsize(text, font=font)
        
    x = (w/3) + (w/3 - tw)/2
    y = 35 # approximate y coordinate
    
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)
    
    # Save image
    img.save(filepath)
    print(f"Fixed {filepath}")
