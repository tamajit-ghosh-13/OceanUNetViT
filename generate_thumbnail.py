import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080

# 1. Base Canvas
canvas = Image.new("RGBA", (W, H), (14, 17, 24, 255))

# Fonts
font_dir = "/System/Library/Fonts"
font_title = ImageFont.truetype(f"{font_dir}/Supplemental/Arial Bold.ttf", 132)
font_h2 = ImageFont.truetype(f"{font_dir}/Supplemental/Arial Bold.ttf", 44)
font_pill = ImageFont.truetype(f"{font_dir}/Supplemental/Arial Bold.ttf", 46)
font_card_title = ImageFont.truetype(f"{font_dir}/Supplemental/Arial Bold.ttf", 26)
font_card_body = ImageFont.truetype(f"{font_dir}/Menlo.ttc", 23)
font_card_big = ImageFont.truetype(f"{font_dir}/Supplemental/Arial Bold.ttf", 36)
font_mono_small = ImageFont.truetype(f"{font_dir}/Menlo.ttc", 21)

# 2. Right Side: Light Background with Diagonal Cut
# Center of diagonal around x=960
top_x = 1040
bot_x = 780

right_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
draw_right = ImageDraw.Draw(right_layer)
polygon = [(top_x, 0), (W, 0), (W, H), (bot_x, H)]
draw_right.polygon(polygon, fill=(248, 250, 252, 255)) # Soft clean white

# Subtle architectural plot in top-right
if os.path.exists("plots/live_ocean_thermal_cross_section.png"):
    cs_img = Image.open("plots/live_ocean_thermal_cross_section.png").convert("RGBA")
    cs_w = 980
    cs_h = int(cs_img.height * (cs_w / cs_img.width))
    cs_img = cs_img.resize((cs_w, cs_h), Image.Resampling.LANCZOS)
    cs_canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cs_canvas.paste(cs_img, (930, 160))
    # Blend with 85% opacity
    cs_alpha = Image.new("L", (W, H), 0)
    cs_alpha.paste(Image.new("L", (cs_w, cs_h), 220), (930, 160))
    right_layer.paste(cs_canvas, (0, 0), cs_alpha)

# Subtle satellite heatmap on left
left_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
if os.path.exists("plots/snapshot_ensemble_thermocline_100m.png"):
    snap_img = Image.open("plots/snapshot_ensemble_thermocline_100m.png").convert("RGBA")
    snap_w = 1100
    snap_h = int(snap_img.height * (snap_w / snap_img.width))
    snap_img = snap_img.resize((snap_w, snap_h), Image.Resampling.LANCZOS)
    snap_dark = Image.new("RGBA", snap_img.size, (14, 17, 24, 170))
    snap_img = Image.alpha_composite(snap_img, snap_dark)
    left_layer.paste(snap_img, (-120, 260))

# Composite left & right
canvas = Image.alpha_composite(canvas, left_layer)

mask = Image.new("L", (W, H), 0)
mask_draw = ImageDraw.Draw(mask)
mask_draw.polygon(polygon, fill=255)
canvas.paste(right_layer, (0, 0), mask)

# Split border & drop shadow
draw = ImageDraw.Draw(canvas)
for offset in range(-8, 9):
    alpha = int(80 * (1 - abs(offset)/9))
    draw.line([(top_x + offset, 0), (bot_x + offset, H)], fill=(0, 0, 0, alpha), width=3)
draw.line([(top_x, 0), (bot_x, H)], fill=(203, 213, 225, 230), width=4)

# 3. macOS Window Chrome / Top Bar
bar_h = 75
draw.rectangle([(0, 0), (W, bar_h)], fill=(18, 21, 30, 255))
# Window controls
dots = [(35, 38, (255, 95, 87)), (65, 38, (254, 188, 46)), (95, 38, (40, 200, 64))]
for dx, dy, color in dots:
    draw.ellipse([(dx - 10, dy - 10), (dx + 10, dy + 10)], fill=color)

# Top Bar text
draw.text((140, 25), "$ oceanembed --region NIO --depths 0..1000m", fill=(148, 163, 184), font=font_mono_small)
draw.text((W - 460, 25), "mission-control.oceanembed.ai  ● LIVE", fill=(100, 116, 139), font=font_mono_small)

# 4. Section Headings
draw.text((80, 120), "$ 2D SATELLITE", fill=(241, 245, 249), font=font_h2)
draw.text((80 + 340, 120), "BLIND SPOT", fill=(239, 68, 68), font=font_h2)

draw.text((1180, 120), "AUTONOMOUS", fill=(15, 23, 42), font=font_h2)
draw.text((1180 + 340, 120), "RECOVERY", fill=(2, 132, 199), font=font_h2)

# 5. Floating Alert Card on Left
card_x, card_y = 70, 715
card_w, card_h = 630, 255
draw.rounded_rectangle([(card_x, card_y), (card_x + card_w, card_y + card_h)], radius=18, fill=(22, 23, 31, 245), outline=(220, 38, 38, 180), width=2)
# Alert badge pill
alert_pill_x, alert_pill_y = card_x + 24, card_y + 22
draw.rounded_rectangle([(alert_pill_x, alert_pill_y), (alert_pill_x + 130, alert_pill_y + 36)], radius=8, fill=(185, 28, 28, 220))
draw.text((alert_pill_x + 12, alert_pill_y + 6), "⚠️ ALERT", fill=(255, 255, 255), font=font_card_title)

draw.text((card_x + 24, card_y + 76), "THERMOCLINE GRADIENT BARRIER", fill=(248, 113, 113), font=font_card_title)
draw.text((card_x + 24, card_y + 116), "• Depth Barrier: 50m – 150m (Severe dT/dz)", fill=(226, 232, 240), font=font_card_body)
draw.text((card_x + 24, card_y + 150), "• Surface Penetration: < 1 mm (Satellites)", fill=(148, 163, 184), font=font_card_body)
draw.text((card_x + 24, card_y + 184), "• Cat-5 Amphan Churn & IOD Fronts", fill=(148, 163, 184), font=font_card_body)
draw.text((card_x + 24, card_y + 218), "-o- Baseline 7-ch Error: > 1.05°C RMSE", fill=(239, 68, 68), font=font_mono_small)

# 6. Status Cards on Right
rcard_x, rcard_y = 1150, 725
rcard_w, rcard_h = 320, 235

draw.rounded_rectangle([(rcard_x, rcard_y), (rcard_x + rcard_w, rcard_y + rcard_h)], radius=18, fill=(255, 255, 255, 250), outline=(226, 232, 240, 255), width=2)
draw.text((rcard_x + 24, rcard_y + 24), "STATUS CARDS", fill=(100, 116, 139), font=font_card_title)
draw.text((rcard_x + 24, rcard_y + 68), "DUO-ELITE: 15", fill=(15, 23, 42), font=font_card_big)
draw.text((rcard_x + 24, rcard_y + 112), "DEPTH LEVELS", fill=(2, 132, 199), font=font_card_big)
draw.text((rcard_x + 24, rcard_y + 172), "v4_ext + v5_ft SIMPLEX", fill=(71, 85, 105), font=font_mono_small)

rcard2_x = rcard_x + 350
rcard2_w = 340
draw.rounded_rectangle([(rcard2_x, rcard_y), (rcard2_x + rcard2_w, rcard_y + rcard_h)], radius=18, fill=(255, 255, 255, 250), outline=(226, 232, 240, 255), width=2)
# Green check pill
draw.rounded_rectangle([(rcard2_x + 24, rcard_y + 22), (rcard2_x + 310, rcard_y + 62)], radius=10, fill=(240, 253, 244), outline=(34, 197, 94, 200), width=1)
draw.text((rcard2_x + 38, rcard_y + 30), "✓ MONOTONICITY PASSED", fill=(22, 101, 52), font=font_card_title)

draw.text((rcard2_x + 24, rcard_y + 84), "IN-SITU VALIDATION", fill=(100, 116, 139), font=font_card_title)
draw.text((rcard2_x + 24, rcard_y + 120), "668,045 FLOATS", fill=(15, 23, 42), font=font_card_big)
draw.text((rcard2_x + 24, rcard_y + 172), "ARGO GDAC • r = 0.903", fill=(16, 185, 129), font=font_mono_small)

# 7. Big Split Center Title: "OCEAN" [▶] "EMBED"
word1 = "OCEAN"
word2 = "EMBED"

b1 = draw.textbbox((0, 0), word1, font=font_title)
w1 = b1[2] - b1[0]
b2 = draw.textbbox((0, 0), word2, font=font_title)
w2 = b2[2] - b2[0]

yt_w, yt_h = 160, 112
gap = 36
total_title_w = w1 + gap + yt_w + gap + w2

start_x = (W - total_title_w) // 2
title_y = 420

x_ocean = start_x
x_yt = x_ocean + w1 + gap
x_embed = x_yt + yt_w + gap

# Glow & shadow for OCEAN
for g_off in range(16, 0, -3):
    draw.text((x_ocean, title_y), word1, fill=(0, 0, 0, 100), font=font_title)
draw.text((x_ocean + 4, title_y + 4), word1, fill=(10, 14, 20, 200), font=font_title)
draw.text((x_ocean, title_y), word1, fill=(255, 255, 255), font=font_title)

# Shadow for EMBED
for g_off in range(16, 0, -3):
    draw.text((x_embed, title_y), word2, fill=(255, 255, 255, 120), font=font_title)
draw.text((x_embed + 3, title_y + 3), word2, fill=(203, 213, 225), font=font_title)
draw.text((x_embed, title_y), word2, fill=(15, 23, 42), font=font_title)

# YouTube Play Button Badge in the exact center
yt_y = title_y + 12

for s_off in range(14, 0, -2):
    draw.rounded_rectangle([(x_yt - s_off, yt_y - s_off + 4), (x_yt + yt_w + s_off, yt_y + yt_h + s_off + 4)], radius=36, fill=(220, 0, 0, int(25 * (1 - s_off/14))))

draw.rounded_rectangle([(x_yt, yt_y), (x_yt + yt_w, yt_y + yt_h)], radius=34, fill=(255, 0, 0, 255))

# White play triangle inside
tri_h = 44
tri_w = 40
tx1 = x_yt + (yt_w - tri_w) // 2 + 5
ty1 = yt_y + (yt_h - tri_h) // 2
tx2 = tx1
ty2 = ty1 + tri_h
tx3 = tx1 + tri_w
ty3 = ty1 + tri_h // 2
draw.polygon([(tx1, ty1), (tx2, ty2), (tx3, ty3)], fill=(255, 255, 255, 255))

# 8. Stat Pill Badge directly below the title (matching MTTR pill in parsed image)
pill_text = "RMSE: 1.05°C  ➔  0.72°C  (-30.8%)"
pill_box = draw.textbbox((0, 0), pill_text, font=font_pill)
pw = pill_box[2] - pill_box[0] + 74
ph = 84
px = (W - pw) // 2
py = title_y + 155

# Glow around pill
for p_glow in range(14, 0, -2):
    draw.rounded_rectangle([(px - p_glow, py - p_glow), (px + pw + p_glow, py + ph + p_glow)], radius=42, fill=(16, 185, 129, int(20 * (1 - p_glow/14))))

draw.rounded_rectangle([(px, py), (px + pw, py + ph)], radius=40, fill=(13, 20, 28, 250), outline=(52, 211, 153, 255), width=3)
draw.text((px + 37, py + 16), pill_text, fill=(52, 211, 153), font=font_pill)

# Save thumbnail
canvas.convert("RGB").save("assets/oceanembed_video_thumbnail.png", "PNG", quality=98)
print("✅ Saved polished high-res thumbnail to assets/oceanembed_video_thumbnail.png")
