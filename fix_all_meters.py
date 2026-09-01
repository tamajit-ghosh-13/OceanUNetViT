import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# 1. Cyclone (Blue -> Yellow -> Red). No clipping.
target_1 = r"""                        <div className="w-full h-3 bg-surface-container rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute inset-0 bg-gradient-to-r from-red-500 via-green-400 to-blue-600"
                            style={{ clipPath: `inset(0 ${100 - Math.min(100, ((inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50) / 120) * 100)}% 0 0)` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, ((inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50) / 120) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""

new_1 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-blue-500 via-yellow-500 to-rose-600 rounded-full border border-glass-border relative">
                          <div 
                            className="absolute top-0 bottom-0 w-2 bg-white shadow-[0_0_10px_white] z-10 rounded-full border border-gray-300"
                            style={{ left: `calc(${Math.min(100, ((inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50) / 120) * 100)}% - 4px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_1, new_1)

# 2. Heatwave (Blue -> Orange -> Red). No clipping.
target_2 = r"""                        <div className="w-full h-3 bg-surface-container rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute inset-0 bg-gradient-to-r from-blue-400 via-orange-400 to-red-600"
                            style={{ clipPath: `inset(0 ${100 - Math.min(100, (2.4 / 4.0) * 100)}% 0 0)` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, (2.4 / 4.0) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""

new_2 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-blue-400 via-orange-400 to-red-600 rounded-full border border-glass-border relative">
                          <div 
                            className="absolute top-0 bottom-0 w-2 bg-white shadow-[0_0_10px_white] z-10 rounded-full border border-gray-300"
                            style={{ left: `calc(${Math.min(100, (2.4 / 4.0) * 100)}% - 4px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_2, new_2)


# 3. Drought (Red -> Green -> Blue). No clipping.
target_3 = r"""                        <div className="w-full h-3 bg-surface-container rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute inset-0 bg-gradient-to-r from-blue-500 via-yellow-500 to-rose-600"
                            style={{ clipPath: `inset(0 ${100 - Math.min(100, ((inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100) / 150) * 100)}% 0 0)` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, ((inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100) / 150) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""

new_3 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-rose-500 via-yellow-400 to-blue-600 rounded-full border border-glass-border relative">
                          <div 
                            className="absolute top-0 bottom-0 w-2 bg-white shadow-[0_0_10px_white] z-10 rounded-full border border-gray-300"
                            style={{ left: `calc(${Math.min(100, ((inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100) / 150) * 100)}% - 4px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_3, new_3)

# 4. Algae (Blue -> Yellow -> Emerald). No clipping.
target_4 = r"""                        <div className="w-full h-3 bg-surface-container rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute inset-0 bg-gradient-to-r from-emerald-600 via-yellow-400 to-green-300"
                            style={{ clipPath: `inset(0 ${100 - Math.min(100, ((100 - 20) / 100) * 100)}% 0 0)` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, ((100 - 20) / 100) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""

new_4 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-blue-400 via-yellow-400 to-emerald-600 rounded-full border border-glass-border relative">
                          <div 
                            className="absolute top-0 bottom-0 w-2 bg-white shadow-[0_0_10px_white] z-10 rounded-full border border-gray-300"
                            style={{ left: `calc(${Math.min(100, ((100 - 20) / 100) * 100)}% - 4px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_4, new_4)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Final meter fix applied!")
