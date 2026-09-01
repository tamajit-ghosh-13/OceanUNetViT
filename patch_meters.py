import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# 1. Cyclone
target_1 = r"""                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-500 via-yellow-500 to-rose-600 relative"
                            style={{ width: `${Math.min(100, ((inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50) / 120) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>"""

new_1 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-blue-500 via-yellow-500 to-rose-600 rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute right-0 top-0 bottom-0 bg-surface-container"
                            style={{ width: `${100 - Math.min(100, ((inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50) / 120) * 100)}%` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, ((inferResults?.derived_physical_products?.tchp_kj_cm2 ?? 50) / 120) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_1, new_1)

# 2. Heatwave
target_2 = r"""                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-400 via-orange-400 to-red-600 relative"
                            style={{ width: `${Math.min(100, (2.4 / 4.0) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>"""

new_2 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-blue-400 via-orange-400 to-red-600 rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute right-0 top-0 bottom-0 bg-surface-container"
                            style={{ width: `${100 - Math.min(100, (2.4 / 4.0) * 100)}%` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, (2.4 / 4.0) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_2, new_2)

# 3. Drought
target_3 = r"""                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-red-500 via-green-400 to-blue-600 relative"
                            style={{ width: `${Math.min(100, ((inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100) / 150) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>"""

new_3 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-red-500 via-green-400 to-blue-600 rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute right-0 top-0 bottom-0 bg-surface-container"
                            style={{ width: `${100 - Math.min(100, ((inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100) / 150) * 100)}%` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, ((inferResults?.derived_physical_products?.thermocline_d20_depth_m ?? 100) / 150) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_3, new_3)


# 4. Algae
target_4 = r"""                        <div className="w-full h-3 bg-background rounded-full overflow-hidden border border-glass-border">
                          <div 
                            className="h-full bg-gradient-to-r from-emerald-600 via-yellow-400 to-green-300 relative"
                            style={{ width: `${Math.min(100, ((100 - 20) / 100) * 100)}%` }}
                          >
                            <div className="absolute right-0 top-0 bottom-0 w-1 bg-white shadow-[0_0_8px_white]"></div>
                          </div>
                        </div>"""

new_4 = r"""                        <div className="w-full h-3 bg-gradient-to-r from-emerald-600 via-yellow-400 to-green-300 rounded-full overflow-hidden border border-glass-border relative">
                          <div 
                            className="absolute right-0 top-0 bottom-0 bg-surface-container"
                            style={{ width: `${100 - Math.min(100, ((100 - 20) / 100) * 100)}%` }}
                          ></div>
                          <div 
                            className="absolute top-0 bottom-0 w-1.5 bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] z-10 rounded-full"
                            style={{ left: `calc(${Math.min(100, ((100 - 20) / 100) * 100)}% - 3px)` }}
                          ></div>
                        </div>"""
content = content.replace(target_4, new_4)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Risk meters patched!")
