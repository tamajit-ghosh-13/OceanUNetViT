import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# 1. Add imports
if "from 'recharts'" not in content:
    import_statement = "import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';\n"
    content = content.replace('import "maplibre-gl/dist/maplibre-gl.css";', 'import "maplibre-gl/dist/maplibre-gl.css";\n' + import_statement)

# 2. Replace static image in Cyclone tab
old_block = r"""                            <div 
                              className="relative group cursor-pointer border border-glass-border rounded-xl overflow-hidden bg-white shadow-sm hover:shadow-xl transition-all duration-300 flex items-center justify-center p-2 mb-5"
                              onClick={() => setSelectedDisasterImage({
                                src: inferResults.visualizations?.cyclone_sim_image || "/simulations/sim_cyclone.png",
                                title: "Cyclone Rapid Intensification Simulation",
                                subtitle: "Vertical Temperature Profile & Cumulative Subsurface UOHC Heat Potential Integration",
                                formula: "UOHC = c_p × ρ × ∫₀^D₂₆ (T(z) - 26°C) dz",
                              })}
                            >
                              <img 
                                src={inferResults.visualizations?.cyclone_sim_image || "/simulations/sim_cyclone.png"} 
                                alt="Cyclone Physics Simulation" 
                                className="w-full h-auto rounded-lg object-contain transition-transform duration-300 group-hover:scale-[1.01]" 
                              />
                              <div className="absolute top-3 right-3 bg-black/75 hover:bg-black/90 backdrop-blur-md text-white text-xs font-semibold px-3 py-1.5 rounded-lg flex items-center gap-1.5 opacity-90 group-hover:opacity-100 transition-opacity shadow-md border border-white/20">
                                <Maximize2 className="w-3.5 h-3.5 text-cyan-400" /> Expand Modal
                              </div>
                            </div>"""

new_block = r"""                            <div className="border border-glass-border rounded-xl overflow-hidden bg-background shadow-sm p-4 mb-5 h-[350px]">
                              <ResponsiveContainer width="100%" height="100%">
                                <ComposedChart
                                  layout="vertical"
                                  data={inferResults.depth_series?.filter(d => d.depth_m <= 300) || []}
                                  margin={{ top: 20, right: 30, left: 10, bottom: 10 }}
                                >
                                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#e5e7eb" />
                                  <XAxis type="number" domain={[15, 'auto']} orientation="top" tick={{fontSize: 12}} />
                                  <YAxis type="number" dataKey="depth_m" reversed={true} tick={{fontSize: 12}} />
                                  <Tooltip 
                                    formatter={(value, name) => [Number(value).toFixed(2) + '°C', name === 'tribreed_degC' ? 'Temperature' : name]} 
                                    labelFormatter={(label) => `Depth: ${label}m`}
                                  />
                                  <ReferenceLine x={26} stroke="#ef4444" strokeWidth={2} strokeDasharray="4 4" label={{ position: 'insideBottomRight', value: '26°C Fuel Threshold', fill: '#ef4444', fontSize: 11, fontWeight: 'bold' }} />
                                  <Line type="monotone" dataKey="tribreed_degC" stroke="#3b82f6" strokeWidth={3} dot={false} isAnimationActive={true} animationDuration={1200} />
                                </ComposedChart>
                              </ResponsiveContainer>
                            </div>"""

content = content.replace(old_block, new_block)

old_text = r"""                            <p><strong className="text-on-surface text-blue-400">Safe Ocean (Blue Line):</strong> Rapid thermal drop; insufficient heat engine fuel.</p>
                            <p><strong className="text-on-surface text-rose-400">Extreme Risk (Red Line):</strong> &gt;26°C warmth penetrates down to D26 = {d26.toFixed(1)}m.</p>"""

new_text = r"""                            <p><strong className="text-on-surface text-blue-500">Live Thermal Column:</strong> Automatically drawn from the AI's 3D array.</p>
                            <p><strong className="text-on-surface text-rose-500">D26 Threshold (Red Dashed):</strong> If the blue curve crosses the red threshold line, it indicates available heat engine fuel at that depth down to D26 = {d26.toFixed(1)}m.</p>"""

content = content.replace(old_text, new_text)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Cyclone UI patched successfully for Live Graph!")
