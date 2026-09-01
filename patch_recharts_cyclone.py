import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# 1. Add imports
if "from 'recharts'" not in content:
    import_statement = "import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine, Area, ComposedChart } from 'recharts';\n"
    content = content.replace('import "maplibre-gl/dist/maplibre-gl.css";', 'import "maplibre-gl/dist/maplibre-gl.css";\n' + import_statement)

# 2. Replace the image in the Cyclone card with the Recharts graph
target_img = r"""                    <div className="flex items-center gap-4 mb-4">
                      <div className="w-16 h-16 bg-surface-container rounded-lg overflow-hidden border border-glass-border relative">
                        <img src="/simulations/sim_cyclone.png" className="w-full h-full object-cover opacity-80" />
                        <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent"></div>
                      </div>
                      <div className="flex-1">"""

new_graph = r"""                    <div className="flex flex-col gap-2 mb-4">
                      <div className="h-48 bg-white rounded-lg border border-glass-border overflow-hidden p-2 shadow-inner pointer-events-none">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart
                            layout="vertical"
                            data={inferResults?.depth_series?.filter((d:any) => d.depth_m <= 150) || []}
                            margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#e5e7eb" />
                            <XAxis type="number" domain={[20, 'auto']} orientation="top" tick={{fontSize: 10}} />
                            <YAxis type="number" dataKey="depth_m" reversed={true} tick={{fontSize: 10}} />
                            <ReferenceLine x={26} stroke="#ef4444" strokeWidth={1} strokeDasharray="4 4" label={{ position: 'insideBottomRight', value: '26°C', fill: '#ef4444', fontSize: 9, fontWeight: 'bold' }} />
                            <Line type="monotone" dataKey="tribreed_degC" stroke="#f43f5e" strokeWidth={2.5} dot={false} isAnimationActive={true} animationDuration={1200} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="flex-1 flex justify-between items-end mt-2">"""

content = content.replace(target_img, new_graph)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Cyclone Recharts injected successfully!")
