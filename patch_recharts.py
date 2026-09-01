import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

target = r"""                            <ResponsiveContainer width="100%" height="100%">
                              <ComposedChart
                                layout="vertical"
                                data={inferResults?.depth_series?.filter((d:any) => d.depth_m <= 150) || []}
                                margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#e5e7eb" />
                                <XAxis type="number" domain={[20, 'auto']} orientation="top" tick={{fontSize: 11}} />
                                <YAxis type="number" dataKey="depth_m" reversed={true} tick={{fontSize: 11}} />
                                <ReferenceLine x={26} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 4" label={{ position: 'insideBottomRight', value: '26°C Fuel Threshold', fill: '#ef4444', fontSize: 11, fontWeight: 'bold' }} />
                                <Line type="monotone" dataKey="tribreed_degC" stroke="#f43f5e" strokeWidth={3} dot={false} isAnimationActive={true} animationDuration={800} />
                              </ComposedChart>
                            </ResponsiveContainer>"""

new_graph = r"""                            <ResponsiveContainer width="100%" height="100%">
                              <ComposedChart
                                layout="vertical"
                                data={inferResults?.depth_series?.filter((d:any) => d.depth_m <= 300) || []}
                                margin={{ top: 10, right: 10, left: -10, bottom: 0 }}
                              >
                                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={true} stroke="#e5e7eb" />
                                <XAxis type="number" domain={[15, 35]} orientation="top" tick={{fontSize: 11}} />
                                <YAxis type="number" dataKey="depth_m" reversed={true} tick={{fontSize: 11}} domain={[0, 300]} />
                                <Tooltip 
                                  cursor={{ stroke: '#f43f5e', strokeWidth: 1, strokeDasharray: '4 4' }}
                                  contentStyle={{ backgroundColor: 'rgba(255,255,255,0.9)', borderRadius: '8px', border: '1px solid #e5e7eb', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                                  formatter={(value: any) => [`${Number(value).toFixed(2)}°C`, 'Temperature']} 
                                  labelFormatter={(label: any) => `Depth: ${label}m`}
                                />
                                <ReferenceLine x={26} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 4" label={{ position: 'insideBottomRight', value: '26°C Fuel Threshold', fill: '#ef4444', fontSize: 11, fontWeight: 'bold' }} />
                                <Line type="monotone" dataKey="tribreed_degC" stroke="#f43f5e" strokeWidth={3} dot={{r: 0}} activeDot={{r: 6, strokeWidth: 0, fill: '#f43f5e'}} isAnimationActive={true} animationDuration={800} />
                              </ComposedChart>
                            </ResponsiveContainer>"""

content = content.replace(target, new_graph)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Recharts patched!")
