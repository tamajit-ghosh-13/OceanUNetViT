import re

with open("frontend/src/app/page.tsx", "r") as f:
    content = f.read()

# 1. Add the sidebar button
sidebar_target = r"""          <button
            onClick={() => setActiveTab("reconstruction")}
            className={`w-full flex items-center gap-3 px-3 py-2.5  font-body-lg text-body-lg font-medium transition-all cursor-pointer ${
              activeTab === "reconstruction"
                ? "bg-primary text-on-surface font-bold shadow-lg shadow-md"
                : "text-text-muted hover:text-on-surface hover:bg-surface-container-high/50"
            }`}
          >
            <Sliders className="w-4 h-4" />
            <span>3D Interactive Depth Slider</span>
          </button>"""

sidebar_injection = r"""          <button
            onClick={() => setActiveTab("reconstruction")}
            className={`w-full flex items-center gap-3 px-3 py-2.5  font-body-lg text-body-lg font-medium transition-all cursor-pointer ${
              activeTab === "reconstruction"
                ? "bg-primary text-on-surface font-bold shadow-lg shadow-md"
                : "text-text-muted hover:text-on-surface hover:bg-surface-container-high/50"
            }`}
          >
            <Sliders className="w-4 h-4" />
            <span>3D Interactive Depth Slider</span>
          </button>

          <button
            onClick={() => setActiveTab("disaster_risk")}
            className={`w-full flex items-center gap-3 px-3 py-2.5  font-body-lg text-body-lg font-medium transition-all cursor-pointer ${
              activeTab === "disaster_risk"
                ? "bg-primary text-on-surface font-bold shadow-lg shadow-md bg-rose-500/10 text-rose-500 border-r-4 border-rose-500"
                : "text-text-muted hover:text-rose-500 hover:bg-rose-500/5"
            }`}
          >
            <ShieldAlert className="w-4 h-4" />
            <span>Disaster Risk Intelligence</span>
          </button>"""

content = content.replace(sidebar_target, sidebar_injection)

# 2. Extract the disaster block from live_infer and wrap it in its own tab
block_target = r"""              {/* DISASTER RISK & ENVIRONMENTAL PHYSICS MODULE */}
              <div className="mt-8 bg-surface-white border border-glass-border p-7 rounded-xl shadow-lg">"""

block_injection = r"""            </div>
          )}

          {/* TAB: DISASTER RISK INTELLIGENCE */}
          {activeTab === "disaster_risk" && (
            <div className="space-y-6">
              {/* DISASTER RISK & ENVIRONMENTAL PHYSICS MODULE */}
              <div className="bg-surface-white border border-glass-border p-7 rounded-xl shadow-lg">"""

content = content.replace(block_target, block_injection)

with open("frontend/src/app/page.tsx", "w") as f:
    f.write(content)
print("Disaster Tab extracted successfully!")
