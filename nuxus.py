
# @title 🧬 **Launch RJ-Project 109: NEXUS**

# ============================================================================
# RJ-NEXUS V2.3: Real Success Discovery
# ============================================================================

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from sklearn.neural_network import MLPRegressor
import random
import itertools
import csv
import json
import os
from datetime import datetime
import base64


import requests
import py3Dmol
from rdkit import Chem
from rdkit.Chem import AllChem

print("🔥 RJ-NEXUS V2.3 BOOTING...")

# ============================================================================
# SECTION 1: AI BRAIN & DATABASE
# ============================================================================

class DrugDatabase:
    DRUGS = {
        "Aspirin": [180.16, 1.2, 40, 30, "Pain Reliever"],
        "Paracetamol": [151.16, 0.5, 35, 25, "Pain Reliever"],
        "Remdesivir": [602.60, 2.2, 120, 85, "Antiviral"],
        "Ibuprofen": [206.30, 3.5, 50, 45, "Anti-inflammatory"],
        "Dexamethasone": [392.47, 1.8, 65, 55, "Steroid"],
        "Hydroxychloroquine": [335.87, 3.6, 55, 60, "Antimalarial"],
        "Azithromycin": [748.98, 4.0, 70, 65, "Antibiotic"],
        "Lopinavir": [628.80, 5.9, 90, 78, "Antiviral"],
        "Favipiravir": [157.10, -0.5, 45, 40, "Antiviral"],
        "Molnupiravir": [329.31, -1.2, 85, 70, "Antiviral"],
        "Paxlovid": [499.53, 2.1, 110, 88, "Antiviral"],
        "Oseltamivir": [312.40, 1.1, 60, 52, "Antiviral"],
    }

    @classmethod
    def get_training_data(cls):
        X_train = []
        Y_train = []
        for drug_data in cls.DRUGS.values():
            X_train.append([drug_data[0], drug_data[1]])
            Y_train.append([drug_data[2], drug_data[3]])
        return np.array(X_train), np.array(Y_train)

    @classmethod
    def parse_drug_input(cls, drug_input_string):
        """Parse multiple drug formats"""
        drugs_data = []
        entries = [x.strip() for x in drug_input_string.split(',') if x.strip()]

        for entry in entries:
            if entry in cls.DRUGS:
                data = cls.DRUGS[entry]
                drugs_data.append({
                    'name': entry,
                    'weight': data[0],
                    'xlogp': data[1]
                })
            elif entry.replace('.', '').isdigit():
                weight = float(entry)
                drugs_data.append({
                    'name': f'Custom-{weight}',
                    'weight': weight,
                    'xlogp': 2.0
                })
            elif any(char in entry for char in ['=', '#', '(', ')', 'C', 'N', 'O']):
                c_count = entry.count('C') + entry.count('c')
                n_count = entry.count('N') + entry.count('n')
                o_count = entry.count('O') + entry.count('o')
                est_weight = (c_count * 12) + (n_count * 14) + (o_count * 16) + 20
                est_xlogp = min(max((c_count * 0.2) - (o_count * 0.3), -2), 6)

                drugs_data.append({
                    'name': f'SMILES-{len(drugs_data)+1}',
                    'weight': est_weight,
                    'xlogp': est_xlogp
                })

        return drugs_data

    @classmethod
    def mix_drugs(cls, drugs_data, mix_ratio):
        """Mix drugs based on ratio (0-100)"""
        if not drugs_data or len(drugs_data) == 1:
            return drugs_data[0] if drugs_data else None

        avg_weight = np.mean([d['weight'] for d in drugs_data])
        avg_xlogp = np.mean([d['xlogp'] for d in drugs_data])

        if mix_ratio > 50:
            boost_factor = 1 + ((mix_ratio - 50) / 100)
            avg_weight *= boost_factor
            avg_xlogp = min(avg_xlogp * 1.1, 5.0)

        mixed_name = " + ".join([d['name'] for d in drugs_data][:3])
        if len(drugs_data) > 3:
            mixed_name += f" + {len(drugs_data)-3} more"

        return {
            'name': f"Mix({mixed_name})",
            'weight': avg_weight,
            'xlogp': avg_xlogp
        }

X_train, Y_train = DrugDatabase.get_training_data()
ai_brain = MLPRegressor(hidden_layer_sizes=(20, 15, 10), max_iter=10000, random_state=42)
ai_brain.fit(X_train, Y_train)
print("✅ AI Ready!\n")

# ============================================================================
# SECTION 2: CORE CLASSES
# ============================================================================

class Virus:
    def __init__(self, name, x, y, z, health, defense):
        self.name = name
        self.pos = [x, y, z]
        self.health = health
        self.max_health = health
        self.defense = defense
        self.has_mutated = False

class Organ:
    def __init__(self, name, x, y, z, health, risk_threshold, color):
        self.name = name
        self.pos = [x, y, z]
        self.health = health
        self.max_health = health
        self.risk_threshold = risk_threshold
        self.color = color
        self.damaged = False

class Drug:
    def __init__(self, name, x, y, z, speed, strength, affinity, toxicity_radius):
        self.name = name
        self.pos = [x, y, z]
        self.speed = speed
        self.strength = strength
        self.max_strength = strength
        self.affinity = affinity
        self.toxicity_radius = toxicity_radius
        self.path = [[x, y, z]]
        self.is_active = False
        self.total_damage_dealt = 0

class BloodCell:
    def __init__(self, name, x, y, z, radius, cell_type):
        self.name = name
        self.pos = [x, y, z]
        self.radius = radius
        self.cell_type = cell_type

# ============================================================================
# SECTION 3: PHYSICS ENGINE
# ============================================================================

def get_distance(pos1, pos2):
    return math.sqrt((pos2[0]-pos1[0])**2 + (pos2[1]-pos1[1])**2 + (pos2[2]-pos1[2])**2)

def move_towards_target(drug, target):
    for i in range(3):
        if drug.pos[i] < target.pos[i]:
            drug.pos[i] = min(drug.pos[i] + drug.speed, target.pos[i])
        elif drug.pos[i] > target.pos[i]:
            drug.pos[i] = max(drug.pos[i] - drug.speed, target.pos[i])

    if len(drug.path) < 100:
        drug.path.append(list(drug.pos))

# ============================================================================
# SECTION 4: SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    def __init__(self, mode, pdb_id, start_vitality, drugs_config):
        self.mode = mode
        self.pdb_id = pdb_id
        self.vitality = start_vitality
        self.max_vitality = start_vitality
        self.drugs = []
        self.viruses = []
        self.organs = []
        self.blood_cells = []
        self.tick = 0
        self.max_ticks = 100
        self.patient_alive = True
        self.log = []
        self.vitality_history = [start_vitality]
        self.virus_health_history = []

        self._initialize_drugs(drugs_config)
        self._initialize_environment()

    def _initialize_drugs(self, drugs_config):
        for drug_cfg in drugs_config:
            drug = Drug(
                name=drug_cfg["name"],
                x=0, y=0, z=0,
                speed=4.0,
                strength=drug_cfg["strength"],
                affinity=drug_cfg["affinity"],
                toxicity_radius=drug_cfg.get("toxicity", 4.0)
            )
            self.drugs.append(drug)

    def _initialize_environment(self):
        self.viruses = [Virus(f"Virus-{self.pdb_id.upper()}", 15, 20, 18, 200, 80)]

        self.organs = [
            Organ("Heart", 5, 5, 5, 100, 80.0, "#FF6B6B"),
            Organ("Liver", 2, 18, 4, 100, 95.0, "#8B4513"),
            Organ("Kidney", 12, 8, 12, 100, 85.0, "#D2691E"),
            Organ("Lung", 8, 15, 8, 100, 80.0, "#FFB6C1")
        ]

        self.blood_cells = [
            BloodCell("RBC-1", 8, 12, 10, 3.5, "RBC"),
            BloodCell("RBC-2", 15, 18, 15, 4.0, "RBC"),
            BloodCell("RBC-3", 10, 5, 20, 3.0, "RBC"),
            BloodCell("WBC-1", 6, 10, 8, 2.5, "WBC"),
            BloodCell("WBC-2", 14, 14, 16, 2.8, "WBC")
        ]

    def _add_log(self, message):
        self.log.append(f"[T{self.tick:02d}] {message}")

    def run(self):
        self._add_log(f"=== RJ-NEXUS [MUTATED MODE] ===")
        self._add_log(f"Target: {self.pdb_id} | Vitality: {self.vitality}%\n")

        while self.tick < self.max_ticks and self.patient_alive and len(self.viruses) > 0:
            self.tick += 1

            vitality_loss = random.uniform(2.5, 5.0)
            self.vitality -= vitality_loss

            if self.vitality <= 0:
                self.vitality = 0
                self._add_log("💀 PATIENT VITALITY DEPLETED!")
                self.patient_alive = False
                self.vitality_history.append(0)
                self.virus_health_history.append(sum([v.health for v in self.viruses]))
                break

            for drug in self.drugs:
                if drug.strength <= 0 or not self.viruses:
                    continue

                target_virus = min(self.viruses, key=lambda v: get_distance(drug.pos, v.pos))
                move_towards_target(drug, target_virus)

                if self.mode == "mutated":
                    acid_dmg = random.uniform(1.5, 3.0)
                    drug.strength -= acid_dmg

                for organ in self.organs:
                    if get_distance(drug.pos, organ.pos) <= drug.toxicity_radius:
                        if drug.affinity > organ.risk_threshold:
                            organ.health = 0
                            organ.damaged = True
                            self.vitality = 0 # 🎯 FIX: Patient vitality goes to 0 instantly
                            self._add_log(f"🚨 {organ.name} failure! System Collapse.")
                            self.patient_alive = False
                            break

                if not self.patient_alive:
                    break

                for cell in self.blood_cells:
                    if get_distance(drug.pos, cell.pos) <= cell.radius:
                        drug.strength -= 3
                        drug.pos[0] += random.uniform(-0.5, 0.5)
                        drug.pos[1] += random.uniform(-0.5, 0.5)

                dist = get_distance(drug.pos, target_virus.pos)
                if dist <= 3.0:
                    if not drug.is_active:
                        drug.is_active = True
                        self._add_log(f"🎯 {drug.name} locked onto virus!")

                    damage = drug.strength * min(drug.affinity / target_virus.defense, 1.2)
                    target_virus.health -= damage
                    drug.total_damage_dealt += damage

                    if target_virus.health <= 0:
                        target_virus.health = 0
                        self._add_log(f"✅ Virus DESTROYED!")

            if not self.patient_alive:
                # 🎯 FIX: Record the 0 before breaking out completely
                self.vitality_history.append(0)
                self.virus_health_history.append(sum([v.health for v in self.viruses]))
                break

            self.viruses = [v for v in self.viruses if v.health > 0]

            self.vitality_history.append(max(0, self.vitality))
            self.virus_health_history.append(sum([v.health for v in self.viruses]))

        if len(self.viruses) == 0 and self.patient_alive:
            self._add_log(f"\n🏆 SUCCESS! Patient saved at {self.vitality:.1f}%!")
            return "SUCCESS"
        else:
            self._add_log(f"\n❌ FAILED.")
            return "FAILED"

# ============================================================================
# SECTION 5: VISUALIZATION
# ============================================================================

def create_3d_trajectory_map(sim_engine):
    fig = go.Figure()

    for v in sim_engine.viruses:
        fig.add_trace(go.Scatter3d(
            x=[v.pos[0]], y=[v.pos[1]], z=[v.pos[2]],
            mode='markers+text',
            marker=dict(size=18, color='purple', symbol='diamond',
                       line=dict(color='darkviolet', width=2)),
            text=[v.name],
            textposition="top center",
            name=v.name,
            hovertemplate=f"<b>{v.name}</b><br>Health: {v.health:.0f}<extra></extra>"
        ))

    for organ in sim_engine.organs:
        u = np.linspace(0, 2 * np.pi, 20)
        v_range = np.linspace(0, np.pi, 20)
        x_organ = 2 * np.outer(np.cos(u), np.sin(v_range)) + organ.pos[0]
        y_organ = 2 * np.outer(np.sin(u), np.sin(v_range)) + organ.pos[1]
        z_organ = 2 * np.outer(np.ones(np.size(u)), np.cos(v_range)) + organ.pos[2]

        fig.add_trace(go.Surface(
            x=x_organ, y=y_organ, z=z_organ,
            colorscale=[[0, organ.color], [1, organ.color]],
            showscale=False,
            opacity=0.6 if not organ.damaged else 0.3,
            name=organ.name,
            hovertemplate=f"<b>{organ.name}</b><br>Health: {organ.health:.0f}%<extra></extra>"
        ))

    for cell in sim_engine.blood_cells:
        if cell.cell_type == "RBC":
            color = 'rgba(220, 20, 60, 0.3)'
            border_color = 'darkred'
        else:
            color = 'rgba(255, 255, 255, 0.4)'
            border_color = 'lightgray'

        fig.add_trace(go.Scatter3d(
            x=[cell.pos[0]], y=[cell.pos[1]], z=[cell.pos[2]],
            mode='markers',
            marker=dict(
                size=cell.radius*8,
                color=color,
                line=dict(color=border_color, width=2)
            ),
            name=cell.name,
            hovertemplate=f"<b>{cell.name}</b><br>Type: {cell.cell_type}<extra></extra>"
        ))

    for idx, drug in enumerate(sim_engine.drugs):
        if len(drug.path) > 1:
            fig.add_trace(go.Scatter3d(
                x=[p[0] for p in drug.path],
                y=[p[1] for p in drug.path],
                z=[p[2] for p in drug.path],
                mode='lines+markers',
                line=dict(color='cyan', width=4),
                marker=dict(size=4, color='blue'),
                name=drug.name,
                hovertemplate=f"<b>{drug.name}</b><br>Damage: {drug.total_damage_dealt:.1f}<extra></extra>"
            ))

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        template="plotly_dark",
        scene=dict(
            bgcolor="#000000",
            xaxis=dict(showbackground=False, gridcolor='#333'),
            yaxis=dict(showbackground=False, gridcolor='#333'),
            zaxis=dict(showbackground=False, gridcolor='#333'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
        ),
        title=dict(
            text=f"🧬 3D Combat Trajectory - MUTATED Mode",
            font=dict(size=16, color='cyan')
        ),
        height=500
    )

    return fig

def create_vitality_graph(sim_engine):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("🫀 Patient Vitality", "🦠 Virus Health"),
        vertical_spacing=0.12
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(sim_engine.vitality_history))),
            y=sim_engine.vitality_history,
            mode='lines+markers',
            line=dict(color='lime', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)',
            marker=dict(size=6, color='green'),
            name='Vitality'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(len(sim_engine.virus_health_history))),
            y=sim_engine.virus_health_history,
            mode='lines+markers',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)',
            marker=dict(size=6, color='darkred'),
            name='Virus'
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Simulation Tick", row=2, col=1)
    fig.update_yaxes(title_text="Vitality (%)", row=1, col=1)
    fig.update_yaxes(title_text="Health", row=2, col=1)

    fig.update_layout(
        height=450,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

# 🎯 NEW: Force Meter Generator (RJ)
def generate_meter_html(affinity):
    # Safe limit is 60, but let's map it smoothly up to 100%
    lock_percent = min(100, max(0, int((affinity / 60.0) * 100)))

    if lock_percent >= 81:
        bar_color, status_text = "#00FF00", "PERFECT DOCK (Core Penetration)"
    elif lock_percent >= 61:
        bar_color, status_text = "#8BC34A", "NEAR-MISS (Sub-optimal Binding)"
    elif lock_percent >= 41:
        bar_color, status_text = "#FFC107", "SHALLOW BIND (Trapped in Pocket)"
    elif lock_percent >= 16:
        bar_color, status_text = "#FF9800", "SURFACE GLITCH (Outer Skin Lock)"
    else:
        bar_color, status_text = "#F44336", "DEFLECTED (Total Rejection)"

    return f"""
    <div style="font-family: sans-serif; padding: 12px; background: rgba(25, 25, 30, 0.9); border-radius: 8px; border: 1px solid {bar_color}; margin-bottom: 12px; box-shadow: 0px 0px 15px {bar_color}40;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong style="font-size: 14px; color: #fff;">🧬 DOCKING FORCE METER</strong>
            <strong style="font-size: 15px; color: {bar_color};">Force: {lock_percent}%</strong>
        </div>
        <div style="width: 100%; background-color: #222; border-radius: 6px; overflow: hidden; border: 1px solid #444;">
            <div style="width: {lock_percent}%; background-color: {bar_color}; height: 26px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; transition: width 1s;">
                {status_text}
            </div>
        </div>
    </div>
    """

# 🎯 UPDATED: 3D Viewer with Physical Deflection Logic
def create_animated_protein_viewer(pdb_id, affinity=0):
    try:
        raw_pdb = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb").text
        x_c, y_c, z_c = [], [], []
        for line in raw_pdb.split('\n'):
            if line.startswith("ATOM"):
                try:
                    x_c.append(float(line[30:38]))
                    y_c.append(float(line[38:46]))
                    z_c.append(float(line[46:54]))
                except: pass
        cx = sum(x_c)/len(x_c) if x_c else 0
        cy = sum(y_c)/len(y_c) if y_c else 0
        cz = sum(z_c)/len(z_c) if z_c else 0

        smiles_weak = "CC(=O)OC1=CC=CC=C1C(=O)O"
        smiles_strong = "CCC(CC)COC(=O)[C@H](C)NP(=O)(OCC1C(C(C(O1)(C#N)C2=CC=C3N2N=CN=C3N)O)O)OC4=CC=CC=C4"

        mol1 = Chem.AddHs(Chem.MolFromSmiles(smiles_weak))
        AllChem.EmbedMolecule(mol1, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol1)
        conf1 = mol1.GetConformer()
        orig1 = [conf1.GetAtomPosition(i) for i in range(mol1.GetNumAtoms())]

        mol2 = Chem.AddHs(Chem.MolFromSmiles(smiles_strong))
        AllChem.EmbedMolecule(mol2, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol2)
        conf2 = mol2.GetConformer()
        orig2 = [conf2.GetAtomPosition(i) for i in range(mol2.GetNumAtoms())]

        lock_percent = min(100, max(0, int((affinity / 60.0) * 100)))

        # 🎯 PHYSICAL DEFLECTION LOGIC
        pos_start = (cx + 50, cy + 50, cz + 50)
        pos_mid = (cx + 15, cy + 15, cz + 15)

        if lock_percent <= 15:
            # Deflected (Bounces away into space)
            pos_end = (cx - 30, cy + 30, cz - 30)
        elif lock_percent <= 25:
            # Passes through but slowly deflects
            pos_end = (cx - 15, cy - 15, cz - 15)
        elif lock_percent <= 40:
            # Surface Glitch (Stops right outside)
            pos_end = (cx + 8, cy + 8, cz + 8)
        elif lock_percent <= 60:
            # Shallow Bind (Stops halfway inside)
            pos_end = (cx + 4, cy + 4, cz + 4)
        elif lock_percent <= 80:
            # Near-Miss (Almost center, slightly off)
            pos_end = (cx + 1.5, cy + 1.5, cz + 1.5)
        else:
            # Perfect Dock (God-Mode)
            pos_end = (cx, cy, cz)

        frames_per_stage = 20
        trajectory_sdf = ""

        for step in range(frames_per_stage):
            t = step / frames_per_stage
            curr_x = pos_start[0] + (pos_mid[0] - pos_start[0]) * t
            curr_y = pos_start[1] + (pos_mid[1] - pos_start[1]) * t
            curr_z = pos_start[2] + (pos_mid[2] - pos_start[2]) * t
            for i in range(mol1.GetNumAtoms()):
                conf1.SetAtomPosition(i, Chem.rdGeometry.Point3D(orig1[i].x + curr_x, orig1[i].y + curr_y, orig1[i].z + curr_z))
            trajectory_sdf += Chem.MolToMolBlock(mol1) + "$$$$\n"

        for step in range(frames_per_stage + 1):
            t = step / frames_per_stage
            eased_t = t * (2 - t)
            curr_x = pos_mid[0] + (pos_end[0] - pos_mid[0]) * eased_t
            curr_y = pos_mid[1] + (pos_end[1] - pos_mid[1]) * eased_t
            curr_z = pos_mid[2] + (pos_end[2] - pos_mid[2]) * eased_t
            for i in range(mol2.GetNumAtoms()):
                conf2.SetAtomPosition(i, Chem.rdGeometry.Point3D(orig2[i].x + curr_x, orig2[i].y + curr_y, orig2[i].z + curr_z))
            trajectory_sdf += Chem.MolToMolBlock(mol2) + "$$$$\n"

        viewer = py3Dmol.view(width="100%", height="100%", query=f"pdb:{pdb_id.lower()}")
        viewer.setBackgroundColor('white')
        viewer.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum', 'opacity': 0.7}})
        viewer.addModelsAsFrames(trajectory_sdf, 'sdf')
        viewer.setStyle({'model': 1}, {'stick': {'colorscheme': 'cyanCarbon', 'radius': 0.3}, 'sphere': {'radius': 0.4}})
        viewer.animate({'reps': 1, 'interval': 40})
        viewer.zoomTo()

        html_code = viewer._make_html()
        b64_html = base64.b64encode(html_code.encode('utf-8')).decode('utf-8')

        iframe_html = f"""
        <div style="border: 3px solid #00FF00; border-radius: 10px; overflow: hidden; background: black; box-shadow: 0px 0px 15px rgba(0, 255, 0, 0.5);">
            <iframe
                src="data:text/html;base64,{b64_html}"
                style="width: 100%; height: 460px; border: none;"
                sandbox="allow-scripts allow-same-origin">
            </iframe>
            <div style="text-align: center; background: #111; padding: 8px;">
                <span style="color: lime; font-weight: bold; font-size: 14px;">
                    🧬 LIVE MUTATION: {pdb_id.upper()} | 🔄 AI Mid-Air Shape-Shifting Active
                </span>
            </div>
        </div>
        """
        return iframe_html

    except Exception as e:
        return "<div>Error generating 3D view</div>"

def generate_meter_html(affinity):
    # Convert Affinity (Max Safe is ~60) to a Percentage (0-100%)
    lock_percent = min(100, max(0, int((affinity / 60.0) * 100)))

    if lock_percent >= 80: bar_color, status_text = "#4CAF50", "EXCELLENT LOCK (Permanent Binding)"
    elif lock_percent >= 40: bar_color, status_text = "#FF9800", "MODERATE LOCK (Might slip out)"
    else: bar_color, status_text = "#F44336", "WEAK LOCK (Rejected by virus)"

    return f"""
    <div style="font-family: sans-serif; padding: 12px; background: rgba(25, 25, 30, 0.9); border-radius: 8px; border: 1px solid {bar_color}; margin-bottom: 12px; box-shadow: 0px 0px 15px {bar_color}40;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <strong style="font-size: 14px; color: #fff;">🧬 LOCKING FORCE METER</strong>
            <strong style="font-size: 15px; color: {bar_color};">Affinity: {affinity:.2f}</strong>
        </div>
        <div style="width: 100%; background-color: #222; border-radius: 6px; overflow: hidden; border: 1px solid #444;">
            <div style="width: {lock_percent}%; background-color: {bar_color}; height: 26px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; transition: width 1s;">
                {lock_percent}% FORCE ACHIEVED
            </div>
        </div>
        <p style="margin: 8px 0 0 0; color: {bar_color}; font-weight: bold; text-align: center; font-size: 14px;">🔥 STATUS: {status_text}</p>
    </div>
    """

# ============================================================================
# SECTION 6: MAIN EXECUTION (WITH HIGH-THROUGHPUT MATRIX & AUTO FILTER FIXED!)
# ============================================================================

def run_mutated_experimental(drug_input, pdb_id, vitality, mix_ratio, exp_mode, progress=gr.Progress()):
    progress(0, desc="Processing DataMatrix...")

    # 1. MANUAL MODE
    if exp_mode == "Manual":
        drugs_data = DrugDatabase.parse_drug_input(drug_input)
        if not drugs_data:
            return "❌ No valid compounds found!", None, None, None, None, None, None

        progress(0.3, desc="Synthesizing compound matrix...")
        mixed_drug = DrugDatabase.mix_drugs(drugs_data, mix_ratio)
        pred = ai_brain.predict([[mixed_drug['weight'], mixed_drug['xlogp']]])[0]

        drug_config = {"name": mixed_drug['name'], "strength": pred[0], "affinity": pred[1], "toxicity": 4.0}
        sim = SimulationEngine("mutated", pdb_id, vitality, [drug_config])
        outcome = sim.run()

        report = f"""=== CLINICAL SIMULATION REPORT (MANUAL) ===
Active Ingredients: {len(drugs_data)}
Synergy Ratio: {mix_ratio}%

Compound Profile:
- Formulation: {mixed_drug['name']}
- Molecular Weight: {mixed_drug['weight']:.2f} g/mol
- XLogP (Lipophilicity): {mixed_drug['xlogp']:.2f}
- Antiviral Potency: {pred[0]:.2f}
- Binding Affinity: {pred[1]:.2f}

Clinical Outcome: {outcome}
Host Survivability: {sim.vitality:.1f}%"""

        log_text = report + "\n\n" + "\n".join(sim.log)
        meter_ui = generate_meter_html(pred[1])
        map_3d = create_3d_trajectory_map(sim)
        vitality_graph = create_vitality_graph(sim)
        protein_viewer = create_animated_protein_viewer(pdb_id, pred[1])
        return meter_ui, log_text, map_3d, vitality_graph, protein_viewer, None, None

    # 2. MEGA-BATCH MODE (HIGH-THROUGHPUT MATRIX SCREENING)
    elif exp_mode == "Mega-Batch":
        drugs_data = DrugDatabase.parse_drug_input(drug_input)
        if not drugs_data:
            return "❌ No valid compounds found!", None, None, None, None, None, None

        progress(0.1, desc="Generating Formulation Matrix...")
        all_combos = []
        for r in range(1, len(drugs_data) + 1):
            all_combos.extend(list(itertools.combinations(drugs_data, r)))

        test_ratios = [5, 15, 25, 40, 55, 75, 100]
        total_tests = len(all_combos) * len(test_ratios)
        results = []

        test_count = 0
        for combo in all_combos:
            for ratio in test_ratios:
                test_count += 1
                progress(0.2 + (0.7 * (test_count/total_tests)), desc=f"Screening Compound {test_count}/{total_tests}...")

                mixed_drug = DrugDatabase.mix_drugs(list(combo), ratio)
                pred = ai_brain.predict([[mixed_drug['weight'], mixed_drug['xlogp']]])[0]

                drug_config = {"name": mixed_drug['name'], "strength": pred[0], "affinity": pred[1], "toxicity": 4.0}
                sim = SimulationEngine("mutated", pdb_id, vitality, [drug_config])
                sim.run()

                if sim.patient_alive and len(sim.viruses) == 0:
                    status = "PASSED: Complete Viral Eradication"
                elif not sim.patient_alive:
                    status = "FATAL: Systemic Toxicity (Organ Failure)"
                else:
                    status = "SUBOPTIMAL: Insufficient Efficacy"

                results.append({
                    'rank': 0, 'name': mixed_drug['name'], 'ratio': ratio,
                    'strength': pred[0], 'affinity': pred[1], 'status': status, 'vitality': sim.vitality, 'sim': sim
                })

        progress(0.9, desc="Compiling Research Data...")
        results.sort(key=lambda x: (x['status'].startswith("PASSED"), x['vitality'], x['strength']), reverse=True)
        for i, res in enumerate(results): res['rank'] = i + 1

        winner = results[0]
        final_sim = winner['sim']

        csv_path = "HTS_Screening_Results.csv"
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Efficacy Rank", "Compound Formulation", "Synergy Concentration (%)", "Antiviral Potency Score", "Cytotoxicity / Affinity", "Clinical Simulation Outcome", "Host Survivability (%)"])
            for r in results:
                writer.writerow([r['rank'], r['name'], f"{r['ratio']}%", round(r['strength'],2), round(r['affinity'],2), r['status'], round(r['vitality'],1)])

        txt_path = "Clinical_Trial_Log.txt"
        with open(txt_path, mode='w', encoding='utf-8') as f:
            f.write(f"=== HIGH-THROUGHPUT SCREENING (HTS) LOG ===\nTarget Pathogen: {pdb_id}\nOptimal Formulation: {winner['name']}\nSynergy Target: {winner['ratio']}%\n\n")
            f.write("\n".join(final_sim.log))

        report = f"""=== HTS MEGA-BATCH REPORT ===
Total Formulations Screened: {total_tests}
🏆 OPTIMAL COMPOUND: {winner['name']}
🧪 PEAK SYNERGY CONCENTRATION: {winner['ratio']}%
Host Survivability: {winner['vitality']:.1f}%
Trial Status: {winner['status']}"""

        log_text = report + "\n\n" + "\n".join(final_sim.log)
        meter_ui = generate_meter_html(winner['affinity'])
        map_3d = create_3d_trajectory_map(final_sim)
        vitality_graph = create_vitality_graph(final_sim)
        protein_viewer = create_animated_protein_viewer(pdb_id, winner['affinity'])
        return meter_ui, log_text, map_3d, vitality_graph, protein_viewer, csv_path, txt_path

    # 3. AUTO DISCOVERY MODE
    else:
        progress(0, desc="Initiating AI Discovery Protocol...")
        num_initial_candidates = 5000
        w_arr = np.random.uniform(50, 2000, num_initial_candidates)
        x_arr = np.random.uniform(-5.0, 10.0, num_initial_candidates)
        predictions = ai_brain.predict(np.column_stack((w_arr, x_arr)))

        safe_indices = np.where(predictions[:, 1] < 59.0)[0]
        if len(safe_indices) == 0:
            top_indices = np.argsort(predictions[:, 0])[-50:]
        else:
            safe_predictions = predictions[safe_indices]
            best_safe_idx = np.argsort(safe_predictions[:, 0])[-50:]
            top_indices = safe_indices[best_safe_idx]

        best_sim, best_vitality, best_affinity = None, -1, 0
        successful_candidates = []

        for idx, test_idx in enumerate(top_indices):
            raw_strength, raw_affinity = predictions[test_idx, 0], predictions[test_idx, 1]
            drug_config = {"name": f"Nano-Gen-{idx+1}", "strength": raw_strength, "affinity": raw_affinity, "toxicity": 3.0}

            test_sim = SimulationEngine("mutated", pdb_id, vitality, [drug_config])
            test_outcome = test_sim.run()

            if test_outcome == "SUCCESS":
                successful_candidates.append({'vitality': test_sim.vitality, 'sim': test_sim, 'affinity': raw_affinity, 'weight': w_arr[test_idx], 'xlogp': x_arr[test_idx], 'strength': raw_strength})

            if best_sim is None or test_sim.vitality > best_vitality:
                best_vitality, best_sim, best_affinity = test_sim.vitality, test_sim, raw_affinity

            if idx % 10 == 0:
                progress(0.4 + (idx / 100) * 0.5, desc=f"Evaluating Phase {idx//10 + 1}... ({len(successful_candidates)} Viable Found)")

        progress(0.9, desc="Isolating Peak Formula...")

        if successful_candidates:
            successful_candidates.sort(key=lambda x: x['vitality'], reverse=True)
            winner = successful_candidates[0]
            final_sim, final_affinity = winner['sim'], winner['affinity']

            closest_drugs = []
            for name, data in DrugDatabase.DRUGS.items():
                diff = abs(data[0] - winner['weight']) + abs(data[1] - winner['xlogp']) * 50
                closest_drugs.append((name, diff))
            closest_drugs.sort(key=lambda x: x[1])
            base_1 = closest_drugs[0][0]
            base_2 = closest_drugs[1][0]

            recipe_text = f"🧬 Molecular Architecture:\n- ~65% {base_1} traits\n- ~35% {base_2} traits\n- Synthesized for Max Efficacy"

            report = f"""=== AI DISCOVERY REPORT ===
Compounds Generated & Screened: {num_initial_candidates:,}
✅ Viable Formulations Found: {len(successful_candidates)}

🏆 OPTIMAL NANO-ENGINEERED DRUG:
- Molar Mass: {winner['weight']:.2f} g/mol
- Lipophilicity (XLogP): {winner['xlogp']:.2f}
- Antiviral Potency: {winner['strength']:.2f}
- Binding Affinity: {final_affinity:.2f}

{recipe_text}

Clinical Outcome: SUCCESS
Host Survivability: {final_sim.vitality:.1f}%"""
        else:
            final_sim, final_affinity = best_sim, best_affinity
            report = f"""=== AI DISCOVERY REPORT ===
Compounds Screened: {num_initial_candidates:,}
⚠️ No viable non-toxic formulations discovered.

Clinical Outcome: FAILED
Peak Host Survivability: {max(0, best_vitality):.1f}%"""

        progress(1.0, desc="Discovery Protocol Complete!")
        log_text = report + "\n\n" + "\n".join(final_sim.log)
        meter_ui = generate_meter_html(final_affinity)
        map_3d = create_3d_trajectory_map(final_sim)
        vitality_graph = create_vitality_graph(final_sim)
        protein_viewer = create_animated_protein_viewer(pdb_id, final_affinity)
        return meter_ui, log_text, map_3d, vitality_graph, protein_viewer, None, None

# ============================================================================
# SECTION 7: GRADIO UI
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft(), css="""
    .drug-input textarea {font-family: 'Courier New'; font-size: 13px; line-height: 1.6;}
    .info-panel {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 color: white; padding: 15px; border-radius: 10px; margin: 10px 0;}
""") as dashboard:

    gr.Markdown("""
    # 🧬 RJ-Project 109: NEXUS Engine `{Auto beta v 0.3}`
    ### 🚀 A Farewell Project by RJ
    
    > *"This is my last project. It's time to say goodbye, but the journey doesn't end here. I'm leaving the Nexus Engine to the community. Keep experimenting, keep discovering, and keep pushing the boundaries of science."*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("""
            <div class="info-panel">
                <h3 style="margin:0 0 10px 0;">💊 Drug Input Formats</h3>
                <ul style="margin:5px 0; padding-left:20px;">
                    <li>Names: <code>Aspirin, Remdesivir</code></li>
                    <li>SMILES: <code>CC(=O)OC1=CC=CC=C1</code></li>
                    <li>Weights: <code>400, 650</code></li>
                    <li>Mixed: <code>Aspirin, 450, Paxlovid</code></li>
                </ul>
                <p style="margin:10px 0 0 0; font-size:12px; opacity:0.9;">
                    <strong>AUTO MODE:</strong> Tests 50+ real simulations to find SUCCESS!
                </p>
            </div>
            """)

            drug_input_box = gr.Textbox(
                label="💊 Drug Input (comma-separated)",
                placeholder="Aspirin, Remdesivir, 450",
                value="Aspirin, Remdesivir, Paxlovid",
                lines=3,
                elem_classes=["drug-input"]
            )

            exp_mode_radio = gr.Radio(
                choices=["Manual", "Auto", "Mega-Batch"],
                value="Manual",
                label="🧪 Experimental Mode",
                info="Manual = Your drugs | Auto = AI finds SAFE SUCCESS | Mega-Batch = Professional Matrix Lab"
            )

            mix_ratio_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=50,
                step=1,
                label="🔬 Drug Mixing Intensity",
                info="0=Simple Mix | 50=Balanced | 100=Synergy Boost"
            )

            pdb_input = gr.Textbox(
                value="6LU7",
                label="🦠 Target Virus PDB Code",
                info="6LU7=COVID | 1CJC=Flu | 7BV2=Omicron"
            )

            vitality_slider = gr.Slider(
                minimum=50,
                maximum=500,
                value=240,
                step=10,
                label="🫀 Patient Starting Vitality (%)"
            )

            run_btn = gr.Button(
                "🚀 RUN MUTATED EXPERIMENT",
                variant="primary",
                size="lg"
            )

        with gr.Column(scale=2):
          
            output_meter = gr.HTML(label="🧬 Locking Force Meter")

            output_log = gr.Textbox(
                label="📋 Mission Log & Results",
                lines=10
            )

            output_trajectory = gr.Plot(label="🗺️ 3D Combat Trajectory Map")
            output_vitality = gr.Plot(label="📊 Vitality & Virus Health Graph")
            output_protein = gr.HTML(label="🧬 Live Animated Protein Viewer")

            with gr.Row():
                output_csv = gr.File(label="📊 Download CSV Leaderboard", file_count="single")
                output_txt = gr.File(label="📄 Download Mission Log", file_count="single")

    run_btn.click(
        fn=run_mutated_experimental,
        inputs=[drug_input_box, pdb_input, vitality_slider, mix_ratio_slider, exp_mode_radio],
        
        outputs=[output_meter, output_log, output_trajectory, output_vitality, output_protein, output_csv, output_txt]
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔥 NEXUS V2.3 READY!")
    print("="*60)
    print("✅ AUTO MODE: Tests 50 Real Simulations")
    print("✅ Guaranteed SUCCESS or Best Result")
    print("✅ All Visual Features Active")
    print("\n🚀 Launching...\n")

    dashboard.launch(debug=True)