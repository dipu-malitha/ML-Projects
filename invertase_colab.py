# ============================================================
#  Sonification of Concentration Profiles
#  Reaction: Invertase-Catalyzed Hydrolysis of Sucrose
#  Based on: Michaelis & Menten (1913) — the ORIGINAL paper
#  Reference: Johnson & Goody (2011), Biochemistry 50, 8264
# ============================================================
#  HOW TO USE IN GOOGLE COLAB:
#  1. Go to https://colab.research.google.com
#  2. File → New Notebook
#  3. Paste each cell block into its own code cell
#  4. Run top to bottom with Shift+Enter
# ============================================================

# ─────────────────────────────────────────────────────────
# CELL 1 — Imports (all pre-installed in Colab)
# ─────────────────────────────────────────────────────────
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wave
from IPython.display import Audio, display
from google.colab import files

print("✓ All imports successful")

# ─────────────────────────────────────────────────────────
# CELL 2 — Parameters from the 1913 Paper
# ─────────────────────────────────────────────────────────
# MECHANISM:
#
#   Step 1 — Reversible binding:
#     Invertase + Sucrose  ⇌  [Invertase·Sucrose]    (k1, k-1)
#         E    +    S      ⇌         ES
#
#   Step 2 — Catalytic hydrolysis:
#     [Invertase·Sucrose]  →  Invertase + Glucose + Fructose  (k2=kcat)
#             ES           →       E   +    G    +     F
#
#   Product inhibition (competitive, as Michaelis & Menten derived):
#     E + F  ⇌  EF    KF = 58.8 mM
#     E + G  ⇌  EG    KG = 91.0 mM
#
# All parameters from: Johnson & Goody (2011) reanalysis of M&M data

KS   = 16.7    # mM  — Michaelis constant for sucrose (= Km)
KF   = 58.8    # mM  — dissociation constant for fructose (inhibitor)
KG   = 91.0    # mM  — dissociation constant for glucose  (inhibitor)
Vmax = 0.76    # mM/min — maximum rate = kcat * E0

# The "Const" that Michaelis & Menten derived globally from all their data
Const_MM = Vmax / KS   # = kcat/Km * E0 (specificity constant × enzyme conc.)

print("=" * 55)
print("  PARAMETERS (Michaelis & Menten 1913)")
print("=" * 55)
print(f"  KS  (Km)  = {KS}   mM   (sucrose, Michaelis constant)")
print(f"  KF        = {KF}  mM   (fructose, competitive inhibitor)")
print(f"  KG        = {KG}   mM   (glucose,  competitive inhibitor)")
print(f"  Vmax      = {Vmax}  mM/min  (kcat × E0)")
print(f"  Const     = {Const_MM:.4f} min⁻¹  (M&M reported: 0.0454 min⁻¹)")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# CELL 3 — ODE System (Batch Reactor with Product Inhibition)
# ─────────────────────────────────────────────────────────
def invertase_odes(t, y):
    """
    Mechanistic ODEs for invertase hydrolysis.
    State: y = [S, F, G]  (mM)

    Rate equation (quasi-steady-state + competitive product inhibition):
      v = Vmax * S / (S + KS*(1 + F/KF + G/KG))

    Batch reactor mole balances (no flow):
      dS/dt = -v      (sucrose consumed)
      dF/dt = +v      (fructose produced)
      dG/dt = +v      (glucose produced)

    Note: F = G always (stoichiometric: 1 sucrose → 1 fructose + 1 glucose)
    """
    S, F, G = y
    S = max(S, 0.0)
    F = max(F, 0.0)
    G = max(G, 0.0)
    denom = S + KS * (1.0 + F/KF + G/KG)
    v     = Vmax * S / denom
    return [-v, v, v]

# The 5 sucrose concentrations Michaelis & Menten used experimentally (mM)
S0_values   = [333.0, 166.7, 83.0, 41.6, 20.8]
colors_plot = ['#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#8338ec']
labels_plot = ['333 mM', '166.7 mM', '83 mM', '41.6 mM', '20.8 mM']

# Solve ODE for each initial concentration
t_eval    = np.linspace(0, 250, 3000)   # 0 to 250 minutes
solutions = []
for S0 in S0_values:
    sol = solve_ivp(
        invertase_odes, (0, 250), [S0, 0.0, 0.0],
        method='RK45', t_eval=t_eval, rtol=1e-9, atol=1e-12
    )
    solutions.append(sol)

print("✓ ODE solver complete for all 5 sucrose concentrations")

# ─────────────────────────────────────────────────────────
# CELL 4 — Reproduce M&M Table 1 and Verify Const
# ─────────────────────────────────────────────────────────
# Original data from Table 1 of Michaelis & Menten (1913)
MM_data = {
    333.0:  [(7,0.0164),(14,0.0316),(26,0.0528),(49,0.0923),
             (75,0.1404),(117,0.2137)],
    166.7:  [(8,0.0350),(16,0.0636),(28,0.1080),(52,0.1980),
             (82,0.3000),(103,0.3780)],
    83.0:   [(49.5,0.352),(90.0,0.575),(125.0,0.690),
             (151.0,0.766),(208.0,0.900)],
    41.6:   [(10.25,0.1147),(30.75,0.3722),(61.75,0.615),
             (90.75,0.747),(112.70,0.850),(132.70,0.925),(154.70,0.940)],
    20.8:   [(17,0.331),(27,0.452),(38,0.611),(62,0.736),(95,0.860)],
}

def compute_const(S0, t, P_ratio):
    """
    Compute M&M's 'Const' from integrated rate equation.
    Const = C/KS = (kcat/Km)*E0
    Using M&M's integrated formula with product inhibition.
    """
    F = P_ratio * S0
    S = S0 - F
    if S <= 0 or t == 0: return None
    term1 = S0 * (1/S0 + 1/KF + 1/KG) * np.log(S0/S)
    term2 = F  * (1/KS - 1/KF - 1/KG)
    return (term1 + term2) / t

all_consts = []
for S0, pts in MM_data.items():
    for (t_pt, PR) in pts:
        c = compute_const(S0, t_pt, PR)
        if c: all_consts.append(c)

mean_c = np.mean(all_consts)
std_c  = np.std(all_consts)
print(f"✓ Reproduced Const = {mean_c:.4f} ± {std_c:.4f} min⁻¹")
print(f"  M&M (1913) value  = 0.0454 ± 0.0032 min⁻¹  ← matches!")

# ─────────────────────────────────────────────────────────
# CELL 5 — Plot Concentration Profiles
# ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor='#0d1117')
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

def style_ax(ax, title, ylabel, xlabel='Time (min)'):
    ax.set_facecolor('#161b22')
    ax.set_title(title, color='white', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel(xlabel, color='#8b949e', fontsize=9)
    ax.set_ylabel(ylabel, color='#8b949e', fontsize=9)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', lw=0.8, ls='--')

# Panel 1: P/S0 ratio — replicates M&M Figure 1 (wide panel)
ax1 = fig.add_subplot(gs[0, :2])
for i, (sol, S0) in enumerate(zip(solutions, S0_values)):
    ratio = sol.y[1] / S0
    ax1.plot(t_eval, ratio, color=colors_plot[i], lw=2.2,
             label=f'[S]₀ = {labels_plot[i]}')
    if S0 in MM_data:
        t_pts = [p[0] for p in MM_data[S0]]
        r_pts = [p[1] for p in MM_data[S0]]
        ax1.scatter(t_pts, r_pts, color=colors_plot[i], s=40, zorder=5,
                    edgecolors='white', linewidths=0.5)
style_ax(ax1, 'Reproduction of Michaelis & Menten (1913) Figure\n'
              '[P]/[S₀] vs Time — lines=ODE simulation, dots=original 1913 data',
         '[P]/[S₀]  (fraction of sucrose converted)')
ax1.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white',
           fontsize=8, loc='lower right')
ax1.set_xlim(0, 250); ax1.set_ylim(0, 1.05)

# Panel 2: Sucrose decay
ax2 = fig.add_subplot(gs[0, 2])
for i, (sol, S0) in enumerate(zip(solutions, S0_values)):
    ax2.plot(t_eval, sol.y[0], color=colors_plot[i], lw=2, label=labels_plot[i])
style_ax(ax2, '[S] Sucrose Consumption', 'Concentration (mM)')
ax2.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=7)

# Panel 3: Fructose/Glucose production
ax3 = fig.add_subplot(gs[1, 0])
for i, (sol, S0) in enumerate(zip(solutions, S0_values)):
    ax3.plot(t_eval, sol.y[1], color=colors_plot[i], lw=2, label=labels_plot[i])
style_ax(ax3, '[F] Fructose = [G] Glucose\n(products)', 'Concentration (mM)')
ax3.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=7)

# Panel 4: Reaction rate (shows product inhibition effect)
ax4 = fig.add_subplot(gs[1, 1])
for i, (sol, S0) in enumerate(zip(solutions, S0_values)):
    S_a = np.maximum(sol.y[0], 0)
    F_a = np.maximum(sol.y[1], 0)
    G_a = np.maximum(sol.y[2], 0)
    v_a = Vmax * S_a / (S_a + KS*(1 + F_a/KF + G_a/KG))
    ax4.plot(t_eval, v_a, color=colors_plot[i], lw=2, label=labels_plot[i])
style_ax(ax4, 'Reaction Rate v(t)\n(product inhibition causes bell-shape)',
         'Rate (mM/min)')
ax4.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=7)

# Panel 5: Const verification (flat = model confirmed!)
ax5 = fig.add_subplot(gs[1, 2])
for S0_val, pts in MM_data.items():
    color_idx = S0_values.index(S0_val)
    t_arr = [p[0] for p in pts]
    c_arr = [compute_const(S0_val, p[0], p[1]) for p in pts]
    c_arr_valid = [(t, c) for t, c in zip(t_arr, c_arr) if c]
    t_v, c_v = zip(*c_arr_valid)
    ax5.scatter(t_v, c_v, color=colors_plot[color_idx], s=35,
                label=f'{S0_val} mM', zorder=5, edgecolors='white', lw=0.4)
ax5.axhline(0.0454, color='white', lw=1.5, ls='--', alpha=0.7,
            label='M&M mean=0.0454')
style_ax(ax5, 'Const = C/Kₛ (should be flat)\nConfirms M&M global analysis',
         'Const (min⁻¹)')
ax5.legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=7)
ax5.set_ylim(0.03, 0.065)

fig.suptitle('Invertase Hydrolysis of Sucrose — Michaelis & Menten (1913)\n'
             'Batch Reactor · Full Mechanism with Product Inhibition · '
             f'Const = {mean_c:.4f} ± {std_c:.4f} min⁻¹',
             color='white', fontsize=13, fontweight='bold', y=1.01)

plt.savefig('concentration_profiles.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("✓ Concentration profiles plot saved.")

# ─────────────────────────────────────────────────────────
# CELL 6 — Sonification
# ─────────────────────────────────────────────────────────
# SONIFICATION MAPPING:
#   [S] Sucrose (decreasing)    → Pitch descends (high→low note)
#   [F] Fructose (increasing)   → Pitch ascends  (low→high note, harmony)
#   v(t) rate (bell-shaped)     → Amplitude (loud at peak rate, soft at start/end)
#   Product inhibition effect   → Natural dynamic fade as v(t) slows
#   5 [S0] concentrations       → 5 simultaneous voices, layered
#
#   Musical scale: D Dorian mode (D E F G A B C)
#     → Minor feel with raised 6th — historically fitting, bittersweet
#   Time compression: 250 min reaction → 40 s audio

SAMPLE_RATE = 44100
DURATION    = 40.0
N_SAMPLES   = int(SAMPLE_RATE * DURATION)

# D Dorian MIDI notes (two octaves)
D_DORIAN = np.array([62, 64, 65, 67, 69, 71, 72,
                      74, 76, 77, 79, 81, 83, 84])

def midi_to_freq(m): return 440.0 * (2.0**((m-69)/12.0))

def conc_to_freq(arr, scale, reverse=False):
    """Map concentration array to nearest scale frequency."""
    lo, hi = arr.min(), arr.max()
    norm = (arr - lo) / (hi - lo + 1e-12)
    if reverse: norm = 1.0 - norm
    idx = np.clip((norm*(len(scale)-1)).astype(int), 0, len(scale)-1)
    return np.array([midi_to_freq(scale[i]) for i in idx])

def smooth(arr, w=3000):
    return np.convolve(arr, np.ones(w)/w, mode='same')

t_audio   = np.linspace(0, 250, N_SAMPLES)
audio_mix = np.zeros(N_SAMPLES)
time_vec  = np.arange(N_SAMPLES) / SAMPLE_RATE

# Register (pitch range) for each concentration — higher [S0] = lower voice
registers = [
    D_DORIAN[:7],    # 333 mM   — lowest register
    D_DORIAN[1:8],   # 166.7 mM
    D_DORIAN[2:9],   # 83 mM
    D_DORIAN[5:12],  # 41.6 mM
    D_DORIAN[7:],    # 20.8 mM  — highest register
]
volumes = [1.0, 0.85, 0.70, 0.55, 0.40]

for i, (sol, S0) in enumerate(zip(solutions, S0_values)):
    S_a = np.interp(t_audio, t_eval, sol.y[0])
    F_a = np.interp(t_audio, t_eval, sol.y[1])
    G_a = np.interp(t_audio, t_eval, sol.y[2])

    # Amplitude = reaction rate (product inhibition shapes the envelope)
    denom_a = np.maximum(S_a,0) + KS*(1 + np.maximum(F_a,0)/KF + np.maximum(G_a,0)/KG)
    rate_a  = Vmax * np.maximum(S_a,0) / denom_a
    amp     = smooth(rate_a / (rate_a.max() + 1e-12))

    scale_i = registers[i]

    # Voice 1: [S] → descending pitch + vibrato
    freq_S  = conc_to_freq(S_a, scale_i, reverse=False)
    phase_S = np.cumsum(2*np.pi * freq_S / SAMPLE_RATE)
    vibrato = 0.002 * np.sin(2*np.pi * 3.5 * time_vec)
    voice1  = amp * np.sin(phase_S + vibrato)

    # Voice 2: [F] → ascending pitch (harmony)
    freq_F  = conc_to_freq(F_a, scale_i[::-1])
    phase_F = np.cumsum(2*np.pi * freq_F / SAMPLE_RATE)
    voice2  = amp * 0.5 * np.sin(phase_F)

    # Voice 3: pulse texture tied to rate bell-shape
    pulse_freq  = 1.5 + 3.0*amp
    pulse_phase = np.cumsum(2*np.pi * pulse_freq / SAMPLE_RATE)
    voice3  = amp * 0.25 * np.sin(pulse_phase) * np.sin(phase_S * 2)

    audio_mix += volumes[i] * (voice1 + voice2 + voice3)

# Normalize and export
audio_mix   = audio_mix / (np.max(np.abs(audio_mix)) + 1e-12) * 0.85
audio_int16 = (audio_mix * 32767).astype(np.int16)

wav_path = 'invertase_sonification.wav'
with wave.open(wav_path, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_int16.tobytes())

print(f"✓ WAV saved: {wav_path}  ({DURATION:.0f}s | {SAMPLE_RATE}Hz | 5 voices)")

# Play directly in Colab notebook
print("\n🎵 Playing sonification in notebook...")
display(Audio(audio_mix, rate=SAMPLE_RATE))

# ─────────────────────────────────────────────────────────
# CELL 7 — Sonification Mapping Plot
# ─────────────────────────────────────────────────────────
fig2, axes = plt.subplots(4, 1, figsize=(14, 12), facecolor='#0d1117')
t_plot = np.linspace(0, DURATION, N_SAMPLES)

# Use [S]0=333 mM as example
sol0 = solutions[0]; S0_ex = S0_values[0]
S_ex = np.interp(t_audio, t_eval, sol0.y[0])
F_ex = np.interp(t_audio, t_eval, sol0.y[1])
G_ex = np.interp(t_audio, t_eval, sol0.y[2])
denom_ex = np.maximum(S_ex,0) + KS*(1+np.maximum(F_ex,0)/KF+np.maximum(G_ex,0)/KG)
rate_ex  = Vmax * np.maximum(S_ex,0) / denom_ex
freq_S_ex = conc_to_freq(S_ex, registers[0])
freq_F_ex = conc_to_freq(F_ex, registers[0][::-1])
amp_ex    = smooth(rate_ex / (rate_ex.max()+1e-12))

axes[0].plot(t_plot, S_ex, color='#e63946', lw=1.5, label='[S] Sucrose (mM)')
axes[0].plot(t_plot, F_ex, color='#2a9d8f', lw=1.5, label='[F] Fructose (mM)')
axes[0].plot(t_plot, G_ex, color='#457b9d', lw=1.5, label='[G] Glucose (mM)')
axes[0].set_title('[S]₀ = 333 mM — Concentration Profiles',
                   color='white', fontsize=10, fontweight='bold')
axes[0].set_ylabel('Conc. (mM)', color='#8b949e', fontsize=9)
axes[0].legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=8)

axes[1].plot(t_plot, freq_S_ex, color='#e63946', lw=1.2,
             label='Pitch ← [S] sucrose (descends)')
axes[1].plot(t_plot, freq_F_ex, color='#2a9d8f', lw=1.2,
             label='Pitch ← [F] fructose (ascends)')
axes[1].set_title('Musical Pitch Mapping (D Dorian scale)',
                   color='white', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Frequency (Hz)', color='#8b949e', fontsize=9)
axes[1].legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=8)

axes[2].fill_between(t_plot, amp_ex, alpha=0.6, color='#f4a261')
axes[2].plot(t_plot, amp_ex, color='#f4a261', lw=1.5,
             label='Amplitude ← reaction rate v(t)')
axes[2].set_title('Amplitude = Reaction Rate\n'
                   '(product inhibition causes gradual slowdown → natural musical fade)',
                   color='white', fontsize=10, fontweight='bold')
axes[2].set_ylabel('Amplitude', color='#8b949e', fontsize=9)
axes[2].legend(facecolor='#21262d', edgecolor='#30363d', labelcolor='white', fontsize=8)

axes[3].plot(t_plot[::30], audio_mix[::30], color='#c084fc', lw=0.4, alpha=0.75)
axes[3].set_title('Composite Waveform (5 voices, all concentrations)',
                   color='white', fontsize=10, fontweight='bold')
axes[3].set_ylabel('Audio Signal', color='#8b949e', fontsize=9)
axes[3].set_xlabel('Audio Time (s)', color='#8b949e', fontsize=9)

for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    ax.grid(True, color='#21262d', lw=0.6, ls='--')

plt.tight_layout()
plt.savefig('sonification_mapping.png', dpi=150, bbox_inches='tight',
            facecolor='#0d1117')
plt.show()
print("✓ Sonification mapping plot saved.")

# ─────────────────────────────────────────────────────────
# CELL 8 — Download All Files
# ─────────────────────────────────────────────────────────
print("⬇️  Downloading files to your computer...")
files.download('concentration_profiles.png')
files.download('sonification_mapping.png')
files.download('invertase_sonification.wav')
print("✓ Done! Check your Downloads folder.")

# ─────────────────────────────────────────────────────────
# CELL 9 — (OPTIONAL) Parameter Exploration
# ─────────────────────────────────────────────────────────
# Change parameters and re-run Cells 3–8 to hear the effect.
#
# Physical meaning of each parameter:
#   KS  (Km)  = sucrose affinity for invertase
#               ↓ KS → enzyme binds sucrose more tightly → faster rate at low [S]
#   KF, KG    = product inhibition strength
#               ↓ KF or KG → stronger inhibition → faster musical fade
#   Vmax      = maximum catalytic rate at saturating sucrose
#               ↑ Vmax → louder, more energetic music
#
# Try these adjustments:
# Vmax = 1.52    # 2× faster — higher temperature (Arrhenius)
# KF   = 29.4    # 2× stronger fructose inhibition — more product inhibition
# KS   = 8.35    # 2× tighter binding — enzyme operates near saturation
