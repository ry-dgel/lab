import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import scipy.constants as const
import scipy.special as special

def r_from_finesse(F):
    return np.sqrt(-(const.pi/F - 1))

##############
# Parameters #
##############
# Laser
wave_len   = 652E-9             # Central wavelength
freq       = const.c/wave_len   # Derived frequency and angular frequency
omega      = 2 * const.pi * freq

# Cavity
length     = 41 * wave_len / 2
F          = 30000
r          = r_from_finesse(F)
line_width = wave_len / (2 * F)
freq_width = const.c / (2 * F * length)

# Modulation
mod_depth   = 1.08
mod_freq   = 120E6
mod_omega  = 2 * const.pi * mod_freq

# Fano
# Experimentally found parameters for producing a fano lineshape resonance.
eta = 0.61 + 0.14j
eps = 0.69
alf = 8.0E-6
#############
# Functions #
#############
# Calculated carrier power at given mod_depth
def pc(mod_depth):
    return special.j0(mod_depth)**2

#Calculated sideband power at given mod_depth
def ps(mod_depth):
    return special.j1(mod_depth)**2

# Reflection Coefficient
def R(omega, r, length):
    phi = omega * 2 * length / const.c
    e   = np.exp(1j*phi)
    return r * (e - 1) / (1 - (r**2) * e)

#Transmission Coefficient
def T(omega, r, length):
    refl = np.abs(R(omega, r, length))**2
    return (1 - refl ** 2)/(1 + refl ** 2)

# Could combine the following two into complex function...
# Sine part of reflected power
def error(omega, mod_omega, r, length, mod_depth, phase = 0):
    Pc = pc(mod_depth)
    Ps = ps(mod_depth)
    return 2 * np.sqrt(Pc * Ps) * np.imag((R(omega, r, length) *
                    np.conj(R(omega + mod_omega, r, length)) -
                    np.conj(R(omega, r, length)) *
                    R(omega - mod_omega, r, length)) * np.exp(1j*phase))

# Cosine part of reflected power
def wrong_error(omega, mod_omega, r, length, mod_depth):
    Pc = pc(mod_depth)
    Ps = ps(mod_depth)
    return 2 * np.sqrt(Pc * Ps) * np.real(R(omega, r, length) *
                   np.conj(R(omega + mod_omega, r, length)) -
                   np.conj(R(omega, r, length)) *
                   R(omega - mod_omega, r, length))

# Transmitted power, sine part
def trans_power(omega, mod_omega, r, length, mod_depth):
    return np.sqrt(T(omega, r, length) * pc(mod_depth) +
                   T(omega + mod_omega, r, length) * ps(mod_depth) +
                   T(omega - mod_omega, r, length) * ps(mod_depth))

########
# Fano #
########
def R_fano(omega, r, length, eta, epsilon, alpha):
    phi = omega * 2 * length / const.c
    e = np.exp(2*alpha + 1j * phi)
    return r * ((r**2 - 1) * epsilon**2 + (e - r**2) * eta) / (r**2 - e)

# Sine part of reflected power with fano
def error_fano(omega, mod_omega, r, length, eta, epsilon, alpha, phase):
    return -np.imag((R_fano(omega, r, length, eta, epsilon, alpha)
                     *np.conj(R_fano(omega + mod_omega, r, length, eta, epsilon, alpha))
                     -np.conj(R_fano(omega, r, length, eta, epsilon, alpha))
                     *R_fano(omega - mod_omega, r, length, eta, epsilon, alpha))
                     *np.exp(1j*phase))

# Choice of error function to plot
err_func = error
# Generate figure
fig, axes = plt.subplots(1, 2, dpi=72)
plt.subplots_adjust(left=0.06, bottom=0.25)

# Length of cavity to plot
lengths = np.linspace(length - 0.08E-9, length + 0.08E-9, 1000)

# Error function over lengths
err = err_func(omega, mod_omega, r, lengths, mod_depth, 0)

# Transmitted signal over lengths
signal = trans_power(omega, mod_omega, r, lengths, mod_depth)

# Plot and get line object
# Error
l, = axes[0].plot((lengths-length)*1E12, err, lw=2, color='red')
# Derivative of error
g, = axes[0].plot((lengths-length)*1E12, 10*np.gradient(err), lw=2, color='grey', linestyle='dashed')
# Transmitted, on second plot
t, = axes[1].plot((lengths - length)*1E12, signal, lw=2, color='blue')
# Set axes, limits, and labels
plt.axis([-80, 80, -1.05, 1.05])
axes[1].set_ylim([0,1.2])
axes[1].set_xlim([-80,80])
axes[0].set_ylim([-1,1])
axes[0].set_xlim([-80,80])
axes[0].set_xlabel("Cavity Length (pm)")
axes[0].set_ylabel("Error Signal (A.U.)")
axes[1].set_ylabel("Transmited Power (A.U.)")

# Define sliders for interactivity
axcolor = '#ebdbb2'
axp  = plt.axes([0.2, 0.15, 0.7, 0.03], facecolor=axcolor)
axmo = plt.axes([0.2, 0.10, 0.7, 0.03], facecolor=axcolor)
axF  = plt.axes([0.2, 0.05, 0.7, 0.03], facecolor=axcolor)
axB  = plt.axes([0.2, 0.00, 0.7, 0.03], facecolor=axcolor)

# Set limits and default values of sliders
smo = Slider(axmo, 'Mod Freq', 50, 2000, valinit = 120.0,
             valstep =  10.0,  valfmt="%1.0f MHz")
sp  = Slider(axp,  'Phase',     0,    2, valinit =   0.0,
             valstep =   0.05, valfmt="%1.2f pi")
sF = Slider(axF,  'Finesse',  15,   60, valinit =  30.0,
             valstep =   0.1,  valfmt="%1.1f 10^3")
sB = Slider(axB,  'Mod Depth', 1,    2, valinit =   1.08,
             valstep =   0.01)

# Define function to update plot when values change
def update(val):
    mod_om = smo.val * 1E6 * 2 * const.pi
    r = r_from_finesse(sF.val * 1000)
    err = err_func(omega, mod_om, r, lengths, sB.val, sp.val * const.pi)
    signal = trans_power(omega, mod_om, r, lengths, sB.val)
    l.set_ydata(err)
    g.set_ydata(10*np.gradient(err))
    t.set_ydata(signal)
    fig.canvas.draw_idle()

# Tell sliders to update plot when they change
smo.on_changed(update)
sp.on_changed(update)
sF.on_changed(update)
sB.on_changed(update)

# Plot!
plt.show()
