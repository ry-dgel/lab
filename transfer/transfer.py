import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import spinmob as sp

################
# Lock-In Data #
################

# Figures out the delimiter used in a csv style file.
def get_delim(file):

    with open(file, "r") as f:
        # Reads file line by line, so that only one line needs to be read
        for _, line in enumerate(f):
            # ignore comments if present
            if '#' not in line:
                for delim in [",", ";", ":", "\t"]:
                    if delim in line:
                        return delim

    print("No delimiters found")
    return None

# Read data from csv file.
def read_csv(file):
    delim = get_delim(file)
    return np.genfromtxt(file, names=True, delimiter=delim)

# Get data from csv file exported from lock in.
def unpack(filename, fields = []):
    chunks = {}
    with open(filename) as f:
        # Skip header line
        next(f)
        for line in f:
            # Each line has form:
            # chunk;timestamp;size;fieldname;data0;data1;...;dataN
            delim = get_delim(filename)
            entries = line.split(delim)

            chunk = entries[0]
            # If this is a new chunk, add to chunks dictionary.
            if chunk not in chunks.keys():
                chunks[chunk] = {}
            # Use chunk dictionary for data storage
            # This separates the runs
            dic = chunks[chunk]

            fieldname = entries[3]
            data = np.array([float(x) for x in entries[4:]])

            # Add named dataset to dictionary for each desired fieldname
            # If no fieldnames specified in fields, just return all.
            if fieldname in fields or len(fields) == 0:
                if fieldname not in dic.keys():
                    dic[fieldname] = data
                else:
                    dic[fieldname] = np.concatenate((dic[fieldname], data))

    data_chunks = list(chunks.values())
    return data_chunks

############
# Plotting #
############
def plot_trans(trans, lines=True, norm=False, unwrap=False):
    amp = trans['r']
    if norm:
        amp /= max(amp)

    phase = trans['phase']
    if unwrap:
        phase = np.unwrap(phase)

    freq = trans['frequency']

    fig, axes = plt.subplots(2, 1, sharex = True, squeeze = True)
    plot_amp(axes[0], amp, freq)
    plot_phase(axes[1], phase, freq)

    if lines:
        axes[0].plot(freq, np.ones_like(freq), linestyle='--', zorder=0)
        # Find Crossing Point
        cross = np.argmax(amp <= 1.0)
        # Plot 1 one pole slope crossing at cross point
        axes[0].plot(freq, freq[cross]/freq, linestyle='--', zorder=0)
        axes[0].plot(freq, freq[cross]**2/(freq**2), linestyle='--',zorder=0)

    axes[0].set_ylim([min(amp)-np.power(10,np.round(np.log(min(amp)))),
                      max(amp)+np.power(10,np.floor(np.log(max(amp))-1))])

    return fig

def plot_amp(ax, amp, freq):
    ax.plot(freq, amp)

    # Set log axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_ylabel("Amplitude")
    ax.set_xlabel("Frequency (Hz)")

def plot_phase(ax, phase, freq):
    ax.plot(freq, phase/(2 * np.pi) * 360)

    ax.set_xscale('log')
    ax.set_ylabel("Phase")
    ax.set_yticks([-180,-90,0,90,180])
    ax.set_xlabel("Frequency (Hz)")

# Instances of plt.close() are so that jupyter doesn't plot every single step.
def plot_funcs(funcs, freq, labels=[]):
    total, axes = plt.subplots(2,1,sharex=True,squeeze=True)
    for index, func in enumerate(funcs):
        if isinstance(func, AnCompFun):
            fig = func.plot(freq)
        else:
            fig = func.plot()
        amp_data = fig.axes[0].lines[0].get_data()
        phase_data = fig.axes[1].lines[0].get_data()
        plot_amp(axes[0],amp_data[1],amp_data[0])
        plot_phase(axes[1], phase_data[1] * (2 * np.pi) / 360, phase_data[0])
        plt.close(fig)
    if index:
        axes[0].legend(labels)
        plt.close(total)
    return total

############
# Analysis #
############
class Compfun:
    def __init__(self, comp, freq):
        if not len(comp) == len(freq):
            raise ValueError("Length Mismatch")

        self.c = comp
        self.f = freq

    @classmethod
    def from_polar(Cls, polar):
        return Cls(polar['r'] * np.exp(1j * polar['phase']),
                       polar['frequency'])

    def __eq__(self, other):
        return (np.array_equal(self.c, other.c)
                and np.array_equal(self.f, other.f))

    def __add__(self,other):
        if isinstance(other, Compfun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mismatch between two functions")
            return Compfun(self.c + other.c, self.f)
        else:
            return Compfun(self.c + other, self.f)

    def __neg__(self):
        return Compfun(-self.c, self.f)

    def __sub__(self,other):
        if isinstance(other, Compfun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mismatch between two functions")
            return Compfun(self.c - other.c, self.f)
        else:
            return Compfun(self.c - other, self.f)

    def __mul__(self, other):
        if isinstance(other, AnCompFun):
            return Compfun(self.c * other.func(self.f), self.f)

        elif isinstance(other, Compfun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mismatch between two functions")
            return Compfun(self.c * other.c, self.f)

    def __truediv__(self,other):

        if isinstance(other, AnCompFun):
            return Compfun(self.c / other.func(self.f), self.f)
        elif isinstance(other, Compfun):
            if not np.array_equal(self.f, other.f):
                raise ValueError("Frequency mistmatch between two functions")
            return Compfun(self.c / other.c, self.f)

    def polar(self):
        return {'r': np.abs(self.c),
                'phase': np.angle(self.c),
                'frequency': self.f}

    def plot(self, **kwargs):
        return plot_trans(self.polar(), **kwargs)

# Combines the dataset of two CompFun objects.
# Averages data points of identical frequency, concats other points.
def merge(cm1, cm2):
    c = cm1.c
    f = cm1.f
    for i, freq in cm2.f:
        if freq in f:
            c[i] = np.average([cm1.c[i], cm2.c[i]])
        else:
            f.append[freq]
            c.append[cm2.c[i]]
    combined = zip(c,f)
    combined.sort(key = lambda pair: pair[1])
    c,f = map(list, zip(*combined))
    return Compfun(c, f)

def load(filename):
    return [Compfun.from_polar(chunk) for chunk in
            unpack(filename, fields=['r','phase','frequency'])]

##############################
# Analytic Complex Functions #
##############################
class AnCompFun:
    def __init__(self, function):
        self.func = function

    def __add__(self, other):
        if isinstance(other, AnCompFun):
            return AnCompFun(lambda f: self.func(f) + other.func(f))
        elif isinstance(other, Compfun):
            return Compfun(self.func(other.f) + other.c, other.f)

    def __mul__(self, other):

        if isinstance(other, Compfun):
            return Compfun(self.func(other.f) * other.c, other.f)

        elif isinstance(other, AnCompFun):
            return AnCompFun(lambda f: (self.func(f) * other.func(f)))

    def __truediv__(self, other):
        if isinstance(other, Compfun):
            return Compfun(self.func(other.f) / other.c, other.f)

        elif isinstance(other, AnCompFun):
            return AnCompFun(lambda f: (self.func(f) / other.func(f)))

    def apply(self, freq):
        return Compfun(self.func(freq), freq)

    def plot(self,freq, **kwargs):
        return self.apply(freq).plot(**kwargs)

def hp(cutoff):
    wc = 2 * np.pi * cutoff
    return AnCompFun(lambda f : 2 * np.pi * f/wc / (2 * np.pi * f/wc - 1j))

def lp(cutoff):
    wc = 2 * np.pi * cutoff
    return AnCompFun(lambda f: -1j / ((2 * np.pi * f/wc) - 1j))

def pi(corner, gain):
    I0 = corner * gain / 2 * np.pi
    return AnCompFun(lambda f: I0/(1j * 2 * np.pi * f) + gain)

def ho(res, damp):
    wres = 2 * np.pi * res
    wdamp = 2 * np.pi * damp
    return AnCompFun(lambda f: 1 / (1 + 1j * 2 * np.pi * f * wdamp / wres**2 - ((2 * np.pi * f)/wres)**2))

def lag(f,a):
    w = 2 * np.pi * f
    return AnCompFun(lambda f: a*(1+1j * 2 * np.pi * f / w)/(1 +  a * 1j * 2 * np.pi * f / w))

def lead(f,a):
    w = 2 * np.pi * f
    return AnCompFun(lambda f: (1 + a*1j * 2 * np.pi * f / w)/(1 + 1j * 2 * np.pi * f / w))

def amp(a):
    return AnCompFun(lambda f: np.ones(np.size(f)) * a)

def delay(delta_t):
    return AnCompFun(lambda f: np.exp(-1j * 2 * np.pi * f * delta_t))

############
# Analysis #
############
# TODO: MAKE THESE WORK
def gain_margin(function):
    if isinstance(function, AnCompFun):
        z = opt.root_scalar(lambda f: np.angle(function.func(f)) + np.pi, bracket=[0,1E15], x0=1000.0)
        return np.abs(function.f(z))
    elif isinstance(function, Compfun):
        z = function.f[np.argmin(np.abs(np.angle(function.c) - np.pi))]
        return np.abs(function.c[z])
    raise ValueError("Wrong Type")

def phase_margin(function):
    if isinstance(function, AnCompFun):
        z = opt.root_scalar(lambda f: np.abs(function.func(f)) - 1, bracket=[0.,1E15], x0=1000.0)
        return np.angle(function.f(z))
    elif isinstance(function, Compfun):
        z = function.f[np.argmin(np.abs(np.abs(function.c) - 1))]
        return np.angle(function.c[z])
    raise ValueError("Wrong Type")
