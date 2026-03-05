import numpy as np
import logging as logg

# === OptiCommPy Imports (same style as your file) ===
from optic.models.devices import iqm, coherentReceiver, basicLaserModel
from optic.models.channels import ssfm
from optic.comm.modulation import grayMapping
from optic.comm.sources import symbolSource
from optic.dsp.core import upsample, pulseShape, pnorm, firFilter, decimate, symbolSync, phaseNoise
from optic.utils import parameters, dBm2W, ber2Qfactor
from optic.plot import pconst
from optic.dsp.equalization import edc, mimoAdaptEqualizer
from optic.dsp.carrierRecovery import cpr
from optic.comm.metrics import fastBERcalc, calcEVM

logg.basicConfig(level=logg.INFO, format='%(message)s', force=True)

# 1. OptiCommPy system parameters

SpS = 16
Rs = 32e9
Fs = Rs * SpS
M = 16
nBits = 100000
rollOff = 0.01
nFilterTaps = 1024
mzmScale = 0.5
Vpi = 2
P_launch_dBm = 0
laserLinewidth = 100e3

paramSymb = parameters()
paramSymb.nSymbols = int(nBits // np.log2(M))
paramSymb.M = M
paramSymb.constType = "qam"
paramSymb.dist = "uniform"
paramSymb.seed = 123
paramSymb.shapingFactor = 0
paramSymb.px = np.ones(M) / M

#Tx pulse shaping
paramPulse = parameters()
paramPulse.pulseType = "rrc"
paramPulse.nFilterTaps = nFilterTaps
paramPulse.rollOff = rollOff
paramPulse.SpS = SpS

#IQM
paramIQM = parameters()
paramIQM.Vpi = Vpi
paramIQM.VbI = -Vpi
paramIQM.VbQ = -Vpi
paramIQM.Vphi = Vpi / 2

#Optical channel (SSFM)
paramCh = parameters()
paramCh.Ltotal = 80
paramCh.Lspan = 80
paramCh.alpha = 0.2
paramCh.D = 16
paramCh.gamma = 1.3
paramCh.Fc = 193.1e12
paramCh.hz = 0.2
paramCh.prgsBar = False
paramCh.Fs = Fs
paramCh.amp = 'edfa'
paramCh.NF = 10

#Receiver front-end
paramLO = parameters()
paramLO.P = 2
paramLO.lw = 100e3
paramLO.RIN_var = 0
paramLO.Fs = Fs
paramLO.seed = 789
paramLO.freqShift = -128e6  # FO

paramFE = parameters()
paramFE.Fs = Fs

paramPD = parameters()
paramPD.B = Rs
paramPD.Fs = Fs
paramPD.ideal = False
paramPD.seed = 1011

#Rx pulse shaping + decimation
paramRxPulse = parameters()
paramRxPulse.SpS = SpS
paramRxPulse.nFilterTaps = nFilterTaps
paramRxPulse.rollOff = rollOff
paramRxPulse.pulseType = "rrc"

paramDec = parameters()
paramDec.SpSin = SpS
paramDec.SpSout = 2

#EDC
paramEDC = parameters()
paramEDC.L = paramCh.Ltotal
paramEDC.D = paramCh.D
paramEDC.Fc = paramCh.Fc
paramEDC.Rs = Rs
paramEDC.Fs = 2 * Rs

#Equalizer
paramEq = parameters()
paramEq.nTaps = 35
paramEq.SpS = paramDec.SpSout
paramEq.numIter = 2
paramEq.storeCoeff = False
paramEq.M = M
paramEq.shapingFactor = 0
paramEq.constType = "qam"
paramEq.prgsBar = False
paramEq.alg = ['da-rde', 'rde']
paramEq.mu = [5e-3, 5e-4]

#CPR
paramCPR = parameters()
paramCPR.alg = 'bps'
paramCPR.M = M
paramCPR.constType = "qam"
paramCPR.shapingFactor = 0
paramCPR.N = 25
paramCPR.B = 64
paramCPR.returnPhases = True
paramCPR.Ts = 1 / Rs

#Optical carrier / LO field
sigTx_length = paramSymb.nSymbols * SpS
if laserLinewidth and laserLinewidth > 0:
    phi_pn = phaseNoise(laserLinewidth, sigTx_length, 1 / Fs, seed=123)
    sigLO = np.exp(1j * phi_pn)
else:
    sigLO = np.ones(sigTx_length, dtype=complex)

# 2. Helper metrics and safe alignment

def nmse_db(x_hat, x, eps=1e-12):
    err = np.mean(np.abs(x_hat - x) ** 2)
    pwr = np.mean(np.abs(x) ** 2) + eps
    return 10 * np.log10((err + eps) / pwr)

def align_same_length(a, b):
    n = min(len(a), len(b))
    return a[:n], b[:n]

# 3. OptiCommPy system forward

def simulate_optical_system(symbTx_in):
    #Tx
    pulse = pulseShape(paramPulse)
    symbolsUp = upsample(symbTx_in, SpS)
    sigTx = firFilter(pulse, symbolsUp)
    sigTx = sigTx / (np.max(np.abs(sigTx)) + 1e-12)

    u_drive = mzmScale * sigTx
    sigTxo = iqm(sigLO, u_drive, paramIQM)

    P_launch_W = dBm2W(P_launch_dBm)
    sigTxo = np.sqrt(P_launch_W) * pnorm(sigTxo)

    #Channel
    sigCh = ssfm(sigTxo, paramCh)

    #Rx front-end
    paramLO.Ns = len(sigCh)
    sigLO_Rx = basicLaserModel(paramLO)
    sigRxFE = coherentReceiver(sigCh, sigLO_Rx, paramFE, paramPD)

    #Rx DSP: matched filter -> decimate -> EDC -> symbolSync -> EQ -> CPR
    rx_pulse = pulseShape(paramRxPulse)
    sigRxMF = firFilter(rx_pulse, sigRxFE)
    sigRxDec = decimate(sigRxMF, paramDec)
    sigRxCD = edc(sigRxDec, paramEDC)

    # symbolSync returns aligned symbols at 2 SpS (because paramDec.SpSout=2)
    symbRxCD = symbolSync(sigRxCD, symbTx_in, 2)

    # IMPORTANT: output "d" (aligned symbols) in same domain as symbTx
    d = pnorm(symbRxCD)

    # Equalizer + CPR (optional, but you used it)
    x = pnorm(sigRxCD)
    paramEq.L = [int(0.2 * d.shape[0]), int(0.8 * d.shape[0])]
    y_EQ = mimoAdaptEqualizer(x, paramEq, d)
    y_CPR, _ = cpr(y_EQ, paramCPR)

    # y_CPR is also symbol-domain-ish, but length/domain can drift.
    # To be safe and consistent with training target, RETURN d.
    return d, y_CPR, sigRxCD, y_EQ

def simulate_optical_system(symbTx_in):
    #Tx
    pulse = pulseShape(paramPulse)
    symbolsUp = upsample(symbTx_in, SpS)
    sigTx = firFilter(pulse, symbolsUp)
    sigTx = sigTx / (np.max(np.abs(sigTx)) + 1e-12)

    u_drive = mzmScale * sigTx
    sigTxo = iqm(sigLO, u_drive, paramIQM)

    P_launch_W = dBm2W(P_launch_dBm)
    sigTxo = np.sqrt(P_launch_W) * pnorm(sigTxo)

    #Channel
    sigCh = ssfm(sigTxo, paramCh)

    #Rx front-end
    paramLO.Ns = len(sigCh)
    sigLO_Rx = basicLaserModel(paramLO)
    sigRxFE = coherentReceiver(sigCh, sigLO_Rx, paramFE, paramPD)

    #Rx DSP: matched filter -> decimate -> EDC -> symbolSync -> EQ -> CPR
    rx_pulse = pulseShape(paramRxPulse)
    sigRxMF = firFilter(rx_pulse, sigRxFE)
    sigRxDec = decimate(sigRxMF, paramDec)
    sigRxCD = edc(sigRxDec, paramEDC)

    # symbolSync returns aligned symbols at 2 SpS (because paramDec.SpSout=2)
    symbRxCD = symbolSync(sigRxCD, symbTx_in, 2)

    # IMPORTANT: output "d" (aligned symbols) in same domain as symbTx
    d = pnorm(symbRxCD)

    # Equalizer + CPR (optional, but you used it)
    x = pnorm(sigRxCD)
    paramEq.L = [int(0.2 * d.shape[0]), int(0.8 * d.shape[0])]
    y_EQ = mimoAdaptEqualizer(x, paramEq, d)
    y_CPR, _ = cpr(y_EQ, paramCPR)

    # y_CPR is also symbol-domain-ish, but length/domain can drift.
    # To be safe and consistent with training target, RETURN d.
    return sigRxCD, d, y_EQ, y_CPR


def opticommpy_blackbox_symbol(symb_in):
    sigRxCD, d, y_EQ, y_CPR = simulate_optical_system(symb_in)
    return d

# 4. Wiener–Hammerstein inverse DPD model (complex L–N–L)

class WHInverseDPD:

    def __init__(self, L=7, ridge=1e-3):
        self.L = L
        self.ridge = ridge
        self.g1 = np.zeros(L, dtype=np.complex128); self.g1[0] = 1+0j
        self.g2 = np.zeros(L, dtype=np.complex128); self.g2[0] = 1+0j
        self.a = np.array([1+0j, 0+0j, 0+0j], dtype=np.complex128)

    def _fir(self, x, h):
        return np.convolve(x, h, mode="full")[:len(x)]

    def apply(self, y):
        u = self._fir(y, self.g1)
        w = self.a[0]*u + self.a[1]*u*(np.abs(u)**2) + self.a[2]*u*(np.abs(u)**4)
        return self._fir(w, self.g2)

    def _solve_ridge(self, Phi, d):
        A = Phi.conj().T @ Phi + self.ridge * np.eye(Phi.shape[1])
        b = Phi.conj().T @ d
        return np.linalg.solve(A, b)

    def fit_post_inverse_als(self, y, u_target, iters=3):
        """
        Train G so that G(y) ≈ u_target (post-inverse).
        """
        y, u_target = align_same_length(y, u_target)

        for _ in range(iters):
            #update g2
            u = self._fir(y, self.g1)
            w = self.a[0]*u + self.a[1]*u*(np.abs(u)**2) + self.a[2]*u*(np.abs(u)**4)

            W = np.zeros((len(w), self.L), dtype=np.complex128)
            for i in range(self.L):
                W[i:, i] = w[:len(w)-i]
            self.g2 = self._solve_ridge(W, u_target)

            #update a
            u = self._fir(y, self.g1)
            Phi = np.column_stack([u, u*(np.abs(u)**2), u*(np.abs(u)**4)])

            # filter each column by g2 so "a" sees the correct cascade
            Phi_f = np.zeros_like(Phi, dtype=np.complex128)
            for k in range(Phi.shape[1]):
                Phi_f[:, k] = self._fir(Phi[:, k], self.g2)

            self.a = self._solve_ridge(Phi_f, u_target)

            #update g1 (stable proxy)
            # strict update is nonlinear; this keeps it stable
            Y = np.zeros((len(y), self.L), dtype=np.complex128)
            for i in range(self.L):
                Y[i:, i] = y[:len(y)-i]
            self.g1 = self._solve_ridge(Y, u_target)

        return self

# 5) ILA iterative loop

def ila_loop_wh(
    tx_target,
    blackbox,
    iters_outer=5,
    iters_inner=3,
    dpd_L=7,
    ridge=1e-3,
    discard=5000
):
    """
    Outer iteration:
      u_k -> y_k = P(u_k)
      Train G_k: y_k -> u_k (post-inverse)
      u_{k+1} = G_k(tx_target)

    track NMSE between y_k and tx_target (same domain, aligned symbols).
    """
    dpd = WHInverseDPD(L=dpd_L, ridge=ridge)

    u = tx_target.copy()
    history = []

    print("\nStarting WH-ILA")
    for k in range(iters_outer):
        y = blackbox(u)  # y is aligned symbols d

        tx_al, y_al = align_same_length(tx_target, y)

        # discard edges for fairness
        if 2*discard < len(tx_al):
            tx_v = tx_al[discard:-discard]
            y_v  = y_al[discard:-discard]
            u_v  = u[:len(tx_al)][discard:-discard]
        else:
            tx_v, y_v, u_v = tx_al, y_al, u[:len(tx_al)]

        e = nmse_db(y_v, tx_v)
        print(f"Iter {k+1:02d} | NMSE(y, x) = {e:.4f} dB")
        history.append(e)

        # train post-inverse on (y -> u)
        dpd.fit_post_inverse_als(y_v, u_v, iters=iters_inner)

        # update predistorted transmit sequence
        u = dpd.apply(tx_target)

    print("WH-ILA done\n")
    return dpd, history

# 6. Main

if __name__ == "__main__":
    # Desired symbols
    symbTx = symbolSource(paramSymb)

    # Train WH inverse DPD by ILA
    dpd, hist = ila_loop_wh(
        tx_target=symbTx,
        blackbox=opticommpy_blackbox_symbol,
        iters_outer=5,
        iters_inner=3,
        dpd_L=7,
        ridge=1e-3,
        discard=5000
    )

    # Final evaluation
    y_final = opticommpy_blackbox_symbol(dpd.apply(symbTx))
    symbTx_al, y_final_al = align_same_length(symbTx, y_final)

    discard = 5000
    idx = np.arange(discard, len(symbTx_al) - discard) if 2*discard < len(symbTx_al) else np.arange(len(symbTx_al))

    BER, SER, SNR = fastBERcalc(y_final_al[idx], symbTx_al[idx], M, 'qam', px=paramSymb.px)
    EVM = calcEVM(y_final_al[idx], M, 'qam', symbTx_al[idx])
    Qfactor = ber2Qfactor(BER[0])

    print(" Final Performance (after WH-ILA) ")
    print(f" Final SER: {SER[0]:.3e}")
    print(f" Final BER: {BER[0]:.3e}")
    print(f" Final SNR: {SNR[0]:.3f} dB")
    print(f" Final EVM: {EVM[0]*100:.3f} %")
    print(f" Final Qfactor: {Qfactor:.3f}")

    # Constellation
    sigRxCD, d, y_EQ, y_CPR = simulate_optical_system(symbTx_al)
    sig_sym = sigRxCD[::2]   
    pconst(sig_sym[idx])
    print("std(|y_final_al|) =", np.std(np.abs(y_final_al[idx])))
    print("unique points approx =", np.unique(np.round(y_final_al[idx], 3)).shape[0])
    pconst(y_final_al[idx])