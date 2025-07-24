from dataclasses import dataclass

@dataclass
class SimParams:
    alpha_scale: float   # множитель для коэффициента затухания α
    chi_scale:   float   # множитель для магнитной восприимчивости χ (теперь скаляр)
    k_scale:  float  # множитель для K(T)
    m_scale: float  # множитель массива m
