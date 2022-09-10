"""Contains additional functions used by motor_sim.py
Reference 1 - "Engineering Model to Calculate Mass Flow Rate of a Two-Phase
Saturated Fluid Through An Injector Orifice", Brian J Solomon
Zimmermann thesis (add link later)
"""

########################################
# Joe Hunt updated 20/06/19            #
# Kacper Kowalski updated 10/09/22     #
# All units SI unless otherwise stated #
########################################

__copyright__ = """

    Copyright 2019 Joe Hunt
    
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

"""

import numpy as np
import os
import scipy.optimize
from scipy import optimize
from dataclasses import dataclass

from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
RP.SETPATHdll(os.environ['RPPREFIX'])
MASS_BASE_SI = RP.GETENUMdll(0,"MASS BASE SI").iEnum


def area(d):
    return 0.25 * np.pi * d * d


def dyer_injector(cpres, inj_dia, lden, inj_pdrop, hl, manifold_P, vpres, numinj):
    """Models the mass flow rate of (initially liquid) n2o through a single
    injector orifice using the 2-phase model proposed by Dyer et al"""
    inj_pdrop_og = 0

    A=area(inj_dia)*numinj

    if inj_pdrop < 3e5:
        #print('accuracy warning: injector pdrop so low that'
        #      '2-phase Dyer model no longer applies. '
        #      'approximating with linear pdrop/mdot characteristic')
        inj_pdrop_og = inj_pdrop
        inj_pdrop = 3e5

    # get downstream spec. enthalpy and density
    h2, rho2 = chamber_vap(vpres, cpres)

    # single-phase incompressible mass flow rate:
    # See Ref 1, page 8, Eqn 2.13
    Cd = 0.6  # Waxman et al, adapted for square edged orifices

    mdot_spi = Cd * A * np.sqrt(2 * lden * inj_pdrop)

    # mass flow rate by homogenous equilibrium model:
    # See Ref 1, page 9, Eqn 2.14
    mdot_hem = Cd * A * rho2 * np.sqrt(2 * (hl - h2))

    if vpres < cpres:
        raise RuntimeError("injector pdrop lower than vapour pressure",
                           "2-phase Dyer model no longer applies!")
    # non-equilibrium parameter k (∝ ratio of bubble growth
    #                              time and liquid residence time)
    # See Ref 1, page 9, Eqn 2.17
    k = 1# np.sqrt((manifold_P - cpres) / (vpres - cpres))

    # mass flow rate by Dyer model
    # See Ref 1, page 11, Eqn 2.21
    mdot_ox = ((k * mdot_spi) + mdot_hem) / (1 + k)

    if 0 < inj_pdrop_og < 3e5:
        mdot_ox *= inj_pdrop_og / (3e5)

    return mdot_ox, mdot_spi, mdot_hem, h2

def dyer_injector_vapour(vden):
    
    mdot_ox=0
    return mdot_ox


def _lookup_index(cpres, OF):
    if not 0 <= cpres <= 90e5:
        raise RuntimeError('chamber pressure out of propep data range!')
    if not 1/39 <= OF <= 39:
        raise RuntimeError('OF out of propep data range!')

    # these are ugle but just leave them as is
    # Henry Franks tried to change these and broke everything, 24/3/21
    rounded_cpres = int(5*round((cpres/(10**5))/5))                             
    cpres_line = int(765*(((100-rounded_cpres)/5)-1)+9)                         
    rounded_oxpct = (round(2*((OF/(1+OF))*100)))/2                              
    oxpct_line = int((4*((97.5-rounded_oxpct)/0.5))+3) 

    return cpres_line + oxpct_line - 1


def c_star_lookup(cpres, OF, propep_data):
    """Looks up ratio of characteristic velocity from chamber pressure and OF
    ratio using propep data
    """
    # note that propep data is in feet/s
    # we multiply by 0.3048 to convert from fps to m/s
    return float(propep_data[_lookup_index(cpres, OF)].split()[4]) * 0.3048


def gamma_lookup(cpres, OF, propep_data):
    """Looks up ratio of specific heats from chamber pressure and OF ratio
    using propep data"""
    return float(propep_data[_lookup_index(cpres, OF)].split()[1])


def vapour_injector(inj_dia, vden, inj_pdrop):
    """Models the mass flow rate of single phase vapour-only through a single
    injector orifice"""
    cd = 0.65  # somewhat a guess
    return cd * area(inj_dia) * np.sqrt(2 * vden * inj_pdrop)


def Z2_solve(temp, P):
    """changed completly"""
    Z=RP.REFPROPdll("NITROUS OXIDE","TP","Zvap",MASS_BASE_SI, 0,0,temp, P, [1.0]).Output[0]
    return Z


def ball_valve_K(Re, d1, d2, L):
    """Returns full-bore ball valve flow coefficient as a thick orifice"""
    rd_2 = d2 * d2 / d1 / d1  # square of the diameter ratio

    if Re < 2500:
        K = (2.72 + rd_2 * (120/Re - 1)) * (1 - rd_2) * (rd_2*rd_2 - 1)
    else:
        K = (2.72 + rd_2 * 4000/Re) * (1 - rd_2) * ((1 / rd_2 / rd_2) - 1)

    K *= 0.584 + 0.0936 / (pow(L / d2, 1.5) + 0.225)

    return K


def Nikuradse(Re):
    """Returns the friction factor, f, for a given Reynolds number, fitting the
    Nikuradse model"""
    return 0.0076 * pow(3170/Re, 0.165) / (1 + pow(3170/Re, 7)) + 16/Re


def thermophys(temp):
    """Get N2O data at a given temperature.
    Uses polynomials from ESDU sheet 91022. All units SI.

    Returns:
        - N2O liquid density
        - vapour density
        - latent heat of vaporization
        - dynamic viscosity
        - vapour pressure for input temperature.


        Changed to REFPROP to get consistent data with density end enthalpy of downstream conditions
        Saturation properties are considered
    """
    properties=RP.REFPROPdll("NITROUS OXIDE","TSAT","Dliq;Dvap;Hliq;Hvap;CV;P;VIS;Zvap;Eliq;Evap",MASS_BASE_SI, 0,0,temp, 0, [1.0])

    lden = properties.Output[0]

    vden = properties.Output[1]

    hl = properties.Output[2]

    hg = properties.Output[3]

    c = properties.Output[4]

    vpres = properties.Output[5]

    ldynvis = properties.Output[6]

    Z = properties.Output[7]

    lu = properties.Output[8]

    vu = properties.Output[9]

    return (lden, vden, hl, hg, c, vpres, ldynvis, Z, lu, vu)

def thermophys_vapour(vden):
    """Returns N2O vapour properties based on assumption that vapour leftover in tank
    stays saturated, otherwise during decompression it would condense"""
    properties=RP.REFPROPdll("NITROUS OXIDE","DQ","PT",MASS_BASE_SI, 0,0,vden, 1, [1.0])
    vap_pres=properties.Output[0]
    temp=properties.Output[1]
    return vap_pres, temp

def chamber_vap(p1, p2):
    """Returns N2O specific enthalpy of vapour and vapour density at chambe
    pressure, again, 2nd argument is guess of vapour temperature

    Changed to refprop, now calculates properties of mixtrure when given isentropic flow to the chamber
    In future can be integrated into thermophys"""
    s=RP.REFPROPdll("NITROUS OXIDE","PSAT","Sliq",MASS_BASE_SI, 0,0,p1, 0, [1.0]).Output[0]
    properties=RP.REFPROPdll("NITROUS OXIDE","PS","H;D",MASS_BASE_SI, 0,0,p2, s, [1.0])
    hg = properties.Output[0]
    vden = properties.Output[1]
    return (hg, vden)

def mach_exit(gamma, NOZZLE_AREA_RATIO):
    """Returns the exit mach number by numerical solution"""

    def mach_error(m):
        """Finds the discrepancy between the non-dimensional mass flow
        rate (stagnation) found using Mach number relations and that
        calculated using current guess of exit conditions"""
        m_new = np.power(
            2 / (gamma + 1) * (1 + (gamma - 1) * m * m / 2),
            (gamma + 1) / (gamma - 1) / 2
        ) / NOZZLE_AREA_RATIO
        return abs(m - m_new)

    return float(
            scipy.optimize.minimize(mach_error, 2.5, tol=1e-9, method='Powell').x)

def EQTemp (Utot, mtot, Vdesired):
    def EQVolumeDifference (Tinit, Utot, mtot, Vdesired):
        properties=RP.REFPROPdll("NITROUS OXIDE","TSAT","Evap;Eliq;Dvap;Dliq",MASS_BASE_SI, 0,0,Tinit, 0, [1.0])
        uvap=properties.Output[0]
        uliq=properties.Output[1]
        rhovap=properties.Output[2]
        rholiq=properties.Output[3]
        x=(Utot/mtot-uliq)/(uvap-uliq)
        Vtank=mtot*((1-x)/rholiq+x/rhovap)
        return Vtank-Vdesired
    Temp=optimize.bisect(EQVolumeDifference, 240, 309.5, xtol=0.01, args=tuple([Utot, mtot, Vdesired]))
    
    return Temp