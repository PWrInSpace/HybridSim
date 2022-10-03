"""Contains additional functions used by motor_sim.py
Reference 1 - "Engineering Model to Calculate Mass Flow Rate of a Two-Phase
Saturated Fluid Through An Injector Orifice", Brian J Solomon

Original program corrected via equations found in
Jonah E. Zimmerman "Self pressurizing tank dynamics" December 2015
https://stacks.stanford.edu/file/druid:ds271zh2987/main_adob-augmented.pdf
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


def dyer_injector(cpres, inj_dia, lden, inj_pdrop, hl, manifold_P, vpres, ls, numinj):
    """Models the mass flow rate of (initially liquid) n2o through a single
    injector orifice using the 2-phase model proposed by Dyer et al"""
    inj_pdrop_og = 0

    A=area(inj_dia)*numinj
    Cd = 0.6  # Waxman et al, adapted for square edged orifices
    # single-phase incompressible mass flow rate:
    # See Ref 1, page 8, Eqn 2.13
    mdot_spi = Cd * A * np.sqrt(2 * lden * inj_pdrop)
    if cpres>vpres: #may occur in hgly pressurized tank
        return mdot_spi
    
    if inj_pdrop < 3e5:
        #print('accuracy warning: injector pdrop so low that'
        #      '2-phase Dyer model no longer applies. '
        #      'approximating with linear pdrop/mdot characteristic')
        inj_pdrop_og = inj_pdrop
        inj_pdrop = 3e5
    
    # get downstream spec. enthalpy and density
    h2, rho2 = chamber_vap(ls, cpres)

    # mass flow rate by homogenous equilibrium model:
    # See Ref 1, page 9, Eqn 2.14
    
    mdot_hem = Cd * A * rho2 * np.sqrt(2 * (hl - h2))

    # non-equilibrium parameter k (∝ ratio of bubble growth
    #                              time and liquid residence time)
    # See Ref 1, page 9, Eqn 2.17
    k = np.sqrt((manifold_P - cpres) / (vpres - cpres))

    # mass flow rate by Dyer model
    # See Ref 1, page 11, Eqn 2.21
    mdot_ox = ((k * mdot_spi) + mdot_hem) / (1 + k)

    if 0 < inj_pdrop_og < 3e5:
        mdot_ox *= inj_pdrop_og / (3e5)
    
    return mdot_ox

def dyer_injector_vapour(vden, inj_dia, numinj, vap_pres, pres_cham, vs, vh):
    """model is up to verification"""
    A=area(inj_dia)*numinj
    Cd = 0.65  # somewhat a guess
    mdot_spi = Cd * A * np.sqrt(2 * vden * (vap_pres-pres_cham))
    """
    Currently unsure of the model correctness, so basic equation is used
    Additional research / test data required
    """
    #k=1
    #h_cham, den_cham = chamber_vap(vs, pres_cham)
    #mdot_hem = Cd * A * den_cham * np.sqrt(2 * (vh - h_cham))
    #mdot_ox = ((k * mdot_spi) + mdot_hem) / (1 + k)
    #return mdot_ox
    return mdot_spi


def _lookup_index(cpres, OF):
    if not 0 <= cpres <= 90e5:
        raise RuntimeError('Chamber pressure out of propep data range!')
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
    """Looks up ratio of characteristic velocity from chamber pressure and OF ratio using propep data"""
    # note that propep data is in feet/s
    # we multiply by 0.3048 to convert from fps to m/s
    return float(propep_data[_lookup_index(cpres, OF)].split()[4]) * 0.3048


def gamma_lookup(cpres, OF, propep_data):
    """Looks up ratio of specific heats from chamber pressure and OF ratio using propep data"""
    return float(propep_data[_lookup_index(cpres, OF)].split()[1])


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
    """Returns the friction factor, f, for a given Reynolds number, fitting the Nikuradse model"""
    return 0.0076 * pow(3170/Re, 0.165) / (1 + pow(3170/Re, 7)) + 16/Re


def thermophys(temp):
    """
        Get N2O data at a given temperature.
        Changed to REFPROP
        Saturation properties are considered
    """

    properties=RP.REFPROPdll("NITROUS OXIDE","TSAT","P;Dliq;Dvap;Hliq;VIS;Sliq",MASS_BASE_SI, 0,0,temp, 0, [1.0])

    pres=properties.Output[0]
    lden = properties.Output[1] 
    vden = properties.Output[2]
    lh = properties.Output[3]
    ldynvis = properties.Output[4]
    ls = properties.Output[5]
    
    return lden, vden, lh, pres, ldynvis, ls

def initial_U(temp, lmass, vmass):
    #calculate initial eternal energy based on temperature and mass composition
    properties=RP.REFPROPdll("NITROUS OXIDE","TSAT","Eliq;Evap",MASS_BASE_SI, 0,0,temp, 0, [1.0])
    lu = properties.Output[0]
    vu = properties.Output[1]
    U=lu*lmass+vu*vmass
    return U

def thermophys_vapour(vden):
    """Returns N2O vapour properties based on assumption that vapour leftover in tank
    stays saturated, otherwise during decompression it would condense"""
    properties=RP.REFPROPdll("NITROUS OXIDE","DQ","P;T;S;H;E",MASS_BASE_SI, 0,0,vden, 1, [1.0])
    vap_pres=properties.Output[0]
    temp=properties.Output[1]
    vs=properties.Output[2]
    vh=properties.Output[3]
    vu=properties.Output[4]
    return vap_pres, temp, vs, vh, vu

def thermophys_vent(temp):
    properties=RP.REFPROPdll("NITROUS OXIDE","TSAT","Dvap;Hvap;Svap;Eliq;Evap;Dliq",MASS_BASE_SI, 0,0,temp, 0, [1.0])
    vden=properties.Output[0]
    vh=properties.Output[1]
    vs=properties.Output[2]
    lu=properties.Output[3]
    vu=properties.Output[4]
    lden=properties.Output[5]
    return vden, vh, vs, lu, vu, lden

def chamber_vap(s, cham_pres):
    """Returns N2O specific enthalpy of vapour and vapour density at chamber
    pressure, again, 2nd argument is guess of vapour temperature

    Changed to refprop, now calculates properties of mixtrure when given isentropic flow to the chamber"""
    properties=RP.REFPROPdll("NITROUS OXIDE","PS","H;D",MASS_BASE_SI, 0,0,cham_pres, s, [1.0])
    h = properties.Output[0]
    den = properties.Output[1]
    return (h, den)

def nozzle(gamma, eps, throatA, pres_cham, eff, pres_ext):
    
    """Returns the exit mach number by numerical solution"""
    def mach_error(m):
        """Finds the discrepancy between the non-dimensional mass flow
        rate (stagnation) found using Mach number relations and that
        calculated using current guess of exit conditions"""
        m_new = np.power(
            2 / (gamma + 1) * (1 + (gamma - 1) * m * m / 2),
            (gamma + 1) / (gamma - 1) / 2
        ) / eps
        return abs(m - m_new)
    mach_exit = float(
            scipy.optimize.minimize(mach_error, 2.5, tol=1e-4, method='Powell').x)
    
    pres_exit = pres_cham * pow(1 + (gamma - 1) * mach_exit * mach_exit * 0.5, -gamma / (gamma - 1))
    thrust = eff * (
        throatA * pres_cham * np.sqrt(
            2 * gamma**2 / (gamma - 1)
            * pow(2 / (gamma + 1), (gamma + 1) / (gamma - 1))
            * (1 - pow(pres_exit / pres_cham, 1 - 1 / gamma))
        ) + (pres_exit - pres_ext) * throatA * eps)
    
    return mach_exit, pres_exit, thrust

def thermophys_liquid(U, m, V):
    u=U/m
    den=m/V
    properties=RP.REFPROPdll("NITROUS OXIDE","DE","T;Dliq;Hliq;P;VISliq;Sliq;QMASS",MASS_BASE_SI, 0,0,den, u, [1.0])
    temp=properties.Output[0]
    lden=properties.Output[1]
    lh=properties.Output[2]
    vap_pres=properties.Output[3]
    ldynvis=properties.Output[4]
    ls=properties.Output[5]
    q=properties.Output[6]
    vmass=m*q
    lmass=m*(1-q)
    return temp, lden, lh, vap_pres, ldynvis, ls, lmass, vmass

#########################################################################
#           Additional functions for tanking simulation                 #
#########################################################################

def pressure_to_temp(p):
    return RP.REFPROPdll("NITROUS OXIDE","PSAT","T",MASS_BASE_SI, 0,0,p, 0, [1.0]).Output[0]

def mass_to_headspace(temp, V, m):
    den=m/V
    properties=RP.REFPROPdll("NITROUS OXIDE","DT","QMASS;Dvap",MASS_BASE_SI, 0,0,den, temp, [1.0])
    q=properties.Output[0]
    vden=properties.Output[1]
    headspace = q*den/vden
    return headspace

def dyer_injector_tanking(p_tank, p_main, inj_dia, lden_main, h_main, s_main):
    
    #decide which values generate here and which use as input
    properties=RP.REFPROPdll("NITROUS OXIDE","PS","D;H;Q",MASS_BASE_SI, 0,0,p_tank,s_main, [1.0])
    den_tank=properties.Output[0]
    h_tank=properties.Output[1]
    q=properties.Output[2]
    
    inj_pdrop=(p_main-p_tank)
    inj_pdrop_og = 0

    A=area(inj_dia)
    Cd = 0.6
    k=1
    
    if inj_pdrop < 3e5:
        inj_pdrop_og = inj_pdrop
        inj_pdrop = 3e5
       
    mdot_spi = Cd * A * np.sqrt(2 * lden_main * inj_pdrop)
    mdot_hem = Cd * A * den_tank * np.sqrt(2 * (h_main - h_tank))
    mdot_ox = ((k * mdot_spi) + mdot_hem) / (1 + k)

    if 0 < inj_pdrop_og < 3e5:
        mdot_ox *= inj_pdrop_og / (3e5)
    
    return mdot_ox, q

def EQvent(den, u):
    properties=RP.REFPROPdll("NITROUS OXIDE","DE","QMASS;P;T;Hliq;Hvap",MASS_BASE_SI, 0,0,den,u, [1.0])
    q=properties.Output[0]
    p=properties.Output[1]
    T=properties.Output[2]
    lh=properties.Output[3]
    vh=properties.Output[4]
    return q, p, T, lh, vh


def initial_thermophys(V, headspace, m, temp, pres):
    if pres:
        properties=RP.REFPROPdll("NITROUS OXIDE","PSAT","T;Dliq;Dvap;Eliq;Evap",MASS_BASE_SI, 0,0,pres,0, [1.0])
        p=pres
        T=properties.Output[0]
    if temp:
        properties=RP.REFPROPdll("NITROUS OXIDE","TSAT","P;Dliq;Dvap;Eliq;Evap",MASS_BASE_SI, 0,0,temp,0, [1.0])
        T=temp
        p=properties.Output[0]
        
    lden=properties.Output[1]
    vden=properties.Output[2]
    if m:
        q=vden*V/m
        lmass=m*(1-q)
        vmass=m*q
        tmass=m
    if headspace:
        lmass = V * (1 - headspace) * lden
        vmass = V * headspace * vden
        q=vmass/(lmass+vmass)
        tmass=lmass+vmass
    lu = properties.Output[3]
    vu = properties.Output[4]
    U=lu*lmass+vu*vmass
    
    return p, T, lden, vden, lmass, vmass, U
        