# Importing the Libraries
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import argparse


def main():
    
    # Taking the User Input from the terminal
    parser = argparse.ArgumentParser(description="Fuzzy Project Assesment")
    parser.add_argument("--funding", type=float,required=True, help="Funding Value (0 to 100)")
    parser.add_argument("--staffing", type=float,required=True, help=" Staff Required (0 to 100)")
    arg = parser.parse_args()
    
    # Defining the Lingustic Variables
    funding = ctrl.Antecedent(np.arange(0,101,1),"funding")
    staffing = ctrl.Antecedent(np.arange(0,101,1),"staffing")
    risk = ctrl.Consequent(np.arange(0,101,1),"risk")
    
    # Defining the membership function for each lingusitic variable

    # Lingustic Variable : funding
    funding["inadequate"] = fuzz.trapmf(funding.universe, [0,0,30,45])
    funding["marginal"] = fuzz.trimf(funding.universe, [30, 50, 70])
    funding["adequate"] = fuzz.trapmf(funding.universe,[55, 70, 100, 100])

    # Lingustic Variable: Staffing
    staffing["small"] = fuzz.trapmf(staffing.universe,[0,0,25,65])
    staffing["large"] = fuzz.trapmf(staffing.universe,[35, 75, 100, 100])

    # Linguistic Variable: risk
    risk["low"] = fuzz.trapmf(risk.universe, [0, 0, 20, 40])
    risk["medium"] = fuzz.trapmf(risk.universe,[25, 45, 55, 75])
    risk["high"] = fuzz.trapmf(risk.universe,[0, 80, 100, 100])
    
    # Debugging the fuzzy system
    #funding.view()
    #staffing.view()
    #risk.view()
    
    # Defining the Rules
    
    rule1 = ctrl.Rule(funding["adequate"] | staffing["small"], risk["low"])
    rule2 = ctrl.Rule(funding["marginal"] & staffing["large"], risk["medium"])
    rule3 = ctrl.Rule(funding["inadequate"], risk["high"])
    
    
    # Build the Fuzzy Systems from the Rules
    
    ctrl_sys = ctrl.ControlSystem([rule1,rule2,rule3])
    ctrl_sim = ctrl.ControlSystemSimulation(ctrl_sys)
    
    # Use the control system to assess the project risk
    ctrl_sim.inputs({"funding":arg.funding, "staffing":arg.staffing})
    
    # crunch the output
    ctrl_sim.compute()
    
    print("The project Risk:", ctrl_sim.output["risk"],"%")    


if __name__ == "__main__":
    
    main()

    
    
