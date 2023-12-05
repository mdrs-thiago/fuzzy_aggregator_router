import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as fctrl

# New Antecedent/Consequent objects hold universe variables and membership
# functions
def create_variables():
    settling_time = fctrl.Antecedent(np.arange(0, 1, 0.05), 'settling_time')
    rise_time = fctrl.Antecedent(np.arange(0, 1, 0.05), 'rise_time')
    overshoot = fctrl.Antecedent(np.arange(0, 1, 0.05), 'overshoot')
    evaluation = fctrl.Consequent(np.arange(0, 1.25, 0.05), 'evaluation')

    # Auto-membership function population is possible with .automf(3, 5, or 7)
    settling_time['low'] = fuzz.trimf(settling_time.universe, [0, 0, 0.25])
    settling_time['medium'] = fuzz.trimf(settling_time.universe, [0, 0.25, 0.5])
    settling_time['high'] = fuzz.trapmf(settling_time.universe, [0.25, 0.5, 1.0, 1.1])

    #Auto-membership for the rise time
    rise_time['low'] = fuzz.trimf(rise_time.universe, [0, 0, 0.25])
    rise_time['medium'] = fuzz.trimf(rise_time.universe, [0, 0.25, 0.5])
    rise_time['high'] = fuzz.trimf(rise_time.universe, [0.25, 0.5, 0.75])
    rise_time['very_high'] = fuzz.trapmf(rise_time.universe, [0.5, 0.75, 1.0, 1.1])

    overshoot['low'] = fuzz.trimf(overshoot.universe, [0, 0, 0.25])
    overshoot['medium'] = fuzz.trimf(overshoot.universe, [0, 0.25, 0.5])
    overshoot['high'] = fuzz.trimf(overshoot.universe, [0.25, 0.5, 0.75])
    overshoot['very_high'] = fuzz.trapmf(overshoot.universe, [0.5, 0.75, 1.0, 1.1])

    # Custom membership functions can be built interactively with a familiar,
    # Pythonic API
    evaluation['very_good'] = fuzz.trimf(evaluation.universe, [0, 0, 0.1])
    evaluation['good'] = fuzz.trimf(evaluation.universe, [0, 0.1, 0.25])
    evaluation['medium'] = fuzz.trimf(evaluation.universe, [0.1, 0.25, 0.5])
    evaluation['bad'] = fuzz.trimf(evaluation.universe, [0.25, 0.5, 0.75])
    evaluation['very_bad'] = fuzz.trapmf(evaluation.universe, [0.5, 0.75, 1.0, 1.2])

    return settling_time, rise_time, overshoot, evaluation

def create_rules(settling_time, rise_time, overshoot, evaluation):
    list_rules = []
    # If overshoot is low and rise time is low, then the evaluation is very good
    list_rules.append(fctrl.Rule(overshoot['low'] & rise_time['low'], evaluation['very_good']))

    #If overshoot is low and rise time is medium, then the evaluation is good
    list_rules.append(fctrl.Rule(overshoot['low'] & rise_time['medium'], evaluation['good']))

    #If overshoot is medium and rise time is low, then the evaluation is good
    list_rules.append(fctrl.Rule(overshoot['medium'] & rise_time['low'], evaluation['good']))

    #If overshoot is medium and rise time is medium, then the evaluation is medium
    list_rules.append(fctrl.Rule(overshoot['medium'] & rise_time['medium'], evaluation['medium']))

    #If overshoot is high, then the evaluation is bad
    list_rules.append(fctrl.Rule(overshoot['high'], evaluation['bad']))

    #If rise time is high, then the evaluation is bad
    list_rules.append(fctrl.Rule(rise_time['high'], evaluation['bad']))

    #If overshoot is very high, then the evaluation is very bad
    list_rules.append(fctrl.Rule(overshoot['very_high'], evaluation['very_bad']))

    #If rise time is very high, then the evaluation is very bad
    list_rules.append(fctrl.Rule(rise_time['very_high'], evaluation['very_bad']))

    #If settling time is low, then the evaluation is very good
    list_rules.append(fctrl.Rule(settling_time['low'], evaluation['very_good']))

    #If settling time is medium, then the evaluation is good
    list_rules.append(fctrl.Rule(settling_time['medium'], evaluation['good']))

    #If settling time is high, then the evaluation is very bad
    list_rules.append(fctrl.Rule(settling_time['high'], evaluation['very_bad']))
    return list_rules 

def create_aggregator():
    settling_time, rise_time, overshoot, evaluation = create_variables()    
    list_rules = create_rules(settling_time, rise_time, overshoot, evaluation)    
    fuzzy_aggregation = fctrl.ControlSystem(list_rules)

    return fctrl.ControlSystemSimulation(fuzzy_aggregation)
