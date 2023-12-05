import control as ctrl
from control import TransferFunction as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
import math 
from scipy.integrate import odeint

# import warnings

# # Disable specific warning types
# warnings.filterwarnings("ignore", category=RuntimeWarning)
# warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)

R_VALUES = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]

C_VALUES = [10, 11, 12, 13, 15, 16, 18, 20, 22, 24, 27, 30, 33, 36, 39, 43, 47, 51, 56, 62, 68, 75, 82, 91]

def is_system_stable(system):
    poles = ctrl.pole(system)
    
    # Check if all poles have negative real parts for continuous-time
    if isinstance(system, ctrl.TransferFunction):
        check = all(p.real < 0 for p in poles)
        # if not check:
        #     print('System not stable!')
        return check
    
    # Check if all poles are within the unit circle for discrete-time
    elif isinstance(system, ctrl.TransferFunctionDiscrete):
        check = all(abs(p) < 1 for p in poles)
        # if not check:
        #     print('System not stable!')
        return check
    
    return False  # Return False for unsupported system types

def convert_sol2RC(x):
    R = np.array([R_VALUES[x[i]]*10**(x[i+1]) for i in range(0, 8, 2)],dtype=np.int64)
    C = np.array([(C_VALUES[x[i]])*10**(float(x[i+1])) for i in range(8, 11, 2)], dtype=np.float64)
    
    Kp = R[3]*(R[0]*C[0] + R[1]*C[1])/(R[2]*R[0]*C[1])
    Kd = R[3]*R[1]*C[0]/R[2]
    Ki = R[3]/(R[0]*R[2]*C[1])
    return Kp, Kd, Ki

def create_control_system_dc_motor(motor='', num=[1], den=[1], v_max=25, max_derivative = 5.0, T=15):
    # Arrays to store results
    def create_pid_control(x):
        def step_input(t):
            return setpoint if t >= 0.1 else 0

        t = np.linspace(0, 15, 5000)

        # Desired angular velocity (setpoint)
        setpoint = 1

        # Initial conditions
        initial_state = [0, 0]

        Kp, Kd, Ki = convert_sol2RC(x)

        plant = tf(num, den)  

        pid_controller = tf([Kd, Kp, Ki], [1, 0])

        closed_loop_system = ctrl.feedback(pid_controller * plant)


        time_values = []
        response_values = []
        integral_error = 0
        prev_error = 0
        prev_control_signal = 0
        inputs = []
        # Simulate the system's response with PID control and voltage saturation
        for i in range(len(t)):
            error = step_input(t[i]) - response_values[-1][0] if i > 0 else setpoint
            
            # Calculate integral error using running sum
            integral_error += error * (t[i] - (t[i-1] if i > 0 else 0))
            
            # Calculate derivative of the error
            derivative_error = (error - prev_error) / (t[i] - (t[i-1] if i > 0 else 0))
            
            # Calculate control signal (PID) with derivative limitation
            control_signal = Kp * error + Ki * integral_error + Kd * derivative_error
            
            # Apply saturation to the control signal
            control_signal = np.clip(control_signal, -v_max, v_max)
            
            # Apply derivative limitation
            derivative_limit = max_derivative * (control_signal - prev_control_signal) / (t[i] - (t[i-1] if i > 0 else 0))
            control_signal = prev_control_signal + np.clip(derivative_limit, -max_derivative, max_derivative) * (t[i] - (t[i-1] if i > 0 else 0))
            
            sol = odeint(motor, initial_state, [t[i], t[i] + t[1]], args=(control_signal,))
            initial_state = sol[1]
            
            time_values.append(t[i])
            response_values.append(sol[0])
            inputs.append(control_signal)
            prev_error = error
            prev_control_signal = control_signal

        time_values = np.array(time_values)
        response_values = np.array(response_values)

        angular_velocity = response_values[:, 0]
        max_overshoot = max(angular_velocity) - setpoint
        try:
            rise_time = next(t[i] for i, value in enumerate(angular_velocity) if value >= setpoint * 0.90) - t[0]
        except:
            rise_time = np.nan
        try:
            settling_time = next(i for i, value in enumerate(angular_velocity) if abs(value - setpoint) <= 0.02 * setpoint) * (t[1] - t[0])
        except:
            settling_time = np.nan

        # print(settling_time, rise_time, max_overshoot)

        return settling_time, rise_time, max_overshoot, time_values, response_values, closed_loop_system, (Kp, Ki, Kd)
    return create_pid_control


def create_control_system_circuit(num=[1], den=[1], T=15):
    def create_pid_control(x):
        plant = tf(num, den)  

        Kp, Kd, Ki = convert_sol2RC(x)
        #print(f'Kp = {Kp}, Kd = {Kd}, Ki = {Ki}')

        pid_controller = tf([Kd, Kp, Ki], [1, 0])

        closed_loop_system = ctrl.feedback(pid_controller * plant)
        time, response = ctrl.step_response(closed_loop_system, T=T)

        try:
            info = ctrl.step_info(closed_loop_system, T=T)
        except:
            info = {'SettlingTime': np.nan, 'RiseTime': np.nan, 'Overshoot': np.nan}
        settling_time = info['SettlingTime']
        rise_time = info['RiseTime']
        overshoot = info['Overshoot']

        return settling_time, rise_time, overshoot, time, response, closed_loop_system, (Kp, Ki, Kd)
    return create_pid_control

def create_control_system_num(num=[1], den=[1], T=15):
    def create_pid_control_num(x):
        plant = tf(num, den)  

        pid_controller = tf([x[0], x[1], x[2]], [1, 0])

        closed_loop_system = ctrl.feedback(pid_controller * plant)
        time, response = ctrl.step_response(closed_loop_system, T=T)

        try:
            info = ctrl.step_info(closed_loop_system, T=T)
        except:
            info = {'SettlingTime': 100, 'RiseTime': 100, 'Overshoot': 100}
        settling_time = info['SettlingTime']
        rise_time = info['RiseTime']
        overshoot = info['Overshoot']

        return settling_time, rise_time, overshoot, time, response, closed_loop_system
    return create_pid_control_num


def plot_response(system, T=15):
    time, response = ctrl.step_response(system, T=T)
    info = ctrl.step_info(system, T=T)
    settling_time = info['SettlingTime']
    rise_time = info['RiseTime']
    overshoot = info['Overshoot']
    peak = info['Peak']
    plt.plot(time, response, linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Response')
    plt.title(f'Step Response of PID-Controlled System. \n ST = {settling_time}, RT = {rise_time}, Overshoot = {overshoot}')
    plt.grid()
    plt.show()


def fuzzy_aggregator_circuit_fitness(fuzzy_evaluator, create_system=create_control_system_circuit, scale=[1, 1], show_plot=False, pygad=False, T=15, **kwargs):
    plant_control = create_system(**kwargs, T=T)
    def eval_study_case(x):
        settling_time, rise_time, overshoot, time, response, closed_loop_system, Ks = plant_control(x.astype(int))
        
        if not is_system_stable(closed_loop_system):
            # print('System not stable')
            return 1.0
        if any(num > 100 for num in Ks):
            # print('There are values of gains greater than 100')
            return 1.0

        if np.isnan(settling_time):
            #print('Settling time is NaN')  
            #settling_time = 1
            return 1.0
        elif settling_time/scale[0] > 1:
            
            settling_time = 1
        else:
            settling_time /= scale[0]
        print(f'Settling time: {settling_time:.2f} s, rise time: {rise_time:.2f} s, overshoot: {overshoot:.2f} %')

        fuzzy_evaluator.input['settling_time'] = settling_time
        if rise_time/scale[1] > 1:
            rise_time = 1
        elif np.isnan(rise_time):
            #print('Rising time is NaN')
            #rise_time = 1
            return 1.0
        else:
            rise_time /= scale[1]
        fuzzy_evaluator.input['rise_time'] = rise_time
        if overshoot/100 > 1:
            overshoot = 1
        elif np.isnan(overshoot):
            #print('Overshooting is NaN')
            #overshoot = 1
            return 1.0
        else:
            overshoot /= 100
        fuzzy_evaluator.input['overshoot'] = overshoot

        fuzzy_evaluator.compute()
        if show_plot:
            print(f'{settling_time, rise_time, overshoot}')
            # evaluation.view(sim=fuzzy_evaluator)
            # plt.figure()
            try:
                plot_response(closed_loop_system, T=T)
            except:
                print('Could not run function plot response')
    
        return fuzzy_evaluator.output['evaluation']
    return eval_study_case


def weighted_sum_circuit_fitness(create_system=create_control_system_circuit, scale=[1, 1], show_plot=False, pygad=False, T=15, **kwargs):
    plant_control = create_system(**kwargs, T=T)
    if pygad: 
        def eval_study_case(ga_instance, x, x_idx):
            settling_time, rise_time, overshoot, time, response, closed_loop_system, Ks = plant_control(x.astype(int))
            #print(f'Settling time: {settling_time:.2f} s, rise time: {rise_time:.2f} s, overshoot: {overshoot:.2f} %')

            if not is_system_stable(closed_loop_system):
                # print('System not stable')
                return 1.0
            if any(num > 100 for num in Ks):
                # print('There are values of gains greater than 100')
                return 1.0

            if np.isnan(settling_time):
                #print('Settling time is NaN')  
                #settling_time = 1
                return 1.0
            elif settling_time/scale[0] > 1:
                settling_time = 1
            else:
                settling_time /= scale[0]

            if rise_time/scale[1] > 1:
                rise_time = 1
            elif np.isnan(rise_time):
                #print('Rising time is NaN')
                #rise_time = 1
                return 1.0
            else:
                rise_time /= scale[1]

            if overshoot/100 > 1:
                overshoot = 1
            elif np.isnan(overshoot):
                #print('Overshooting is NaN')
                #overshoot = 1
                return 1.0
            else:
                overshoot /= 100

            if show_plot:
                print(f'{settling_time, rise_time, overshoot}')
                # evaluation.view(sim=fuzzy_evaluator)
                # plt.figure()
                try:
                    plot_response(closed_loop_system, T=T)
                except:
                    print('Could not run function plot response')

        
            return (settling_time + rise_time + overshoot)/3
    else:
        def eval_study_case(x):
            settling_time, rise_time, overshoot, time, response, closed_loop_system, Ks = plant_control(x.astype(int))
            #print(f'Settling time: {settling_time:.2f} s, rise time: {rise_time:.2f} s, overshoot: {overshoot:.2f} %')

            if not is_system_stable(closed_loop_system):
                # print('System not stable')
                return 1.0
            if any(num > 100 for num in Ks):
                # print('There are values of gains greater than 100')
                return 1.0

            if np.isnan(settling_time):
                #print('Settling time is NaN')  
                #settling_time = 1
                return 1.0
            elif settling_time/scale[0] > 1:
                settling_time = 1
            else:
                settling_time /= scale[0]

            if rise_time/scale[1] > 1:
                rise_time = 1
            elif np.isnan(rise_time):
                #print('Rising time is NaN')
                #rise_time = 1
                return 1.0
            else:
                rise_time /= scale[1]

            if overshoot/100 > 1:
                overshoot = 1
            elif np.isnan(overshoot):
                #print('Overshooting is NaN')
                #overshoot = 1
                return 1.0
            else:
                overshoot /= 100

            if show_plot:
                print(f'{settling_time, rise_time, overshoot}')
                # evaluation.view(sim=fuzzy_evaluator)
                # plt.figure()
                try:
                    plot_response(closed_loop_system, T=T)
                except:
                    print('Could not run function plot response')

        
            return (settling_time + rise_time + overshoot)/3
    return eval_study_case

def mono_objective_circuit_fitness(create_system=create_control_system_circuit, show_plot=False, pygad=False, T=15, **kwargs):
    plant_control = create_system(**kwargs, T=T)
    if pygad: 
        def eval_study_case(ga_instance, x, x_idx):
            settling_time, rise_time, overshoot, time, response, closed_loop_system, Ks = plant_control(x.astype(int))
            #print(f'Settling time: {settling_time:.2f} s, rise time: {rise_time:.2f} s, overshoot: {overshoot:.2f} %')

            if not is_system_stable(closed_loop_system):
                # print('System not stable')
                return 100000
            if any(num > 100 for num in Ks):
                # print('There are values of gains greater than 100')
                return 100000

            if np.isnan(settling_time) or np.isnan(rise_time) or np.isnan(overshoot):
                return 100000

            error = np.sqrt(np.mean((response - 1)**2))

            if show_plot:
                print(f'{settling_time, rise_time, overshoot}')
                # evaluation.view(sim=fuzzy_evaluator)
                # plt.figure()
                try:
                    plot_response(closed_loop_system, T=T)
                except:
                    print('Could not run function plot response')

        
            return error
    else:
        def eval_study_case(x):
            settling_time, rise_time, overshoot, time, response, closed_loop_system, Ks = plant_control(x.astype(int))
            #print(f'Settling time: {settling_time:.2f} s, rise time: {rise_time:.2f} s, overshoot: {overshoot:.2f} %')

            if not is_system_stable(closed_loop_system):
                # print('System not stable')
                return 100000
            if any(num > 100 for num in Ks):
                # print('There are values of gains greater than 100')
                return 100000

            error = np.sqrt(np.mean((response - 1)**2))

            if np.isnan(settling_time) or np.isnan(rise_time) or np.isnan(overshoot):
                return 100000

            if show_plot:
                print(f'{settling_time, rise_time, overshoot}')
                # evaluation.view(sim=fuzzy_evaluator)
                # plt.figure()
                try:
                    plot_response(closed_loop_system, T=T)
                except:
                    print('Could not run function plot response')

        
            return error
    return eval_study_case













def fuzzy_aggregator_parameters_fitness(num, den, fuzzy_evaluator, scale=[1, 1], show_plot=False, pygad=False):
    plant_control = create_control_system_num(num, den)
    if pygad:
        def eval_study_case(ga_instance, x, x_idx):
            settling_time, rise_time, overshoot, time, response, closed_loop_system = plant_control(x)
            # print(f'Settling time: {settling_time:.2f} s, rise time: {rise_time:.2f} s, overshoot: {overshoot:.2f} %')
            if not is_system_stable(closed_loop_system):
                return 1.0

            if settling_time/scale[0] > 1 or settling_time == np.nan:
                settling_time = 1
            else:
                settling_time /= scale[0]
            
            fuzzy_evaluator.input['settling_time'] = settling_time
            if rise_time/scale[1] > 1 or rise_time == np.nan:
                rise_time = 1
            else:
                rise_time /= scale[1]
            fuzzy_evaluator.input['rise_time'] = rise_time
            if overshoot/100 > 1 or overshoot == np.nan:
                overshoot = 1
            else:
                overshoot /= 100 
            fuzzy_evaluator.input['overshoot'] = overshoot

            fuzzy_evaluator.compute()
            if show_plot:
                print(f'{settling_time, rise_time, overshoot}')
                # evaluation.view(sim=fuzzy_evaluator)
                # plt.figure()
                plot_response(closed_loop_system, T=15)
            
            return fuzzy_evaluator.output['evaluation']
    else:
        def eval_study_case(x):
            settling_time, rise_time, overshoot, time, response, closed_loop_system = plant_control(x)
            # print(f'Settling time: {settling_time:.2f} s, rise time: {rise_time:.2f} s, overshoot: {overshoot:.2f} %')
            if not is_system_stable(closed_loop_system):
                return 1.0
  
            if settling_time/scale[0] > 1 or settling_time == np.nan:
                settling_time = 1
            else:
                settling_time /= scale[0]
            
            fuzzy_evaluator.input['settling_time'] = settling_time
            if rise_time/scale[1] > 1 or rise_time == np.nan:
                rise_time = 1
            else:
                rise_time /= scale[1]
            fuzzy_evaluator.input['rise_time'] = rise_time
            if overshoot/100 > 1 or overshoot == np.nan:
                overshoot = 1
            else:
                overshoot /= 100 
            fuzzy_evaluator.input['overshoot'] = overshoot

            fuzzy_evaluator.compute()
            if show_plot:
                print(f'{settling_time, rise_time, overshoot}')
                # evaluation.view(sim=fuzzy_evaluator)
                # plt.figure()
                plot_response(closed_loop_system, T=15)
            
            return fuzzy_evaluator.output['evaluation']
    return eval_study_case