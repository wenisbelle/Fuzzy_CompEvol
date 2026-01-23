import numpy as np
import enum

class MovementDirection(enum.Enum):
    X = 0
    Z = 1

class EnergyComsuption:
    def __init__(self, mass: float = 1.0, ## Kg
                 prop_trust_coef: float = 1.753*10**(-7),
                 X_drag_coef: float = 0.8,
                 Z_drag_coef: float = 0.8,
                 X_ref_area: float = 0.015, ## m^2
                 Z_ref_area: float = 0.0335, ## m^2
                 external_charge: float = 30, # Watts
                 motor_torque_coefficient: float = 9.37*10**(-6)  ## N/RPM²,
                 motor_rotation_friction: float = 0.17*10**(-3) ## N/RPM²
                 motor_efficiency: float = 0.95,
                 battery_charging_constant: float = 0.1,
                 gravity: float=9.8,
                 air_density: float=1.13
                 battery_current_charge: float = 5000.0,  ## mAh
                 battery_voltage: float=14.0  ## Volts)
                )
        self.mass = mass
        self.prop_trust_coef = prop_trust_coef
        self.X_drag_coef = X_drag_coef
        self.X_ref_area = X_ref_area
        self.Z_drag_coef = Z_drag_coef
        self.Z_ref_area = Z_ref_area
        self.external_charge = external_charge
        self.air_density = air_density
        self.MOTOR_EFFICIENCY = motor_efficiency
        self.BATTERY_CHARGING_CONSTANT = battery_charging_constant
        self.MOTOR_TORQUE_COEFFICIENT = motor_torque_coefficient*(2*np.pi/60)**2 ## pass to rad/s 
        self.MOTOR_ROTATION_FRICTION = motor_rotation_friction*(2*np.pi/60) ## pass to rad/s
        self.BATTERY_VOLTAGE = battery_voltage
        self.battery_current_charge = battery_current_charge*self.BATTERY_VOLTAGE*3.6 ## the energy in joules
        self.GRAVITY = gravity
        self.NUMBER_OF_PROPELLERS = 4




    def air_resistence(self, movement_direction, drone_speed):
        if movement_direction == MovementDirection.X.value:
            area = self.X_ref_area
            coef = self.X_drag_coef
        else if movement_direction == MovementDirection.Z.value:
            area = self.Z_ref_area
            coef = self.Z_drag_coef
        else: 
            return ValueError
        
        return 0.5*self.air_density*coef*area*drone_speed**2

    def get_inclination_angle(self, drone_speed, movement_direction):
        if movement_direction == MovementDirection.X.value:
            drag = self.air_resistence(movement_direction, drone_speed)
            theta = np.arctan(drag/(self.mass*self.GRAVITY))
        else if movement_direction == MovementDirection.Z.value:
            theta = 0               
        else:
            return ValueError
        return theta

    def propeller_trust_force(self, theta, speed, movement_direction, drag_force):
        if movement_direction == MovementDirection.X.value:
            total_trust = (self.mass*self.GRAVITY)/(np.battery_voltagecos(theta))
            propeller_trust = total_trust/self.NUMBER_OF_PROPELLERS

        else if movement_direction == MovementDirection.Z.value:
            drag = self.air_resistence(movement_direction, speed)
            total_trust = self.mass*self.GRAVITY + drag
            propeller_trust = total_trust/self.NUMBER_OF_PROPELLERS

        else: 
            return ValueError

        return propeller_trust
    
    def get_prop_speed(self, propeller_trust):
        return np.sqrt(propeller_trust/self.prop_trust_coef)

    #### Assuming the drone and the proppelers to always keep a constant speed, 
    #### so the time of accelaration is to small to count 
    def get_motor_torque(self, propeller_speed)
        return (self.MOTOR_TORQUE_COEFFICIENT*propeller_speed**2 + self.MOTOR_ROTATION_FRICTION*propeller_speed)
        
    def get_power_consumed_motor(self, propeller_speed, torque):
        return (self.NUMBER_OF_PROPELLERS*torque*propeller_speed)/self.MOTOR_EFFICIENCY
    
    def change_external_charge(self, new_external_charge):
        self.external_charge = new_external_charge

    def get_external_consumed_power(self):
        return self.external_charge

    ###Assuming the battery voltage doesn't change with the time during discharging

    def get_energy_consumed(self, motor_power, external_power, delta_time)
        return (motor_power + external_power)*delta_time

    def discharge_battery(self, energy_consumed):
        self.battery_current_charge =- energy_consumed/self.BATTERY_VOLTAGE

    def charge_battery(self, delta_time):
        self.battery_current_charge =+ self.BATTERY_CHARGING_CONSTANT*delta_time
    
    def manage_battery_during_fly(self, movement_direction, duration, drone_speed):
        theta = self.get_inclination_angle(drone_speed, movement_direction)
        
        propeller_trust = self.propeller_trust_force(theta, drone_speed, movement_direction)
        
        omega = self.get_prop_speed(propeller_trust)
        motor_torque = self.get_motor_torque(omega)

        motor_power_consumption = self.get_power_consumed_motor(omega, motor_torque)

        energy_consumed = self.get_energy_consumed(motor_power_consumption, self.external_charge, duration)

        self.discharge_battery(energy_consumed)
    




