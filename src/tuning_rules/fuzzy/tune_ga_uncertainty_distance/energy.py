import numpy as np
import enum

class MovementDirection(enum.Enum):
    X = 0
    Z = 1

class BatteryError(Exception):
    """Battery without energy"""
    pass

class EnergyComsuption:
    def __init__(self, mass: float = 1.5, ## Kg
                 payload: float = 0.5, ## kg, battery more payload
                 power_efficiency: float = 0.5, ### from the article
                 lift_drag_ration: float = 3.0, ### benchmarket 
                 external_power: float =10.0, # Watts
                 battery_charging_constant: float = 0.01,
                 battery_capacity: float = 5000.0,  ## mAh
                 battery_voltage: float=14.0,  ## Volts)
                 battery_initial_status: float = 1.0
                ):
        self.mass = mass
        self.payload = payload
        self.external_power = external_power

        self.POWER_EFFICIENCY = power_efficiency
        self.RATIO = lift_drag_ration
        self.BATTERY_CHARGING_CONSTANT = battery_charging_constant
        self.BATTERY_VOLTAGE = battery_voltage
        
        self.GRAVITY = 9.18
        self.AIR_DENSITY = 1.225
        self.RADIUS_PROP = 0.127
        self.NUMBER_ROTORS = 4
        self.disk_area = self.NUMBER_ROTORS*np.pi*(self.RADIUS_PROP**2)

        self.battery_total_energy = battery_capacity*self.BATTERY_VOLTAGE*3.6 ## the energy in joules
        self.battery_current_energy = battery_initial_status*self.battery_total_energy
        self.battery_status = battery_initial_status


    def get_hover_power(self):
        
        total_mass = self.mass + self.payload
        thrust_needed = total_mass * self.GRAVITY

        ideal_hover_power = (thrust_needed**1.5) / np.sqrt(2 * self.AIR_DENSITY * self.disk_area)

        # This values is changed to match the information presented in the holybro site:
        ### Flight time: ~18 minutes hover with no additional payload. Tested with 5000mAh Battery.
        return ideal_hover_power / 0.6

    
    def get_power_consumed(self, drone_speed):
        total_mass = self.mass + self.payload
        hover_power = self.get_hover_power()
        #print(f"THe hover power is: {hover_power}")
        return total_mass*self.GRAVITY*drone_speed/(self.POWER_EFFICIENCY*self.RATIO) + self.external_power + hover_power

   
    def change_external_power(self, new_external_charge):
        self.external_charge = new_external_charge

    ###Assuming the battery voltage doesn't change with the time during discharging

    def get_energy_consumed(self, power_consumed, delta_time):
        return power_consumed*delta_time

    def discharge_battery(self, energy_consumed):
        self.battery_current_energy -= energy_consumed

        if self.battery_current_energy <= 0.0:
            self.battery_current_energy = 0.0 
            self.update_battery_status()       
            raise BatteryError("Drone has no energy.")


    def charge_battery(self, delta_time):
        self.battery_status += self.BATTERY_CHARGING_CONSTANT*delta_time

        self.battery_status = max(1.0, self.battery_status)
    
    def get_current_battery_energy(self):
        return self.battery_current_energy
    
    def get_battery_status(self):
        return self.battery_status

    def update_battery_status(self):
        self.battery_status = (self.battery_current_energy/self.battery_total_energy)
    
    
    def manage_battery_during_fly(self, duration, drone_speed):
        power = self.get_power_consumed(drone_speed)

        energy_consumed = self.get_energy_consumed(power, duration)

        self.discharge_battery(energy_consumed)

        self.update_battery_status()

        return self.battery_status
    




