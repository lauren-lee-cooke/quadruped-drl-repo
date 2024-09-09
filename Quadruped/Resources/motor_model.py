class LinearMotorModel:
    def __init__(self, tau_max, omega_max):
        self.tau_max = tau_max
        self.omega_max = omega_max
    
    def compute_torque_limits(self, omega):
        """
        Computes the torque limits for a given angular velocity,
        ensuring the torque does not exceed the maximum torque.

        Parameters:
        omega (float): The angular velocity.

        Returns:
        tuple: A tuple containing the lower and upper torque limits.
        """
        # Calculate the initial torque limits based on the linear model
        lower_limit = -self.tau_max - (self.tau_max / self.omega_max) * omega
        upper_limit = self.tau_max - (self.tau_max / self.omega_max) * omega

        # Clamp the limits to the maximum torque bounds
        lower_limit = max(lower_limit, -self.tau_max)
        upper_limit = min(upper_limit, self.tau_max)
        
        return lower_limit, upper_limit

# Example usage
# if __name__ == "__main__":
#     tau_max = 33.5  # Example stall torque (Nm)
#     omega_max = 21.0  # Example no-load speed (rad/s)
    
#     motor_model = LinearMotorModel(tau_max, omega_max)
    
#     omega = 15.0  # Example angular velocity (rad/s)
#     lower_limit, upper_limit = motor_model.compute_torque_limits(omega)
    
#     print(f"At omega = {omega} rad/s:")
#     print(f"Lower torque limit: {lower_limit} Nm")
#     print(f"Upper torque limit: {upper_limit} Nm")