#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Jam Simulator
Simulates traffic jam formation on a circular road with deceleration zones.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Wedge
import matplotlib.patches as mpatches


class Vehicle:
    """Represents a single vehicle on the circular road"""
    
    def __init__(self, angle, speed, max_speed):
        """
        Initialize vehicle
        
        Args:
            angle: Initial angular position (radians)
            speed: Initial angular velocity (rad/s)
            max_speed: Maximum angular velocity
        """
        self.angle = angle
        self.speed = speed
        self.max_speed = max_speed
        
    def update_position(self, dt):
        """Update vehicle position based on current speed"""
        self.angle += self.speed * dt
        # Keep angle in [0, 2π]
        self.angle = self.angle % (2 * np.pi)


class TrafficSimulator:
    """Traffic jam simulator on a circular road"""
    
    # Reference: 120 km/h = one lap per minute = 2π/60 rad/s
    # Assumed circular road circumference: 2000m (1 rad ≈ 318.3m)
    REFERENCE_SPEED_KMPH = 120  # km/h for one lap per minute
    REFERENCE_SPEED_RADS = 2 * np.pi / 60  # rad/s
    CIRCUMFERENCE_M = 2000  # meters
    RADIUS_M = CIRCUMFERENCE_M / (2 * np.pi)  # ≈ 318.3m
    
    def __init__(self, 
                 num_vehicles=30,
                 max_speed_kmh=100,
                 decel_zone_center_deg=90,
                 decel_zone_width_deg=40,
                 decel_zone_speed_kmh=65,
                 safe_distance_m=50,
                 brake_distance_m=35,
                 accel_rate_mps2=2.0,
                 decel_rate_mps2=4.9):
        """
        Initialize traffic simulator with intuitive real-world parameters
        
        Args:
            num_vehicles: Number of vehicles on the road
            max_speed_kmh: Maximum speed (km/h)
            decel_zone_center_deg: Center angle of deceleration zone (degrees)
            decel_zone_width_deg: Width of deceleration zone (degrees)
            decel_zone_speed_kmh: Speed in deceleration zone (km/h)
            safe_distance_m: Safe following distance (meters)
            brake_distance_m: Distance at which to start braking (meters)
            accel_rate_mps2: Acceleration rate (m/s²)
            decel_rate_mps2: Deceleration rate (m/s²)
        """
        self.num_vehicles = num_vehicles
        
        # Convert speeds from km/h to rad/s
        self.max_speed = self._kmh_to_rads(max_speed_kmh)
        decel_zone_speed_rads = self._kmh_to_rads(decel_zone_speed_kmh)
        self.decel_zone_speed_factor = decel_zone_speed_rads / self.max_speed
        
        # Convert deceleration zone from degrees to radians
        decel_zone_center_rad = np.radians(decel_zone_center_deg)
        decel_zone_half_width_rad = np.radians(decel_zone_width_deg / 2)
        self.decel_zone_start = decel_zone_center_rad - decel_zone_half_width_rad
        self.decel_zone_end = decel_zone_center_rad + decel_zone_half_width_rad
        
        # Convert distances from meters to radians
        self.safe_distance = self._meters_to_rads(safe_distance_m)
        self.brake_distance = self._meters_to_rads(brake_distance_m)
        
        # Convert acceleration/deceleration from m/s² to rad/s per frame (dt=0.1s)
        self.accel_rate = self._mps2_to_rads_per_frame(accel_rate_mps2)
        self.decel_rate = self._mps2_to_rads_per_frame(decel_rate_mps2)
        
        # Store original values for display
        self.max_speed_kmh = max_speed_kmh
        self.decel_zone_speed_kmh = decel_zone_speed_kmh
        self.safe_distance_m = safe_distance_m
        self.brake_distance_m = brake_distance_m
        self.accel_rate_mps2 = accel_rate_mps2
        self.decel_rate_mps2 = decel_rate_mps2
        
        # Initialize vehicles
        self.vehicles = []
        for i in range(num_vehicles):
            angle = (2 * np.pi / num_vehicles) * i
            vehicle = Vehicle(angle, self.max_speed, self.max_speed)
            self.vehicles.append(vehicle)
            
        self.time = 0
    
    def _kmh_to_rads(self, speed_kmh):
        """Convert speed from km/h to rad/s"""
        # 120 km/h = 2π/60 rad/s (one lap per minute)
        return speed_kmh * self.REFERENCE_SPEED_RADS / self.REFERENCE_SPEED_KMPH
    
    def _meters_to_rads(self, distance_m):
        """Convert distance from meters to radians"""
        # distance [m] / radius [m] = angle [rad]
        return distance_m / self.RADIUS_M
    
    def _mps2_to_rads_per_frame(self, accel_mps2, dt=0.1):
        """Convert acceleration from m/s² to rad/s per frame"""
        # acceleration [m/s²] / radius [m] = angular acceleration [rad/s²]
        # angular acceleration [rad/s²] * dt [s] = angular velocity change per frame [rad/s]
        angular_accel = accel_mps2 / self.RADIUS_M
        return angular_accel * dt
        
    def is_in_decel_zone(self, angle):
        """Check if angle is within deceleration zone"""
        # Handle wrap-around
        if self.decel_zone_start <= angle <= self.decel_zone_end:
            return True
        return False
    
    def get_distance_to_front(self, vehicle_idx):
        """Calculate distance to the vehicle in front"""
        current_angle = self.vehicles[vehicle_idx].angle
        front_idx = (vehicle_idx + 1) % self.num_vehicles
        front_angle = self.vehicles[front_idx].angle
        
        # Calculate angular distance (always positive in direction of travel)
        distance = front_angle - current_angle
        if distance <= 0:
            distance += 2 * np.pi
            
        return distance, front_idx
    
    def update(self, dt=0.1):
        """Update simulation state"""
        # Calculate target speeds for all vehicles
        target_speeds = []
        
        for i, vehicle in enumerate(self.vehicles):
            # Default target is max speed
            target_speed = self.max_speed
            
            # Check if in deceleration zone
            if self.is_in_decel_zone(vehicle.angle):
                target_speed = self.max_speed * self.decel_zone_speed_factor
            
            # Check distance to front vehicle
            distance, front_idx = self.get_distance_to_front(i)
            front_speed = self.vehicles[front_idx].speed
            
            # Adjust speed based on following distance
            if distance < self.brake_distance:
                # Too close - match or slow down to front vehicle speed
                target_speed = min(target_speed, front_speed * 0.9)
            elif distance < self.safe_distance:
                # Approaching - match front vehicle speed
                target_speed = min(target_speed, front_speed)
            
            target_speeds.append(target_speed)
        
        # Update vehicle speeds and positions
        for i, vehicle in enumerate(self.vehicles):
            target_speed = target_speeds[i]
            
            # Accelerate or decelerate towards target speed
            if vehicle.speed < target_speed:
                vehicle.speed = min(vehicle.speed + self.accel_rate, target_speed)
            elif vehicle.speed > target_speed:
                vehicle.speed = max(vehicle.speed - self.decel_rate, target_speed)
            
            # Ensure speed doesn't exceed max
            vehicle.speed = max(0, min(vehicle.speed, self.max_speed))
            
            # Update position
            vehicle.update_position(dt)
        
        self.time += dt
    
    def get_visualization_data(self):
        """Get data for visualization"""
        angles = [v.angle for v in self.vehicles]
        speeds = [v.speed for v in self.vehicles]
        return angles, speeds


def create_animation(simulator, duration=60, fps=30):
    """Create animated visualization of the traffic simulator"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Circular road
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Traffic Flow on Circular Road', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Draw road
    road = Circle((0, 0), 1.0, fill=False, edgecolor='gray', linewidth=8, alpha=0.3)
    ax1.add_patch(road)
    
    # Draw deceleration zone
    decel_zone = Wedge((0, 0), 1.0, 
                       np.degrees(simulator.decel_zone_start),
                       np.degrees(simulator.decel_zone_end),
                       width=0.2, facecolor='red', alpha=0.2, edgecolor='red')
    ax1.add_patch(decel_zone)
    
    # Add label for deceleration zone
    ax1.text(0, 1.3, 'Deceleration Zone\n(Uphill/Tunnel)', 
             ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    # Initialize vehicle scatter plot
    scatter = ax1.scatter([], [], c=[], s=100, cmap='RdYlGn', vmin=0, 
                         vmax=simulator.max_speed, edgecolors='black', linewidths=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Speed', rotation=270, labelpad=15)
    
    # Right plot: Speed vs Position
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(0, simulator.max_speed * 1.1)
    ax2.set_xlabel('Position (radians)', fontsize=12)
    ax2.set_ylabel('Speed (rad/s)', fontsize=12)
    ax2.set_title('Speed Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark deceleration zone on right plot
    ax2.axvspan(simulator.decel_zone_start, simulator.decel_zone_end, 
                alpha=0.2, color='red', label='Decel Zone')
    ax2.legend()
    
    speed_scatter = ax2.scatter([], [], c=[], s=50, cmap='RdYlGn', 
                               vmin=0, vmax=simulator.max_speed,
                               edgecolors='black', linewidths=0.5)
    
    # Time text
    time_text = ax1.text(-1.4, -1.4, '', fontsize=10)
    
    # Stats text
    stats_text = ax2.text(0.02, 0.98, '', transform=ax2.transAxes,
                         verticalalignment='top', fontsize=9,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        """Initialize animation"""
        scatter.set_offsets(np.empty((0, 2)))
        speed_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        stats_text.set_text('')
        return scatter, speed_scatter, time_text, stats_text
    
    def animate(frame):
        """Animation update function"""
        # Update simulation
        simulator.update(dt=0.1)
        
        # Get visualization data
        angles, speeds = simulator.get_visualization_data()
        
        # Convert to cartesian coordinates for circular plot
        x = np.cos(angles)
        y = np.sin(angles)
        positions = np.column_stack([x, y])
        
        # Update circular road plot
        scatter.set_offsets(positions)
        scatter.set_array(np.array(speeds))
        
        # Update speed vs position plot
        speed_positions = np.column_stack([angles, speeds])
        speed_scatter.set_offsets(speed_positions)
        speed_scatter.set_array(np.array(speeds))
        
        # Update time
        time_text.set_text(f'Time: {simulator.time:.1f}s')
        
        # Calculate statistics
        avg_speed = np.mean(speeds)
        min_speed = np.min(speeds)
        speed_variance = np.var(speeds)
        
        stats_text.set_text(f'Avg Speed: {avg_speed:.4f}\n'
                           f'Min Speed: {min_speed:.4f}\n'
                           f'Variance: {speed_variance:.6f}')
        
        return scatter, speed_scatter, time_text, stats_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=duration*fps, interval=1000/fps, blit=True)
    
    plt.tight_layout()
    return fig, anim


def main():
    """Main function to run the traffic simulator"""
    
    # Create simulator with intuitive real-world parameters
    # Reference: 120 km/h = one lap per minute (circular road: 2000m circumference)
    simulator = TrafficSimulator(
        num_vehicles=60,              # Number of vehicles
        max_speed_kmh=100,            # Maximum speed: 100 km/h
        decel_zone_center_deg=90,     # Deceleration zone center: 90° (top of circle)
        decel_zone_width_deg=40,      # Deceleration zone width: 40°
        decel_zone_speed_kmh=50,      # Speed in decel zone: 65 km/h (uphill/tunnel)
        safe_distance_m=30,           # Safe following distance: 50m (≈2 second rule)
        brake_distance_m=20,          # Brake initiation distance: 35m
        accel_rate_mps2=1.5,          # Acceleration: 2.0 m/s² (normal acceleration)
        decel_rate_mps2=3.5,          # Deceleration: 4.9 m/s² (normal braking)
    )
    
    print("Traffic Jam Simulator")
    print("=" * 70)
    print(f"Reference: 120 km/h = one lap per minute (circular road: 2000m)")
    print("=" * 70)
    print(f"Number of vehicles:        {simulator.num_vehicles}")
    print(f"Max speed:                 {simulator.max_speed_kmh} km/h")
    print(f"Deceleration zone:         {np.degrees(simulator.decel_zone_start):.1f}° - {np.degrees(simulator.decel_zone_end):.1f}°")
    print(f"Speed in decel zone:       {simulator.decel_zone_speed_kmh} km/h ({simulator.decel_zone_speed_factor * 100:.0f}% of max)")
    print(f"Safe following distance:   {simulator.safe_distance_m}m")
    print(f"Brake initiation distance: {simulator.brake_distance_m}m")
    print(f"Acceleration rate:         {simulator.accel_rate_mps2} m/s²")
    print(f"Deceleration rate:         {simulator.decel_rate_mps2} m/s²")
    print("=" * 70)
    print("\nStarting animation... (Close window to exit)")
    
    # Create and show animation
    fig, anim = create_animation(simulator, duration=120, fps=30)
    plt.show()


if __name__ == "__main__":
    main()

