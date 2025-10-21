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
                 min_distance_m=2.0,
                 desired_time_headway=1.5,
                 accel_rate_mps2=2.0,
                 comfortable_decel_mps2=3.0):
        """
        Initialize traffic simulator with IDM (Intelligent Driver Model)
        
        Args:
            num_vehicles: Number of vehicles on the road
            max_speed_kmh: Maximum desired speed (km/h)
            decel_zone_center_deg: Center angle of deceleration zone (degrees)
            decel_zone_width_deg: Width of deceleration zone (degrees)
            decel_zone_speed_kmh: Desired speed in deceleration zone (km/h)
            min_distance_m: Minimum distance to front vehicle (meters)
            desired_time_headway: Desired time headway (seconds)
            accel_rate_mps2: Maximum acceleration (m/s²)
            comfortable_decel_mps2: Comfortable deceleration (m/s²)
        """
        self.num_vehicles = num_vehicles
        
        # Convert speeds from km/h to m/s and rad/s
        self.max_speed_ms = max_speed_kmh / 3.6  # m/s
        self.max_speed = self._kmh_to_rads(max_speed_kmh)  # rad/s
        self.decel_zone_speed_ms = decel_zone_speed_kmh / 3.6  # m/s
        self.decel_zone_speed = self._kmh_to_rads(decel_zone_speed_kmh)  # rad/s
        
        # Convert deceleration zone from degrees to radians
        decel_zone_center_rad = np.radians(decel_zone_center_deg)
        decel_zone_half_width_rad = np.radians(decel_zone_width_deg / 2)
        self.decel_zone_start = decel_zone_center_rad - decel_zone_half_width_rad
        self.decel_zone_end = decel_zone_center_rad + decel_zone_half_width_rad
        
        # IDM parameters in m/s units
        self.min_distance_m = min_distance_m  # s0 in IDM
        self.desired_time_headway = desired_time_headway  # T in IDM
        self.max_accel_mps2 = accel_rate_mps2  # a_max in IDM
        self.comfortable_decel_mps2 = comfortable_decel_mps2  # b in IDM
        
        # Store original values for display
        self.max_speed_kmh = max_speed_kmh
        self.decel_zone_speed_kmh = decel_zone_speed_kmh
        self.min_distance_m_display = min_distance_m
        self.desired_time_headway_display = desired_time_headway
        self.accel_rate_mps2 = accel_rate_mps2
        self.comfortable_decel_mps2_display = comfortable_decel_mps2
        
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
        """Update simulation state using IDM (Intelligent Driver Model)"""
        # Calculate accelerations for all vehicles using IDM
        accelerations = []
        
        for i, vehicle in enumerate(self.vehicles):
            # Convert current speed from rad/s to m/s for IDM calculation
            v_ms = vehicle.speed * self.RADIUS_M  # Current speed in m/s
            
            # Determine desired speed based on location
            if self.is_in_decel_zone(vehicle.angle):
                v0_ms = self.decel_zone_speed_ms  # Desired speed in decel zone
            else:
                v0_ms = self.max_speed_ms  # Normal desired speed
            
            # Get distance and speed of front vehicle
            distance_rad, front_idx = self.get_distance_to_front(i)
            distance_m = distance_rad * self.RADIUS_M  # Convert to meters
            v_front_ms = self.vehicles[front_idx].speed * self.RADIUS_M  # Front vehicle speed in m/s
            
            # Calculate relative speed (positive = approaching)
            delta_v = v_ms - v_front_ms
            
            # IDM: Calculate desired gap (s*)
            # s* = s0 + v*T + (v*Δv)/(2*sqrt(a*b))
            interaction_term = (v_ms * delta_v) / (2 * np.sqrt(self.max_accel_mps2 * self.comfortable_decel_mps2))
            s_star = self.min_distance_m + v_ms * self.desired_time_headway + interaction_term
            
            # Ensure s_star is non-negative
            s_star = max(s_star, self.min_distance_m)
            
            # IDM: Calculate acceleration
            # a = a_max * [1 - (v/v0)^4 - (s*/s)^2]
            
            # Free road acceleration term
            if v0_ms > 0:
                free_road_term = 1.0 - (v_ms / v0_ms) ** 4
            else:
                free_road_term = 0
            
            # Interaction term
            if distance_m > 0:
                interaction_term = (s_star / distance_m) ** 2
            else:
                interaction_term = float('inf')  # Emergency brake
            
            # Total acceleration
            accel_mps2 = self.max_accel_mps2 * (free_road_term - interaction_term)
            
            # Convert acceleration from m/s² to rad/s²
            accel_rads2 = accel_mps2 / self.RADIUS_M
            
            accelerations.append(accel_rads2)
        
        # Update vehicle speeds and positions
        for i, vehicle in enumerate(self.vehicles):
            accel = accelerations[i]
            
            # Update speed: v_new = v_old + a * dt
            vehicle.speed += accel * dt
            
            # Ensure speed is non-negative and doesn't exceed max
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
    # Convert max speed to km/h for display
    max_speed_kmh = simulator.max_speed * simulator.RADIUS_M * 3.6
    scatter = ax1.scatter([], [], c=[], s=100, cmap='RdYlGn', vmin=0, 
                         vmax=max_speed_kmh, edgecolors='black', linewidths=1)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Speed (km/h)', rotation=270, labelpad=15)
    
    # Right plot: Speed vs Position
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(0, max_speed_kmh * 1.1)
    ax2.set_xlabel('Position (radians)', fontsize=12)
    ax2.set_ylabel('Speed (km/h)', fontsize=12)
    ax2.set_title('Speed Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark deceleration zone on right plot
    ax2.axvspan(simulator.decel_zone_start, simulator.decel_zone_end, 
                alpha=0.2, color='red', label='Decel Zone')
    ax2.legend()
    
    speed_scatter = ax2.scatter([], [], c=[], s=50, cmap='RdYlGn', 
                               vmin=0, vmax=max_speed_kmh,
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
        
        # Convert speeds from rad/s to km/h for display
        speeds_kmh = np.array(speeds) * simulator.RADIUS_M * 3.6
        
        # Convert to cartesian coordinates for circular plot
        x = np.cos(angles)
        y = np.sin(angles)
        positions = np.column_stack([x, y])
        
        # Update circular road plot
        scatter.set_offsets(positions)
        scatter.set_array(speeds_kmh)
        
        # Update speed vs position plot
        speed_positions = np.column_stack([angles, speeds_kmh])
        speed_scatter.set_offsets(speed_positions)
        speed_scatter.set_array(speeds_kmh)
        
        # Update time
        time_text.set_text(f'Time: {simulator.time:.1f}s')
        
        # Calculate statistics in km/h
        avg_speed_kmh = np.mean(speeds_kmh)
        min_speed_kmh = np.min(speeds_kmh)
        speed_variance_kmh = np.var(speeds_kmh)
        
        stats_text.set_text(f'Avg Speed: {avg_speed_kmh:.1f} km/h\n'
                           f'Min Speed: {min_speed_kmh:.1f} km/h\n'
                           f'Variance: {speed_variance_kmh:.2f}')
        
        return scatter, speed_scatter, time_text, stats_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=duration*fps, interval=1000/fps, blit=True)
    
    plt.tight_layout()
    return fig, anim


def main():
    """Main function to run the traffic simulator"""
    
    # Create simulator with IDM (Intelligent Driver Model)
    # Reference: 120 km/h = one lap per minute (circular road: 2000m circumference)
    # All vehicles are configured as "poor drivers" (aggressive, sudden braking)
    simulator = TrafficSimulator(
        num_vehicles=30,              # Number of vehicles
        max_speed_kmh=100,            # Maximum speed: 100 km/h
        decel_zone_center_deg=90,     # Deceleration zone center: 90° (top of circle)
        decel_zone_width_deg=40,      # Deceleration zone width: 40°
        decel_zone_speed_kmh=65,      # Speed in decel zone: 50 km/h (uphill/tunnel)
        min_distance_m=1.5,           # Minimum distance: 1.5m (close following)
        desired_time_headway=0.9,     # Time headway: 0.9s (short gap - aggressive)
        accel_rate_mps2=2.5,          # Max acceleration: 2.5 m/s² (rapid acceleration)
        comfortable_decel_mps2=6.0,   # Comfortable deceleration: 6.0 m/s² (sudden braking)
    )
    
    print("Traffic Jam Simulator with IDM (Intelligent Driver Model)")
    print("=" * 70)
    print(f"Reference: 120 km/h = one lap per minute (circular road: 2000m)")
    print(f"Driver Type: POOR DRIVERS (aggressive, sudden braking)")
    print("=" * 70)
    print(f"Number of vehicles:          {simulator.num_vehicles}")
    print(f"Max desired speed:           {simulator.max_speed_kmh} km/h")
    print(f"Deceleration zone:           {np.degrees(simulator.decel_zone_start):.1f}° - {np.degrees(simulator.decel_zone_end):.1f}°")
    print(f"Desired speed in decel zone: {simulator.decel_zone_speed_kmh} km/h")
    print(f"")
    print(f"IDM Parameters (Poor Driver Settings):")
    print(f"  Minimum distance (s0):     {simulator.min_distance_m}m (close following)")
    print(f"  Time headway (T):          {simulator.desired_time_headway}s (short gap)")
    print(f"  Max acceleration (a):      {simulator.accel_rate_mps2} m/s² (rapid accel)")
    print(f"  Comfortable decel (b):     {simulator.comfortable_decel_mps2_display} m/s² (sudden brake)")
    print("=" * 70)
    print("\nStarting animation... (Close window to exit)")
    
    # Create and show animation
    fig, anim = create_animation(simulator, duration=120, fps=30)
    plt.show()


if __name__ == "__main__":
    main()

