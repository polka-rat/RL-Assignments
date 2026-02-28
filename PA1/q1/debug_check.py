# Debug script to check (4,4) action values
import numpy as np

# Simulate the values at (4,4) Phase 0
state = [0, 4, 4]
gamma = 0.95

# Values from the grid
V_44 = -7.9
V_43 = 0.6
V_34 = 0  # Boulder terminal
V_34_actual = 0  # Terminal state

print("Debug: Checking (4,4) in Phase 0 with gamma=0.95")
print(f"V(4,4) = {V_44}")
print(f"V(4,3) = {V_43}")
print()

# South action: intended=(4,3), perpendicular E=(4,4) boundary, W=(3,4) boulder, stay=(4,4)
south_val = (0.7 * (-1 + gamma * V_43) + 
             0.1 * (-1 + gamma * V_44) +  # East hits boundary, stays
             0.1 * (-100 + 0) +            # West hits boulder
             0.1 * (-1 + gamma * V_44))    # Stay
print(f"South action value: {south_val:.3f}")
print(f"  - 0.7 * (r + γV(4,3)) = 0.7 * ({-1} + {gamma * V_43:.3f}) = {0.7 * (-1 + gamma * V_43):.3f}")
print(f"  - 0.1 * (r + γV(4,4)) [E boundary] = 0.1 * ({-1 + gamma * V_44:.3f}) = {0.1 * (-1 + gamma * V_44):.3f}")
print(f"  - 0.1 * (r + 0) [W boulder] = 0.1 * (-100) = {0.1 * (-100):.3f}")
print(f"  - 0.1 * (r + γV(4,4)) [stay] = 0.1 * ({-1 + gamma * V_44:.3f}) = {0.1 * (-1 + gamma * V_44):.3f}")
print()

# Hover action
hover_val = -1 + gamma * V_44
print(f"Hover action value: {hover_val:.3f}")
print(f"  - r + γV(4,4) = {-1} + {gamma * V_44:.3f} = {hover_val:.3f}")
print()

# North action: intended=(4,4) boundary skip
print("North action: Skipped (hits boundary)")
print()

# East action: intended=(4,4) boundary skip  
print("East action: Skipped (hits boundary)")
print()

# West action: intended=(3,4) boulder
west_val = (0.7 * (-100 + 0) +           # West to boulder
            0.1 * (-1 + gamma * V_44) +  # North hits boundary, stays
            0.1 * (-1 + gamma * V_43) +  # South
            0.1 * (-1 + gamma * V_44))   # Stay
print(f"West action value: {west_val:.3f}")
print()

print(f"Best action: {'Hover' if hover_val > max(south_val, west_val) else 'South' if south_val > west_val else 'West'}")
print(f"Hover: {hover_val:.3f}, South: {south_val:.3f}, West: {west_val:.3f}")
