# Debug analysis to understand policy near boulders
# This will help us see what's really happening

# Check states adjacent to boulders (2,4) and (3,4)
print("=== Analyzing states near boulders ===\n")

# Boulder is at (2,4) and (3,4)
# Let's check state (2,3) which is south of boulder at (2,4)
state_below_boulder = [0, 2, 3]  # Phase 0, position (2,3)
print(f"State (0, 2, 3) - below boulder at (2,4):")
print(f"  Value: {gridWorld[0, 2, 3]:.2f}")

# Check what actions lead to from this state
for direction in ['N', 'S', 'E', 'W']:
    temp_val = 0
    p_states = possible_states(state_below_boulder, direction)
    print(f"\n  Action {direction}:")
    print(f"    Intended outcome (0.7): {p_states[0]} -> reward={reward(p_states[0])}, value={gridWorld[p_states[0][0], p_states[0][1], p_states[0][2]]:.2f}, terminal={terminal(p_states[0])}")
    for i in range(1, 4):
        print(f"    Side effect #{i} (0.1): {p_states[i]} -> reward={reward(p_states[i])}, value={gridWorld[p_states[i][0], p_states[i][1], p_states[i][2]]:.2f}, terminal={terminal(p_states[i])}")
    
    # Calculate expected value for this action
    temp_val += 0.7 * (reward(p_states[0]) + 0.95 * gridWorld[p_states[0][0], p_states[0][1], p_states[0][2]])
    for i in range(1, 4):
        p_state = p_states[i]
        temp_val += 0.1 * (reward(p_state) + 0.95 * gridWorld[p_state[0], p_state[1], p_state[2]])
    print(f"    Expected value: {temp_val:.2f}")

optimal = get_optimal_action(state_below_boulder, 0.95, gridWorld)
print(f"\n  Optimal action: {optimal}")
