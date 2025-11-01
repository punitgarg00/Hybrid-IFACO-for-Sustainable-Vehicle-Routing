# IFACO VRPTW with Multi-Objective Optimization (Distance + CO₂ Emissions)

## Overview
This corrected version implements **true multi-objective optimization** that simultaneously minimizes both **total distance** and **CO₂ emissions** during the search process.

## Key Changes Made

### 1. Multi-Objective Fitness Function
**Before:** Only distance was minimized
```python
fitness = total_distance
```

**After:** Both distance and emissions are minimized
```python
fitness = lambda1 * total_distance + lambda2 * total_emission
```

Where:
- `lambda1 = 1.0` (weight for distance)
- `lambda2 = 10.0` (weight for emissions)

### 2. Updated Files

#### `ifaco_core_co2.py`
- **ArtificialFish.calculate_fitness()**: Now calculates emissions during fitness evaluation and combines both objectives
- Fitness is computed as weighted sum of distance and CO₂ emissions

#### `aco.py`
- **Ant class**: Added `total_emission` and `fitness` attributes
- **construct_solution()**: Now calculates emissions after route construction
- **calculate_fitness()**: New method to compute multi-objective fitness
- **Ant comparison**: Changed from `total_distance` to `fitness` for selecting best ants
- **Neighborhood search**: Recalculates emissions and fitness after route improvements

#### `ifaco_main_co2.py`
- Updated reporting to show both distance and multi-objective fitness values
- Better visualization of the optimization results

#### `afsa.py`
- No changes needed (uses fitness from ArtificialFish class in core)

### 3. How It Works

1. **AFSA Phase**: Fish solutions are evaluated using the combined fitness (distance + weighted emissions)
2. **ACO Phase**: Ants are compared and selected based on combined fitness
3. **Best Solution**: The final solution minimizes both objectives simultaneously

### 4. Adjusting Weights

You can modify the weights in the fitness function to balance between distance and emissions:

**In `ifaco_core_co2.py` (ArtificialFish.calculate_fitness):**
```python
lambda1 = 1.0  # Weight for distance
lambda2 = 10.0  # Weight for emissions
```

**In `aco.py` (Ant.calculate_fitness):**
```python
lambda1 = 1.0  # Weight for distance
lambda2 = 10.0  # Weight for emissions
```

**Recommendations:**
- Equal priority: `lambda1 = 1.0, lambda2 = 1.0`
- Distance priority: `lambda1 = 1.0, lambda2 = 0.1`
- Emission priority: `lambda1 = 1.0, lambda2 = 20.0`

### 5. Running the Code

```bash
python ifaco_main_co2.py
```

The algorithm will now:
- Optimize for both distance and emissions during search
- Report both individual objectives and combined fitness
- Save detailed results including emission analysis

## Output

The algorithm reports:
- **Total Distance**: Sum of all route distances (km)
- **Total CO₂ Emissions**: Sum of all emissions (kg CO₂)
- **Multi-objective Fitness**: Combined objective value
- **Per-route emission breakdown**: Detailed analysis

## Research Implications

This implementation enables:
1. True multi-objective optimization (not just post-hoc analysis)
2. Pareto-optimal solutions when varying weights
3. Environmental impact-aware route planning
4. Trade-off analysis between cost (distance) and sustainability (emissions)

## Files Included
- `ifaco_core_co2.py` - Core data structures and emission calculator
- `aco.py` - Ant Colony Optimization with multi-objective fitness
- `afsa.py` - Artificial Fish Swarm Algorithm
- `ifaco_main_co2.py` - Main execution script
- `C101_sample.txt` - Solomon benchmark instance
- `README.md` - This file

## Citation
If you use this code for research, please cite the original IFACO paper and acknowledge the multi-objective extension for environmental optimization.
