"""
Artificial Fish Swarm Algorithm - CORRECTED MULTI-OBJECTIVE VERSION

All behaviors properly implemented with multi-objective fitness
"""

from ifaco_core_co2 import *

class ArtificialFishSwarmAlgorithm:
    """Artificial Fish Swarm Algorithm for VRPTW with Multi-Objective Optimization"""

    def __init__(self, instance: VRPTWInstance, fish_count: int = 20, max_iterations: int = 75):
        self.instance = instance
        self.fish_count = fish_count
        self.max_iterations = max_iterations

        # AFSA parameters
        self.visual_range = 0.5  # Visual range for detecting nearby fish
        self.step_size = 0.3     # Step size for movement
        self.delta = 0.1         # Crowding factor
        self.try_number = 30     # Number of attempts in prey behavior

        # Initialize fish population
        self.fishes = []
        for i in range(fish_count):
            try:
                fish = ArtificialFish(instance, i)
                self.fishes.append(fish)
            except Exception as e:
                print(f"Warning: Could not create fish {i}: {e}")
                continue

        if not self.fishes:
            self.fishes = [ArtificialFish(instance, 0)]

        self.best_fish = min(self.fishes, key=lambda f: f.fitness)
        self.bulletin_board = self.best_fish
        self.fitness_history = []

    def prey_behavior(self, fish: ArtificialFish) -> ArtificialFish:
        """Prey behavior - search for better food (better fitness)"""
        best_fish = fish

        for _ in range(self.try_number):
            try:
                neighbor = self.generate_neighbor(fish)
                if neighbor and neighbor.fitness < best_fish.fitness:
                    best_fish = neighbor
            except Exception:
                continue

        return best_fish

    def swarm_behavior(self, fish: ArtificialFish) -> Optional[ArtificialFish]:
        """Swarm behavior - move towards center of nearby fish"""
        try:
            nearby_fish = self.get_nearby_fish(fish)

            if len(nearby_fish) < 2:
                return None

            center_fish = self.calculate_center(nearby_fish)

            if center_fish and center_fish.fitness < fish.fitness:
                return center_fish

            return None

        except Exception:
            return None

    def follow_behavior(self, fish: ArtificialFish) -> Optional[ArtificialFish]:
        """Follow behavior - follow the best nearby fish"""
        try:
            nearby_fish = self.get_nearby_fish(fish)

            if not nearby_fish:
                return None

            best_nearby = min(nearby_fish, key=lambda f: f.fitness)

            if best_nearby.fitness < fish.fitness:
                new_fish = self.move_towards(fish, best_nearby)
                if new_fish and new_fish.fitness < fish.fitness:
                    return new_fish

            return None

        except Exception:
            return None

    def get_nearby_fish(self, fish: ArtificialFish) -> List[ArtificialFish]:
        """
        Get fish within visual range - CORRECTED AND UNCOMMENTED

        Uses solution dissimilarity as distance metric
        """
        nearby = []

        try:
            for other_fish in self.fishes:
                if other_fish.fish_id != fish.fish_id:
                    # Calculate distance between fish positions
                    distance = calculate_distance(fish.position, other_fish.position, self.instance)

                    # Normalize by number of customers for fair comparison
                    n_customers = len(self.instance.customers) - 1
                    normalized_distance = distance / max(n_customers, 1)

                    if normalized_distance < self.visual_range:
                        nearby.append(other_fish)
        except Exception as e:
            print(f"Warning in get_nearby_fish: {e}")
            pass

        return nearby

    def generate_neighbor(self, fish: ArtificialFish) -> Optional[ArtificialFish]:
        """Generate neighbor solution using route modification operators"""
        try:
            if not fish.position:
                return None

            # Deep copy the position
            new_position = [route[:] for route in fish.position]

            # Random operator selection (1: swap, 2: relocate, 3: 2-opt)
            modification_type = random.randint(1, 3)

            if modification_type == 1 and new_position:
                # Swap two customers within a route
                route_idx = random.randint(0, len(new_position) - 1)
                route = new_position[route_idx]

                if len(route) >= 2:
                    i, j = random.sample(range(len(route)), 2)
                    route[i], route[j] = route[j], route[i]

            elif modification_type == 2 and len(new_position) >= 2:
                # Relocate customer from one route to another
                from_route_idx = random.randint(0, len(new_position) - 1)
                to_route_idx = random.randint(0, len(new_position) - 1)

                if from_route_idx != to_route_idx and new_position[from_route_idx]:
                    customer = random.choice(new_position[from_route_idx])
                    customer_demand = self.instance.customers[customer].demand

                    # Check capacity constraint
                    to_route_load = sum(self.instance.customers[c].demand 
                                      for c in new_position[to_route_idx])

                    if to_route_load + customer_demand <= self.instance.vehicle_capacity:
                        new_position[from_route_idx].remove(customer)
                        new_position[to_route_idx].append(customer)

            else:
                # 2-opt within a route
                if new_position:
                    route_idx = random.randint(0, len(new_position) - 1)
                    route = new_position[route_idx]

                    if len(route) >= 2:
                        i = random.randint(0, len(route) - 2)
                        j = random.randint(i + 1, len(route) - 1)
                        route[i:j+1] = route[i:j+1][::-1]

            # Remove empty routes
            new_position = [route for route in new_position if route]

            # Create new fish with modified position
            new_fish = ArtificialFish(self.instance, fish.fish_id)
            new_fish.position = new_position
            new_fish.fitness = new_fish.calculate_fitness()

            return new_fish

        except Exception as e:
            print(f"Error in generate_neighbor: {e}")
            return None

    def calculate_center(self, fish_list: List[ArtificialFish]) -> Optional[ArtificialFish]:
        """
        Calculate center position of fish group

        For VRPTW, we use the best fish in the group as approximation of center
        """
        if not fish_list:
            return None

        try:
            # Use best fish as center representative
            center_fish = min(fish_list, key=lambda f: f.fitness)

            # Generate a neighbor of the best fish to move towards
            return self.generate_neighbor(center_fish)

        except Exception:
            return None

    def move_towards(self, fish: ArtificialFish, target: ArtificialFish) -> Optional[ArtificialFish]:
        """
        Move fish towards target

        In discrete space, we generate a neighbor solution that's closer to target
        """
        try:
            # Generate multiple neighbors and choose one closer to target
            best_neighbor = None
            best_distance_to_target = float('inf')

            for _ in range(5):  # Try 5 neighbors
                neighbor = self.generate_neighbor(fish)

                if neighbor:
                    # Calculate distance from neighbor to target
                    dist = calculate_distance(neighbor.position, target.position, self.instance)

                    if dist < best_distance_to_target and neighbor.fitness < fish.fitness:
                        best_distance_to_target = dist
                        best_neighbor = neighbor

            return best_neighbor

        except Exception:
            return None

    def run(self) -> ArtificialFish:
        """Run AFSA algorithm with multi-objective optimization"""
        print(f"Running MULTI-OBJECTIVE AFSA with {self.fish_count} fish for {self.max_iterations} iterations...")
        print(f"Fitness = {ArtificialFish.LAMBDA_DISTANCE}×Distance + {ArtificialFish.LAMBDA_EMISSION}×Emissions")

        for iteration in range(self.max_iterations):
            new_fishes = []

            for fish in self.fishes:
                try:
                    new_fish = fish

                    # Try prey behavior
                    prey_result = self.prey_behavior(fish)
                    if prey_result.fitness < new_fish.fitness:
                        new_fish = prey_result

                    # Try swarm behavior
                    swarm_result = self.swarm_behavior(fish)
                    if swarm_result and swarm_result.fitness < new_fish.fitness:
                        new_fish = swarm_result

                    # Try follow behavior
                    follow_result = self.follow_behavior(fish)
                    if follow_result and follow_result.fitness < new_fish.fitness:
                        new_fish = follow_result

                    new_fishes.append(new_fish)

                except Exception as e:
                    print(f"Error processing fish {fish.fish_id}: {e}")
                    new_fishes.append(fish)

            # Update fish population
            self.fishes = new_fishes

            # Update best fish
            current_best = min(self.fishes, key=lambda f: f.fitness)

            if current_best.fitness < self.best_fish.fitness:
                self.best_fish = current_best
                self.bulletin_board = current_best

            self.fitness_history.append(self.best_fish.fitness)

            # Progress reporting
            if iteration % 15 == 0:
                print(f"AFSA Iteration {iteration}: Best fitness = {self.best_fish.fitness:.2f}")
                print(f"  └─ Distance: {self.best_fish.total_distance:.2f} km, "
                      f"Emissions: {self.best_fish.total_emission:.2f} kg CO₂")

        print(f"\nAFSA completed. Final best fitness: {self.best_fish.fitness:.2f}")
        print(f"  Distance: {self.best_fish.total_distance:.2f} km")
        print(f"  Emissions: {self.best_fish.total_emission:.2f} kg CO₂")

        return self.best_fish
