"""
Improved Ant Colony Optimization - CORRECTED MULTI-OBJECTIVE VERSION

Consistent with AFSA multi-objective fitness calculation
"""

from ifaco_core_co2 import *
import math
import random
from typing import List, Optional

class Ant:
    """Individual ant for ACO algorithm with multi-objective fitness"""

    # Use same lambda values as ArtificialFish for consistency
    LAMBDA_DISTANCE = 1.0  # Weight for distance objective
    LAMBDA_EMISSION = 1.0  # Weight for emission objective

    def __init__(self, instance: VRPTWInstance, ant_id: int):
        self.instance = instance
        self.ant_id = ant_id
        self.routes = []
        self.total_distance = 0.0
        self.total_load = 0
        self.total_emission = 0.0  # CO₂ emissions
        self.fitness = float('inf')  # Multi-objective fitness

    def construct_solution(self, pheromone: np.ndarray, alpha: float, beta: float,
                          lamda: float, crowding_threshold: float):
        """Construct solution using transition probability"""
        n_customers = len(self.instance.customers) - 1
        if n_customers <= 0:
            return

        unvisited = list(range(1, n_customers + 1))
        current_route = []
        current_load = 0
        current_time = 0.0
        current_pos = 0  # Start from depot

        self.routes = []
        self.total_distance = 0.0
        self.total_load = 0

        max_attempts = n_customers * 3
        attempts = 0

        while unvisited and attempts < max_attempts:
            attempts += 1

            allowed = self._get_allowed_customers(current_pos, current_load,
                                                  current_time, unvisited)

            if not allowed:
                # End current route, start new one
                if current_route:
                    self.routes.append(current_route.copy())
                    self.total_distance += float(self.instance.distance_matrix[current_pos][0])

                current_route = []
                current_load = 0
                current_time = 0.0
                current_pos = 0
                continue

            # Apply crowding degree filter
            filtered_allowed = self._filter_by_crowding_degree(
                current_pos, allowed, pheromone, crowding_threshold)

            if not filtered_allowed:
                filtered_allowed = allowed

            # Select next customer
            next_customer = self._select_next_customer(
                current_pos, filtered_allowed, pheromone, alpha, beta, lamda,
                current_load, current_time)

            if next_customer is None:
                break

            # Move to selected customer
            customer = self.instance.customers[next_customer]
            current_route.append(next_customer)
            current_load += customer.demand
            self.total_load += customer.demand

            # Update time and distance
            travel_distance = float(self.instance.distance_matrix[current_pos][next_customer])
            self.total_distance += travel_distance
            current_time = max(current_time + travel_distance, customer.ready_time) + customer.service_time
            current_pos = next_customer
            unvisited.remove(next_customer)

        # Add final route
        if current_route:
            self.routes.append(current_route)
            self.total_distance += float(self.instance.distance_matrix[current_pos][0])

        # Calculate total emissions for the solution
        try:
            emissions_data = EmissionCalculator.calculate_solution_emissions(
                self.routes, self.instance
            )
            self.total_emission = emissions_data['total_emission_kg_co2']
        except:
            self.total_emission = 0.0

        # Calculate multi-objective fitness
        self.calculate_fitness()

    def calculate_fitness(self):
        """
        Calculate multi-objective fitness: weighted sum of distance and emissions

        Fitness = λ1 × Distance + λ2 × Emissions

        Uses same lambda values as ArtificialFish for consistency
        """
        self.fitness = (self.LAMBDA_DISTANCE * self.total_distance + 
                       self.LAMBDA_EMISSION * self.total_emission)

    def _select_next_customer(self, current_pos: int, allowed: list,
                            pheromone: np.ndarray, alpha: float, beta: float,
                            lamda: float, current_load: int, current_time: float):
        """
        Select next customer using transition probability

        Considers pheromone, distance, load, and time window factors
        """
        if not allowed:
            return None

        if len(allowed) == 1:
            return allowed[0]

        try:
            numerators = []

            for j in allowed:
                # Pheromone factor: τ_ij(t)^α
                tau_ij = max(float(pheromone[current_pos][j]), 1e-10)
                pheromone_factor = tau_ij ** alpha

                # Distance heuristic: η_ij^β where η_ij = 1/d_ij
                distance = max(float(self.instance.distance_matrix[current_pos][j]), 1e-10)
                eta_ij = 1.0 / distance
                distance_factor = eta_ij ** beta

                # Load factor
                load_factor = (current_load ** lamda) if current_load > 0 else 1.0

                # Time window urgency factor
                customer = self.instance.customers[j]
                arrival_time = current_time + distance
                time_penalty = abs(arrival_time - customer.ready_time) + abs(arrival_time - customer.due_time)
                time_factor = 1.0 / max(time_penalty, 1e-6)

                # Complete numerator calculation
                numerator = pheromone_factor * distance_factor * load_factor * time_factor
                numerators.append(max(numerator, 1e-10))

            # Calculate probabilities
            denominator = sum(numerators)
            if denominator <= 0:
                return random.choice(allowed)

            probabilities = [num / denominator for num in numerators]

            # Roulette wheel selection
            r = random.random()
            cumulative = 0.0

            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    return allowed[i]

            return allowed[-1]

        except Exception as e:
            print(f"Error in customer selection: {e}")
            return random.choice(allowed) if allowed else None

    def _filter_by_crowding_degree(self, current_pos: int, allowed: list,
                                   pheromone: np.ndarray, threshold: float):
        """Filter customers by crowding degree"""
        filtered = []

        try:
            for customer_id in allowed:
                crowding_degree = self._calculate_crowding_degree(current_pos, customer_id, pheromone)
                if crowding_degree < threshold:
                    filtered.append(customer_id)
        except Exception:
            return allowed

        return filtered if filtered else allowed

    def _calculate_crowding_degree(self, i: int, j: int, pheromone: np.ndarray):
        """Calculate crowding degree: ϑ_ij = 2τ_ij / Σ(τ_ik)"""
        try:
            tau_ij = max(float(pheromone[i][j]), 1e-10)
            sum_tau = max(float(np.sum(pheromone[i, :])), 1e-10)
            return 2.0 * tau_ij / sum_tau
        except Exception:
            return 0.5

    def _get_allowed_customers(self, current_pos: int, current_load: int,
                              current_time: float, unvisited: list):
        """Get feasible customers considering ALL constraints"""
        allowed = []

        for customer_id in unvisited:
            try:
                customer = self.instance.customers[customer_id]

                # Check capacity constraint
                if current_load + customer.demand > self.instance.vehicle_capacity:
                    continue

                # Check time window constraint
                travel_time = float(self.instance.distance_matrix[current_pos][customer_id])
                arrival_time = current_time + travel_time

                if arrival_time > customer.due_time:
                    continue

                allowed.append(customer_id)

            except Exception:
                continue

        return allowed

class RouteOptimizer:
    """Neighborhood search operators - 2-opt, Insertion, Crossover"""

    @staticmethod
    def two_opt_improve(route: List[int], distance_matrix: np.ndarray) -> List[int]:
        """2-opt route improvement"""
        if len(route) < 4:
            return route

        best_route = route.copy()
        improved = True

        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue

                    new_route = route[:i] + route[i:j][::-1] + route[j:]
                    old_dist = RouteOptimizer._calculate_route_distance(route, distance_matrix)
                    new_dist = RouteOptimizer._calculate_route_distance(new_route, distance_matrix)

                    if new_dist < old_dist:
                        best_route = new_route
                        improved = True
                        route = new_route.copy()
                        break

                if improved:
                    break

        return best_route

    @staticmethod
    def insertion_operator(routes: List[List[int]], instance: VRPTWInstance) -> List[List[int]]:
        """Route insertion operator"""
        if len(routes) < 2:
            return routes

        new_routes = [route.copy() for route in routes if route]

        for _ in range(min(3, len(new_routes))):
            source_idx = random.randint(0, len(new_routes) - 1)
            if len(new_routes[source_idx]) <= 1:
                continue

            target_idx = random.randint(0, len(new_routes) - 1)
            if target_idx == source_idx:
                continue

            cust_idx = random.randint(0, len(new_routes[source_idx]) - 1)
            customer_id = new_routes[source_idx][cust_idx]
            insert_pos = random.randint(0, len(new_routes[target_idx]))

            temp_target = new_routes[target_idx].copy()
            temp_target.insert(insert_pos, customer_id)

            temp_source = new_routes[source_idx].copy()
            temp_source.pop(cust_idx)

            if (RouteOptimizer._is_route_feasible(temp_target, instance) and
                RouteOptimizer._is_route_feasible(temp_source, instance)):

                old_dist = (RouteOptimizer._calculate_route_distance(new_routes[source_idx], instance.distance_matrix) +
                           RouteOptimizer._calculate_route_distance(new_routes[target_idx], instance.distance_matrix))

                new_dist = (RouteOptimizer._calculate_route_distance(temp_source, instance.distance_matrix) +
                           RouteOptimizer._calculate_route_distance(temp_target, instance.distance_matrix))

                if new_dist < old_dist:
                    new_routes[source_idx] = temp_source
                    new_routes[target_idx] = temp_target

        return [route for route in new_routes if route]

    @staticmethod
    def route_crossover_operator(routes: List[List[int]], instance: VRPTWInstance) -> List[List[int]]:
        """Route crossover operator"""
        if len(routes) < 2:
            return routes

        new_routes = [route.copy() for route in routes if route]

        if len(new_routes) >= 2:
            idx1, idx2 = random.sample(range(len(new_routes)), 2)
            route1, route2 = new_routes[idx1], new_routes[idx2]

            if len(route1) > 2 and len(route2) > 2:
                cross_point1 = random.randint(1, len(route1) - 2)
                cross_point2 = random.randint(1, len(route2) - 2)

                new_route1 = route1[:cross_point1] + route2[cross_point2:]
                new_route2 = route2[:cross_point2] + route1[cross_point1:]

                new_route1 = RouteOptimizer._remove_duplicates(new_route1)
                new_route2 = RouteOptimizer._remove_duplicates(new_route2)

                if (RouteOptimizer._is_route_feasible(new_route1, instance) and
                    RouteOptimizer._is_route_feasible(new_route2, instance)):

                    old_dist = (RouteOptimizer._calculate_route_distance(route1, instance.distance_matrix) +
                               RouteOptimizer._calculate_route_distance(route2, instance.distance_matrix))

                    new_dist = (RouteOptimizer._calculate_route_distance(new_route1, instance.distance_matrix) +
                               RouteOptimizer._calculate_route_distance(new_route2, instance.distance_matrix))

                    if new_dist < old_dist:
                        new_routes[idx1] = new_route1
                        new_routes[idx2] = new_route2

        return [route for route in new_routes if route]

    @staticmethod
    def _remove_duplicates(route: List[int]) -> List[int]:
        seen = set()
        result = []
        for customer in route:
            if customer not in seen:
                seen.add(customer)
                result.append(customer)
        return result

    @staticmethod
    def _is_route_feasible(route: List[int], instance: VRPTWInstance) -> bool:
        if not route:
            return True

        total_demand = sum(instance.customers[cust].demand for cust in route)
        if total_demand > instance.vehicle_capacity:
            return False

        current_time = 0.0
        current_pos = 0

        for cust_id in route:
            customer = instance.customers[cust_id]
            travel_time = instance.distance_matrix[current_pos][cust_id]
            arrival_time = current_time + travel_time

            if arrival_time > customer.due_time:
                return False

            current_time = max(arrival_time, customer.ready_time) + customer.service_time
            current_pos = cust_id

        return True

    @staticmethod
    def _calculate_route_distance(route: List[int], distance_matrix: np.ndarray) -> float:
        if not route:
            return 0.0

        total = 0.0
        total += float(distance_matrix[0][route[0]])

        for i in range(len(route) - 1):
            total += float(distance_matrix[route[i]][route[i+1]])

        total += float(distance_matrix[route[-1]][0])

        return total

class ImprovedAntColonyOptimization:
    """ACO implementation with multi-objective fitness"""

    def __init__(self, instance: VRPTWInstance, num_ants: int = 25, max_iterations: int = 120,
                 alpha: float = 1.0, beta: float = 3.0, lamda: float = 1.5,
                 rho: float = 0.25, q: float = 100.0, c: float = 0.1):
        self.instance = instance
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.lamda = lamda
        self.rho = rho
        self.q = q
        self.c = c

        n = len(instance.customers)
        self.pheromone = np.ones((n, n), dtype=np.float64) * 0.1

        self.best_solution = None
        self.best_fitness = float('inf')  # Now using multi-objective fitness
        self.iteration_best_fitnesses = []

    def run(self) -> Optional[Ant]:
        """Run ACO algorithm with multi-objective optimization"""
        print(f"Running MULTI-OBJECTIVE ACO with {self.num_ants} ants for {self.max_iterations} iterations...")
        print(f"Parameters: α={self.alpha}, β={self.beta}, λ={self.lamda}, ρ={self.rho}")
        print(f"Fitness = {Ant.LAMBDA_DISTANCE}×Distance + {Ant.LAMBDA_EMISSION}×Emissions")

        for iteration in range(self.max_iterations):
            crowding_threshold = 1.0 - math.exp(-self.c * iteration)

            ants = []
            for ant_id in range(self.num_ants):
                ant = Ant(self.instance, ant_id)
                ant.construct_solution(self.pheromone, self.alpha, self.beta,
                                     self.lamda, crowding_threshold)
                ants.append(ant)

            valid_ants = [ant for ant in ants if ant.routes and ant.total_distance > 0]

            if valid_ants:
                # Use multi-objective fitness for comparison
                iteration_best = min(valid_ants, key=lambda ant: ant.fitness)

                # Apply neighborhood search
                improved_ant = self._apply_complete_neighborhood_search(iteration_best)

                if improved_ant.fitness < iteration_best.fitness:
                    iteration_best = improved_ant

                if iteration_best.fitness < self.best_fitness:
                    self.best_fitness = iteration_best.fitness
                    self.best_solution = iteration_best
                    print(f"Iteration {iteration}: New best fitness = {self.best_fitness:.2f}")
                    print(f"  └─ Distance: {iteration_best.total_distance:.2f} km, "
                          f"Emissions: {iteration_best.total_emission:.2f} kg CO₂")

                self.iteration_best_fitnesses.append(iteration_best.fitness)

                self._update_pheromone_elitist(iteration_best)
            else:
                self.iteration_best_fitnesses.append(float('inf'))

            if iteration % 20 == 0 and self.best_solution:
                print(f"Iteration {iteration}: Current best fitness = {self.best_fitness:.2f}")

        if self.best_solution:
            print(f"\nACO completed. Final best fitness: {self.best_fitness:.2f}")
            print(f"  Distance: {self.best_solution.total_distance:.2f} km")
            print(f"  Emissions: {self.best_solution.total_emission:.2f} kg CO₂")
            print(f"  Number of routes: {len([r for r in self.best_solution.routes if r])}")

        return self.best_solution

    def _apply_complete_neighborhood_search(self, ant: Ant) -> Ant:
        """Apply all three neighborhood operators"""
        improved_ant = ant

        try:
            routes_2opt = []
            for route in ant.routes:
                if route:
                    improved_route = RouteOptimizer.two_opt_improve(route, self.instance.distance_matrix)
                    routes_2opt.append(improved_route)

            routes_insertion = RouteOptimizer.insertion_operator(routes_2opt, self.instance)
            routes_crossover = RouteOptimizer.route_crossover_operator(routes_insertion, self.instance)

            if routes_crossover:
                new_ant = Ant(self.instance, ant.ant_id)
                new_ant.routes = routes_crossover
                new_ant.total_distance = self._calculate_total_distance(routes_crossover)

                # Calculate emissions for improved solution
                try:
                    emissions_data = EmissionCalculator.calculate_solution_emissions(
                        routes_crossover, self.instance
                    )
                    new_ant.total_emission = emissions_data['total_emission_kg_co2']
                except:
                    new_ant.total_emission = 0.0

                new_ant.calculate_fitness()

                if new_ant.fitness < improved_ant.fitness:
                    improved_ant = new_ant

        except Exception as e:
            print(f"Error in neighborhood search: {e}")

        return improved_ant

    def _update_pheromone_elitist(self, best_ant):
        """Elitist pheromone update - only best ant deposits"""
        # Evaporate all pheromones: τ_ij(t+1) = (1-ρ)τ_ij(t)
        self.pheromone *= (1.0 - self.rho)

        # Only best ant deposits: Δτ_ij = Q/F_k (using multi-objective fitness)
        delta = self.q / max(best_ant.fitness, 1e-10)

        for route in best_ant.routes:
            prev = 0  # depot
            for customer in route:
                self.pheromone[prev][customer] += delta
                prev = customer
            self.pheromone[prev][0] += delta  # return to depot

    def _calculate_total_distance(self, routes: List[List[int]]) -> float:
        total = 0.0
        for route in routes:
            if not route:
                continue

            total += float(self.instance.distance_matrix[0][route[0]])

            for i in range(len(route) - 1):
                total += float(self.instance.distance_matrix[route[i]][route[i+1]])

            total += float(self.instance.distance_matrix[route[-1]][0])

        return total

    def initialize_pheromone_from_afsa(self, afsa_best_solution):
        """Initialize pheromone matrix from AFSA solution"""
        if not afsa_best_solution or not afsa_best_solution.position:
            return

        try:
            self.pheromone.fill(0.1)

            # Use multi-objective fitness value from AFSA
            initial_deposit = self.q / max(afsa_best_solution.fitness, 1e-10)

            for route in afsa_best_solution.position:
                if not route:
                    continue

                self.pheromone[0][route[0]] += initial_deposit

                for i in range(len(route) - 1):
                    self.pheromone[route[i]][route[i+1]] += initial_deposit

                self.pheromone[route[-1]][0] += initial_deposit

            print(f"Initialized ACO pheromone from AFSA (multi-objective fitness: {afsa_best_solution.fitness:.2f})")
            print(f"  └─ AFSA Distance: {afsa_best_solution.total_distance:.2f} km, "
                  f"Emissions: {afsa_best_solution.total_emission:.2f} kg CO₂")

        except Exception as e:
            print(f"Error initializing pheromone from AFSA: {e}")
