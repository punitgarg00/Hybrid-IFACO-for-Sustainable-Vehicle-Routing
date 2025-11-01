"""
IFACO - Improved Fish Swarm-Ant Colony Optimization for Vehicle Routing Problem with Time Windows
Extended version with CO₂ Emission Calculation - CORRECTED MULTI-OBJECTIVE VERSION

Novelty: CO₂ emission tracking based on load and distance
Formula: E_ij = γ × d_ij × L_ij

Multi-objective fitness: F = λ1 × Distance + λ2 × Emissions

where:
- γ (gamma): emission factor (kg CO₂ per km per unit load)
- d_ij: distance from customer i to customer j (km)
- L_ij: load carried on edge (i,j) (units)
- λ1, λ2: weight parameters for multi-objective optimization

Typical emission factors (from research):
- Light Duty Vehicle (<3.5T): ~0.307 kg CO₂/km base
- Medium Duty Vehicle (<12T): ~0.593 kg CO₂/km base
- Heavy Duty Vehicle (>12T): ~0.738 kg CO₂/km base
- Per tonne-km: 0.04-0.065 kg CO₂/tonne-km (road transport average)
"""

import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import json
import os

@dataclass
class Customer:
    """Customer data structure for VRPTW"""
    id: int
    x: float
    y: float
    demand: int
    ready_time: int
    due_time: int
    service_time: int

@dataclass
class VRPTWInstance:
    """VRPTW Instance data structure with CO₂ emission support"""
    name: str
    num_vehicles: int
    vehicle_capacity: int
    customers: List[Customer]
    distance_matrix: np.ndarray
    emission_factor: float = 0.05  

class SolomonDataLoader:
    """Load and parse Solomon VRPTW benchmark instances - CORRECTED VERSION"""

    @staticmethod
    def load_solomon_instance(filename: str, emission_factor: float = 0.05) -> VRPTWInstance:
        """
        Load Solomon instance from file - handles C101 format

        Args:
            filename: Path to Solomon instance file
            emission_factor: γ (kg CO₂ per km per unit load)
                           Default: 0.05 (moderate estimate for freight vehicles)
                           Typical range: 0.04-0.065 based on road transport literature
        """
        try:
            print(f"Loading instance: {filename}")
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            customers = []
            data_start = 0

            # Find where customer data starts
            for i, line in enumerate(lines):
                if any(header in line.upper() for header in ['CUSTOMER', 'CUST NO', 'XCOORD', 'YCOORD', 'DEMAND']):
                    continue

                parts = line.split()
                if len(parts) >= 7:
                    try:
                        int(parts[0])
                        data_start = i
                        break
                    except ValueError:
                        continue

            print(f"Found customer data starting at line {data_start + 1}")

            # Parse customer data
            for i in range(data_start, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 7:
                    try:
                        customer = Customer(
                            id=int(parts[0]),
                            x=float(parts[1]),
                            y=float(parts[2]),
                            demand=int(parts[3]),
                            ready_time=int(parts[4]),
                            due_time=int(parts[5]),
                            service_time=int(parts[6])
                        )
                        customers.append(customer)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse line {i+1}: {line} - {e}")
                        continue

            if not customers:
                raise ValueError("No customer data found")

            print(f"Loaded {len(customers)} customers")

            num_vehicles = 25
            vehicle_capacity = 200

            total_demand = sum(c.demand for c in customers[1:])
            if total_demand > 0:
                min_vehicles = math.ceil(total_demand / vehicle_capacity)
                num_vehicles = max(min_vehicles, num_vehicles)

            print(f"Using default values: {num_vehicles} vehicles, capacity {vehicle_capacity}")
            print(f"🌍 CO₂ emission factor (γ): {emission_factor} kg CO₂ per km per unit load")

            # Calculate distance matrix
            n = len(customers)
            distance_matrix = np.zeros((n, n), dtype=np.float64)

            for i in range(n):
                for j in range(n):
                    if i != j:
                        dx = customers[i].x - customers[j].x
                        dy = customers[i].y - customers[j].y
                        distance_matrix[i][j] = math.sqrt(dx*dx + dy*dy)

            instance_name = filename.replace('.txt', '').replace('_sample', '')

            return VRPTWInstance(
                name=instance_name,
                num_vehicles=num_vehicles,
                vehicle_capacity=vehicle_capacity,
                customers=customers,
                distance_matrix=distance_matrix,
                emission_factor=emission_factor
            )

        except Exception as e:
            print(f"Error loading Solomon instance: {e}")
            raise ValueError(f"Could not load instance file {filename}: {e}")

class EmissionCalculator:
    """Calculate CO₂ emissions for routes using E_ij = γ × d_ij × L_ij formula"""

    @staticmethod
    def calculate_edge_emission(distance: float, load: int, emission_factor: float) -> float:
        """
        Calculate CO₂ emission for a single edge
        Formula: E_ij = γ × d_ij × L_ij

        Args:
            distance: Distance d_ij (km)
            load: Load L_ij (units)
            emission_factor: γ (kg CO₂ per km per unit load)

        Returns:
            Emission in kg CO₂
        """
        return emission_factor * distance * load

    @staticmethod
    def calculate_route_emissions(route: List[int], instance: VRPTWInstance) -> Dict[str, Any]:
        """
        Calculate total emissions for a single route with detailed breakdown

        The load decreases as deliveries are made along the route:
        - Initially, vehicle carries full route load
        - After each delivery, load reduces by that customer's demand

        Returns:
            Dictionary with emission details including per-edge breakdown
        """
        if not route:
            return {
                'total_emission': 0.0,
                'total_distance': 0.0,
                'total_load_distance': 0.0,
                'edges': []
            }

        edges = []
        total_emission = 0.0
        total_distance = 0.0
        total_load_distance = 0.0

        # Calculate initial load for the route
        route_load = sum(instance.customers[cid].demand for cid in route)
        current_load = route_load

        # From depot to first customer
        first_customer = route[0]
        distance = float(instance.distance_matrix[0][first_customer])
        emission = EmissionCalculator.calculate_edge_emission(
            distance, current_load, instance.emission_factor
        )

        edges.append({
            'from': 0,
            'to': first_customer,
            'distance': distance,
            'load': current_load,
            'emission': emission
        })

        total_emission += emission
        total_distance += distance
        total_load_distance += distance * current_load

        # Between consecutive customers
        for i in range(len(route) - 1):
            from_customer = route[i]
            to_customer = route[i + 1]

            # Update load (reduce by delivered demand)
            current_load -= instance.customers[from_customer].demand

            distance = float(instance.distance_matrix[from_customer][to_customer])
            emission = EmissionCalculator.calculate_edge_emission(
                distance, current_load, instance.emission_factor
            )

            edges.append({
                'from': from_customer,
                'to': to_customer,
                'distance': distance,
                'load': current_load,
                'emission': emission
            })

            total_emission += emission
            total_distance += distance
            total_load_distance += distance * current_load

        # From last customer back to depot
        last_customer = route[-1]
        current_load -= instance.customers[last_customer].demand

        distance = float(instance.distance_matrix[last_customer][0])
        emission = EmissionCalculator.calculate_edge_emission(
            distance, current_load, instance.emission_factor
        )

        edges.append({
            'from': last_customer,
            'to': 0,
            'distance': distance,
            'load': current_load,
            'emission': emission
        })

        total_emission += emission
        total_distance += distance
        total_load_distance += distance * current_load

        return {
            'total_emission': total_emission,
            'total_distance': total_distance,
            'total_load_distance': total_load_distance,
            'edges': edges
        }

    @staticmethod
    def calculate_solution_emissions(routes: List[List[int]], instance: VRPTWInstance) -> Dict[str, Any]:
        """
        Calculate total emissions for all routes in a solution

        Returns:
            Dictionary with comprehensive emission analysis
        """
        route_emissions = []
        total_emission = 0.0
        total_distance = 0.0
        total_load_distance = 0.0

        for i, route in enumerate(routes):
            if not route:
                continue

            route_data = EmissionCalculator.calculate_route_emissions(route, instance)
            route_data['route_id'] = i + 1
            route_data['num_customers'] = len(route)
            route_emissions.append(route_data)

            total_emission += route_data['total_emission']
            total_distance += route_data['total_distance']
            total_load_distance += route_data['total_load_distance']

        return {
            'total_emission_kg_co2': total_emission,
            'total_distance_km': total_distance,
            'total_load_distance_unit_km': total_load_distance,
            'num_routes': len([r for r in routes if r]),
            'emission_per_km': total_emission / total_distance if total_distance > 0 else 0.0,
            'emission_per_route': total_emission / len([r for r in routes if r]) if routes else 0.0,
            'emission_factor_used': instance.emission_factor,
            'route_emissions': route_emissions
        }

class ArtificialFish:
    """Individual fish in the AFSA algorithm - CORRECTED MULTI-OBJECTIVE VERSION"""

    # Class-level parameters for multi-objective fitness
    LAMBDA_DISTANCE = 1.0  # Weight for distance objective
    LAMBDA_EMISSION = 1.0  # Weight for emission objective

    def __init__(self, instance: VRPTWInstance, fish_id: int):
        self.instance = instance
        self.fish_id = fish_id
        self.position = self.generate_random_solution()
        self.total_distance = 0.0
        self.total_emission = 0.0
        self.fitness = self.calculate_fitness()

    def generate_random_solution(self) -> List[List[int]]:
        """Generate random feasible solution"""
        try:
            n_customers = len(self.instance.customers) - 1
            if n_customers <= 0:
                return []

            customers = list(range(1, n_customers + 1))
            random.shuffle(customers)

            routes = []
            current_route = []
            current_load = 0

            for customer_id in customers:
                customer = self.instance.customers[customer_id]

                if current_load + customer.demand <= self.instance.vehicle_capacity:
                    current_route.append(customer_id)
                    current_load += customer.demand
                else:
                    if current_route:
                        routes.append(current_route)
                    current_route = [customer_id]
                    current_load = customer.demand

            if current_route:
                routes.append(current_route)

            return routes

        except Exception:
            return [[1]] if len(self.instance.customers) > 1 else []

    def calculate_fitness(self) -> float:
        """
        Calculate MULTI-OBJECTIVE fitness of the solution

        Fitness = λ1 × Distance + λ2 × Emissions

        Returns:
            Combined fitness value (lower is better)
        """
        if not self.position:
            self.total_distance = float('inf')
            self.total_emission = float('inf')
            return float('inf')

        total_distance = 0.0

        try:
            # Calculate total distance
            for route in self.position:
                if not route:
                    continue

                # Distance from depot to first customer
                total_distance += float(self.instance.distance_matrix[0][route[0]])

                # Distance between consecutive customers
                for i in range(len(route) - 1):
                    total_distance += float(self.instance.distance_matrix[route[i]][route[i+1]])

                # Distance from last customer back to depot
                total_distance += float(self.instance.distance_matrix[route[-1]][0])

            self.total_distance = total_distance

            # Calculate total emissions using EmissionCalculator
            try:
                emissions_data = EmissionCalculator.calculate_solution_emissions(
                    self.position, self.instance
                )
                self.total_emission = emissions_data['total_emission_kg_co2']
            except Exception as e:
                print(f"Warning: Could not calculate emissions for fish {self.fish_id}: {e}")
                self.total_emission = 0.0

            # Multi-objective fitness calculation
            fitness = (self.LAMBDA_DISTANCE * self.total_distance + 
                      self.LAMBDA_EMISSION * self.total_emission)

            return fitness

        except Exception as e:
            print(f"Error calculating fitness for fish {self.fish_id}: {e}")
            self.total_distance = float('inf')
            self.total_emission = float('inf')
            return float('inf')

def calculate_distance(pos1: List[List[int]], pos2: List[List[int]], instance: VRPTWInstance) -> float:
    """Calculate distance between two fish positions (for AFSA behaviors)"""
    try:
        set1 = set()
        set2 = set()

        for route in pos1:
            set1.update(route)

        for route in pos2:
            set2.update(route)

        # Symmetric difference gives measure of solution dissimilarity
        return len(set1.symmetric_difference(set2))

    except:
        return float('inf')

class RouteOptimizer:
    """Neighborhood search operators for route optimization"""

    @staticmethod
    def two_opt_improve(route: List[int], distance_matrix: np.ndarray) -> List[int]:
        """Apply 2-opt improvement to a single route"""
        if len(route) < 4:
            return route[:]

        best_route = route[:]
        best_distance = RouteOptimizer._calculate_route_distance(route, distance_matrix)

        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Create new route by reversing segment between i and j
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_distance = RouteOptimizer._calculate_route_distance(new_route, distance_matrix)

                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        route = new_route
                        break

                if improved:
                    break

        return best_route

    @staticmethod
    def insertion_operator(routes: List[List[int]], instance: VRPTWInstance) -> List[List[int]]:
        """Apply insertion operator to improve routes"""
        if not routes or len(routes) < 2:
            return routes

        improved_routes = [route[:] for route in routes]

        for i, route in enumerate(improved_routes):
            if not route:
                continue

            for customer in route[:]:
                # Try to insert customer in other routes
                for j, other_route in enumerate(improved_routes):
                    if i == j:
                        continue

                    # Check if customer can be moved
                    customer_demand = instance.customers[customer].demand
                    other_load = sum(instance.customers[c].demand for c in other_route)

                    if other_load + customer_demand <= instance.vehicle_capacity:
                        # Try insertion at different positions
                        best_pos = -1
                        best_distance_reduction = 0

                        for pos in range(len(other_route) + 1):
                            # Calculate distance change
                            old_distance = RouteOptimizer._calculate_route_distance(route, instance.distance_matrix)
                            old_distance += RouteOptimizer._calculate_route_distance(other_route, instance.distance_matrix)

                            # Create new routes
                            new_route1 = [c for c in route if c != customer]
                            new_route2 = other_route[:pos] + [customer] + other_route[pos:]

                            new_distance = RouteOptimizer._calculate_route_distance(new_route1, instance.distance_matrix)
                            new_distance += RouteOptimizer._calculate_route_distance(new_route2, instance.distance_matrix)

                            distance_reduction = old_distance - new_distance

                            if distance_reduction > best_distance_reduction:
                                best_distance_reduction = distance_reduction
                                best_pos = pos

                        if best_pos >= 0:
                            # Apply the best insertion
                            improved_routes[i] = [c for c in improved_routes[i] if c != customer]
                            improved_routes[j].insert(best_pos, customer)
                            break

        return [route for route in improved_routes if route]

    @staticmethod
    def crossover_operator(parent1: List[List[int]], parent2: List[List[int]],
                          instance: VRPTWInstance) -> List[List[int]]:
        """Apply crossover operator between two solutions"""
        if not parent1 or not parent2:
            return parent1 if parent1 else parent2

        # Simple order crossover
        all_customers1 = []
        all_customers2 = []

        for route in parent1:
            all_customers1.extend(route)

        for route in parent2:
            all_customers2.extend(route)

        if not all_customers1 or not all_customers2:
            return parent1

        # Create child by taking partial sequence from parent1 and filling with parent2
        n = len(all_customers1)
        start = random.randint(0, n//2)
        end = random.randint(start + 1, n)

        child_sequence = all_customers1[start:end][:]

        for customer in all_customers2:
            if customer not in child_sequence:
                child_sequence.append(customer)

        # Convert back to routes respecting capacity constraints
        return RouteOptimizer._sequence_to_routes(child_sequence, instance)

    @staticmethod
    def _sequence_to_routes(customers: List[int], instance: VRPTWInstance) -> List[List[int]]:
        """Convert customer sequence to feasible routes"""
        routes = []
        current_route = []
        current_load = 0

        for customer_id in customers:
            customer = instance.customers[customer_id]

            if current_load + customer.demand <= instance.vehicle_capacity:
                current_route.append(customer_id)
                current_load += customer.demand
            else:
                if current_route:
                    routes.append(current_route)
                current_route = [customer_id]
                current_load = customer.demand

        if current_route:
            routes.append(current_route)

        return routes

    @staticmethod
    def _calculate_route_distance(route: List[int], distance_matrix: np.ndarray) -> float:
        """Calculate total distance for a single route"""
        if not route:
            return 0.0

        distance = distance_matrix[0][route[0]]  # Depot to first

        for i in range(len(route) - 1):
            distance += distance_matrix[route[i]][route[i+1]]

        distance += distance_matrix[route[-1]][0]  # Last to depot

        return float(distance)
