"""
Complete IFACO Implementation with CO₂ Emission Tracking
Extended to include emission calculation and reporting
"""

from ifaco_core_co2 import *
from afsa import ArtificialFishSwarmAlgorithm
from aco import ImprovedAntColonyOptimization
import json
import traceback
import time

def convert_to_serializable(obj):
    """Convert numpy types to JSON-compatible types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

class IFACO_CO2:
    """IFACO Algorithm Implementation with CO₂ Emission Tracking"""

    def __init__(self, instance_file: str, emission_factor: float = 0.05):
        """
        Initialize IFACO with CO₂ emission tracking

        Args:
            instance_file: Path to Solomon instance file
            emission_factor: γ (kg CO₂ per km per unit load)
                           Default: 0.05
                           Range: 0.04-0.065 based on road transport research
        """
        print("="*80)
        print("IFACO WITH CO₂ EMISSION TRACKING")
        print("Novelty: E_ij = γ × d_ij × L_ij")
        print("="*80)

        # Load instance with emission factor
        self.instance = SolomonDataLoader.load_solomon_instance(instance_file, emission_factor)

        print(f"\nLoaded instance: {self.instance.name}")
        print(f"Customers: {len(self.instance.customers)}")
        print(f"Vehicle capacity: {self.instance.vehicle_capacity}")
        print(f"Max vehicles: {self.instance.num_vehicles}")
        print(f"🌍 Emission factor: {self.instance.emission_factor} kg CO₂/(km·unit load)")

        # Algorithm parameters
        print("\n--- ALGORITHM PARAMETERS ---")
        self.fish_count = 50
        self.afsa_iterations = 100
        print(f"AFSA: {self.fish_count} fish, {self.afsa_iterations} iterations")

        self.num_ants = 100
        self.aco_iterations = 100
        self.alpha = 1.0
        self.beta = 3.0
        self.lamda = 1.5
        self.rho = 0.25
        self.q = 100.0
        print(f"ACO: {self.num_ants} ants, {self.aco_iterations} iterations")
        print(f"Parameters: α={self.alpha}, β={self.beta}, λ={self.lamda}, ρ={self.rho}")

        # Results storage
        self.results = {
            'instance_name': self.instance.name,
            'emission_factor': self.instance.emission_factor,
            'parameters': {
                'fish_count': self.fish_count,
                'afsa_iterations': self.afsa_iterations,
                'num_ants': self.num_ants,
                'aco_iterations': self.aco_iterations,
                'alpha': self.alpha,
                'beta': self.beta,
                'lambda': self.lamda,
                'rho': self.rho,
                'q': self.q
            },
            'afsa_results': {},
            'aco_results': {},
            'final_solution': {},
            'emission_analysis': {}
        }

    def run(self) -> dict:
        """Run complete IFACO algorithm with CO₂ tracking"""
        print("\n" + "="*60)
        print("STARTING IFACO WITH CO₂ TRACKING")
        print("="*60)

        start_time = time.time()

        try:
            # Phase 1: Run AFSA
            print("\n🐟 PHASE 1: Artificial Fish Swarm Algorithm")
            print("-" * 50)
            afsa_start = time.time()

            afsa = ArtificialFishSwarmAlgorithm(
                self.instance,
                fish_count=self.fish_count,
                max_iterations=self.afsa_iterations
            )

            afsa_best_fish = afsa.run()
            afsa_time = time.time() - afsa_start

            # Calculate AFSA emissions
            afsa_emissions = EmissionCalculator.calculate_solution_emissions(
                afsa_best_fish.position, self.instance
            )

            # Store AFSA results
            self.results['afsa_results'] = {
                'best_fitness': float(afsa_best_fish.fitness),
                'best_routes': convert_to_serializable(afsa_best_fish.position),
                'num_routes': len([r for r in afsa_best_fish.position if r]),
                'execution_time': afsa_time,
                'fitness_history': convert_to_serializable(afsa.fitness_history),
                'emissions': convert_to_serializable(afsa_emissions)
            }

            print(f"\n✅ AFSA Results:")
            print(f"   Best fitness (distance): {afsa_best_fish.fitness:.2f} km")
            print(f"   Routes: {len([r for r in afsa_best_fish.position if r])}")
            print(f"   🌍 Total CO₂ emissions: {afsa_emissions['total_emission_kg_co2']:.2f} kg")
            print(f"   Execution time: {afsa_time:.2f} seconds")

            # Phase 2: Run ACO
            print("\n🐜 PHASE 2: Ant Colony Optimization")
            print("-" * 50)
            aco_start = time.time()

            aco = ImprovedAntColonyOptimization(
                self.instance,
                num_ants=self.num_ants,
                max_iterations=self.aco_iterations,
                alpha=self.alpha,
                beta=self.beta,
                lamda=self.lamda,
                rho=self.rho,
                q=self.q
            )

            # Initialize ACO pheromone from AFSA solution
            aco.initialize_pheromone_from_afsa(afsa_best_fish)

            aco_best_ant = aco.run()
            aco_time = time.time() - aco_start

            if aco_best_ant:
                # Calculate ACO emissions
                aco_emissions = EmissionCalculator.calculate_solution_emissions(
                    aco_best_ant.routes, self.instance
                )

                # Store ACO results
                self.results['aco_results'] = {
                    'best_distance': float(aco_best_ant.total_distance),
                    'best_routes': convert_to_serializable(aco_best_ant.routes),
                    'num_routes': len([r for r in aco_best_ant.routes if r]),
                    'total_load': int(aco_best_ant.total_load),
                    'execution_time': aco_time,
                    'iteration_history': convert_to_serializable(aco.iteration_best_distances),
                    'emissions': convert_to_serializable(aco_emissions)
                }

                print(f"\n✅ ACO Results:")
                print(f"   Best distance: {aco_best_ant.total_distance:.2f} km")
                print(f"   Routes: {len([r for r in aco_best_ant.routes if r])}")
                print(f"   Total load: {aco_best_ant.total_load}")
                print(f"   🌍 Total CO₂ emissions: {aco_emissions['total_emission_kg_co2']:.2f} kg")
                print(f"   Execution time: {aco_time:.2f} seconds")

                final_solution = aco_best_ant
                final_distance = aco_best_ant.total_distance
                final_emissions = aco_emissions
            else:
                print("\n⚠️ ACO failed, using AFSA solution")
                final_solution = afsa_best_fish
                final_distance = afsa_best_fish.fitness
                final_emissions = afsa_emissions

                self.results['aco_results'] = {
                    'error': 'ACO failed to find valid solution',
                    'execution_time': aco_time
                }

            # Store final results
            total_time = time.time() - start_time

            if hasattr(final_solution, 'routes'):  # ACO solution
                final_routes = final_solution.routes
            else:  # AFSA solution
                final_routes = final_solution.position

            self.results['final_solution'] = {
                'algorithm': 'ACO' if hasattr(final_solution, 'routes') else 'AFSA',
                'total_distance': float(final_distance),
                'routes': convert_to_serializable(final_routes),
                'num_routes': len([r for r in final_routes if r]),
                'num_customers_served': sum(len(route) for route in final_routes if route),
                'total_execution_time': total_time
            }

            self.results['emission_analysis'] = convert_to_serializable(final_emissions)

            # Final summary
            print("\n" + "="*60)
            print("🎯 FINAL IFACO RESULTS WITH CO₂ TRACKING")
            print("="*60)
            print(f"Algorithm used: {self.results['final_solution']['algorithm']}")
            print(f"Total distance: {final_distance:.2f} km")
            print(f"Number of routes: {len([r for r in final_routes if r])}")
            print(f"Customers served: {sum(len(route) for route in final_routes if route)}")
            print(f"\n🌍 EMISSION RESULTS:")
            print(f"   Total CO₂ emissions: {final_emissions['total_emission_kg_co2']:.2f} kg")
            print(f"   CO₂ per km: {final_emissions['emission_per_km']:.4f} kg/km")
            print(f"   CO₂ per route: {final_emissions['emission_per_route']:.2f} kg")
            print(f"   Total load×distance: {final_emissions['total_load_distance_unit_km']:.2f} unit·km")
            print(f"\nTotal execution time: {total_time:.2f} seconds")

            # Validate solution
            self._validate_solution(final_routes)

            return self.results

        except Exception as e:
            print(f"\n❌ Error in IFACO execution: {e}")
            traceback.print_exc()
            self.results['error'] = str(e)
            self.results['traceback'] = traceback.format_exc()
            return self.results

    def _validate_solution(self, routes: List[List[int]]):
        """Validate the final solution"""
        print("\n🔍 Solution Validation:")
        try:
            all_customers = set()
            total_demand = 0

            for i, route in enumerate(routes):
                if not route:
                    continue

                route_demand = sum(self.instance.customers[c].demand for c in route)
                total_demand += route_demand
                print(f"   Route {i+1}: {len(route)} customers, demand {route_demand}")

                # Check capacity constraint
                if route_demand > self.instance.vehicle_capacity:
                    print(f"   ⚠️ Route {i+1} exceeds capacity!")

                # Track customers
                for customer in route:
                    if customer in all_customers:
                        print(f"   ⚠️ Customer {customer} appears multiple times!")
                    all_customers.add(customer)

            # Check if all customers are served
            expected_customers = set(range(1, len(self.instance.customers)))
            missing = expected_customers - all_customers
            extra = all_customers - expected_customers

            if missing:
                print(f"   ⚠️ Missing customers: {missing}")
            if extra:
                print(f"   ⚠️ Invalid customers: {extra}")

            print(f"   Total demand served: {total_demand}")
            print(f"   Customers served: {len(all_customers)}/{len(expected_customers)}")

            if not missing and not extra:
                print("   ✅ Solution is valid!")

        except Exception as e:
            print(f"   ❌ Validation error: {e}")

    def save_results(self, filename: str = "ifaco_co2_results.json"):
        """Save results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n💾 Results saved to {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")

def main():
    """Main execution function"""
    try:
        instance_file = "C101_sample.txt"

        # You can adjust the emission factor here:
        # - 0.04: Conservative estimate
        # - 0.05: Moderate estimate (default)
        # - 0.065: Higher estimate based on average road transport
        emission_factor = 0.05  # kg CO₂ per km per unit load

        print("IFACO WITH CO₂ EMISSION TRACKING")
        print("Vehicle Routing Problem with Time Windows + Environmental Impact")
        print("\nNovelty: E_ij = γ × d_ij × L_ij (load-based emissions)")

        # Create and run IFACO with CO₂ tracking
        ifaco = IFACO_CO2(instance_file, emission_factor=emission_factor)
        results = ifaco.run()

        # Save results
        ifaco.save_results("ifaco_co2_results.json")

        return results

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
