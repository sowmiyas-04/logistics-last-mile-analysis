# Route Optimization (Pseudocode)

Vehicle Routing Problem using Google OR-Tools. This is a plan, to be implemented in Week 4.

1. Build a distance matrix between the hub and all delivery points
2. Create a routing model with N vehicles
3. Add constraints:
   - vehicle capacity (maximum parcels per vehicle)
   - delivery time windows
4. Set the objective: minimize total distance travelled
5. Solve the model
6. For each vehicle, print the ordered list of stops and total distance
7. Compare total distance and cost with the current manual routes
