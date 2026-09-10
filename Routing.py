#!/usr/bin/env python3

import json
import argparse
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import math

try:
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from PIL import Image
except ImportError:
    pass  # Will handle gracefully for JSON-only mode


class RoutingGraph:
    """Manages the boundary graph and performs routing operations."""

    def __init__(self, boundary_graph: Dict):
        """
        Initialize routing graph from boundary graph JSON.

        Args:
            boundary_graph: Parsed boundary graph JSON (nodes, edges)
        """
        self.boundary_graph = boundary_graph
        self.nodes = {node["id"]: node for node in boundary_graph["nodes"]}
        self.edges = boundary_graph["edges"]
        self.edge_dict: Dict[str, List[str]] = {}  # node_id -> [connected_node_ids]
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Build adjacency list from edges."""
        self.edge_dict = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            from_id = edge["from_id"]
            to_id = edge["to_id"]
            if from_id not in self.edge_dict:
                self.edge_dict[from_id] = []
            if to_id not in self.edge_dict:
                self.edge_dict[to_id] = []
            self.edge_dict[from_id].append(to_id)
            self.edge_dict[to_id].append(from_id)

    def get_node_coordinates(self, node_id: str) -> Tuple[float, float]:
        """Get (x, y) coordinates of a node."""
        node = self.nodes[node_id]
        return node["x"], node["y"]

    def get_all_boundary_nodes(self) -> Set[str]:
        """Return set of all boundary node IDs."""
        return set(self.nodes.keys())

    def is_2regular(self) -> bool:
        """Check if graph is 2-regular (each node has exactly 2 neighbors)."""
        for node_id in self.nodes:
            if len(self.edge_dict[node_id]) != 2:
                return False
        return True


class LeftHandWalker:
    """Performs left-hand-wall walk on a 2-regular graph."""

    def __init__(self, graph: RoutingGraph):
        """
        Initialize walker.

        Args:
            graph: RoutingGraph instance
        """
        self.graph = graph

    def walk_cycle(
        self, start_node: str, prev_node: Optional[str] = None
    ) -> List[str]:
        """
        Perform left-hand-wall walk starting from start_node.

        Walk the cycle by always taking the left-hand side. Returns an ordered list
        of node IDs visited, ending when returning to start (not repeating start).

        Args:
            start_node: Starting node ID
            prev_node: Previous node (to determine incoming direction), None if start

        Returns:
            Ordered list of node IDs [start_node, ..., last_node_before_returning]
        """
        route = [start_node]
        current = start_node
        previous = prev_node

        # If no previous node, pick the first neighbor (deterministic)
        if previous is None:
            neighbors = self.graph.edge_dict[current]
            if len(neighbors) < 2:
                return route  # Single node or dead end
            previous = neighbors[0]  # Arbitrary choice for initial direction

        while True:
            neighbors = self.graph.edge_dict[current]

            # Exclude the previous node
            next_candidates = [n for n in neighbors if n != previous]

            if len(next_candidates) != 1:
                raise ValueError(
                    f"Non-2-regular node {current}: {len(next_candidates)} next options"
                )

            next_node = next_candidates[0]

            if next_node == start_node:
                # Completed the cycle
                break

            route.append(next_node)
            previous = current
            current = next_node

        return route

    def walk_component(self, start_node: str, entry_direction: Tuple[float, float]) -> List[str]:
        """
        Walk a component starting from start_node with a given entry direction.

        For left-hand-wall logic, pick the first edge that's on the left side of
        the incoming direction, then follow the cycle.

        Args:
            start_node: Node to start from
            entry_direction: (dx, dy) direction we're coming from

        Returns:
            Ordered list of nodes in this component cycle
        """
        # Normalize entry direction
        mag = math.sqrt(entry_direction[0] ** 2 + entry_direction[1] ** 2)
        if mag < 1e-9:
            # Undefined entry; pick first neighbor arbitrarily
            neighbors = self.graph.edge_dict[start_node]
            if len(neighbors) < 1:
                return [start_node]
            next_node = neighbors[0]
        else:
            entry_dir = (entry_direction[0] / mag, entry_direction[1] / mag)
            neighbors = self.graph.edge_dict[start_node]
            x0, y0 = self.graph.get_node_coordinates(start_node)

            # Compute left-hand neighbor
            left_neighbor = None
            best_angle = None

            for neighbor in neighbors:
                x1, y1 = self.graph.get_node_coordinates(neighbor)
                dx, dy = x1 - x0, y1 - y0
                neighbor_dir = (dx, dy)
                mag_n = math.sqrt(dx ** 2 + dy ** 2)
                if mag_n < 1e-9:
                    continue
                neighbor_dir = (dx / mag_n, dy / mag_n)

                # Compute left side using cross product: is neighbor left of entry?
                # Cross product: entry × neighbor (in 2D: e_x * n_y - e_y * n_x)
                cross = entry_dir[0] * neighbor_dir[1] - entry_dir[1] * neighbor_dir[0]

                # Angle from entry direction to neighbor (signed)
                angle = math.atan2(cross, entry_dir[0] * neighbor_dir[0] + entry_dir[1] * neighbor_dir[1])

                if left_neighbor is None or angle < best_angle:
                    left_neighbor = neighbor
                    best_angle = angle

            next_node = left_neighbor if left_neighbor else neighbors[0]

        # Now walk the cycle starting from start_node -> next_node
        return self.walk_cycle(start_node, next_node)


class BridgeValidator:
    """Validates bridge segments for intersection-free stitching."""

    def __init__(self, graph: RoutingGraph):
        """Initialize validator with graph."""
        self.graph = graph
        self.bridges: List[Tuple[str, str]] = []  # (from_id, to_id) for each bridge
        self.intersection_buffer = 0.5  # 0.5m buffer

    def segment_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        buffer: float = 0.0,
    ) -> bool:
        """
        Check if segment p1-p2 intersects segment p3-p4 (with buffer tolerance).

        Returns True if intersection (excluding shared endpoints).

        Args:
            p1, p2: Endpoints of first segment
            p3, p4: Endpoints of second segment
            buffer: Expansion buffer around segments

        Returns:
            True if segments intersect or touch within buffer
        """
        # Parametric intersection test
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-9:
            # Parallel or collinear
            return False

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        eps = 1e-6
        # Exclude intersections at shared endpoints
        if (abs(t) < eps or abs(t - 1) < eps) and (abs(u) < eps or abs(u - 1) < eps):
            return False

        # Check if within segment bounds (with buffer)
        if buffer > 0:
            return (
                -buffer <= t <= 1 + buffer
                and -buffer <= u <= 1 + buffer
            )
        else:
            return 0 <= t <= 1 and 0 <= u <= 1

    def bridge_is_valid(
        self, node_a: str, node_b: str, main_route: List[str]
    ) -> bool:
        """
        Check if bridge a->b is valid (doesn't intersect boundary edges or prior bridges).

        Args:
            node_a: First endpoint
            node_b: Second endpoint
            main_route: Current main route

        Returns:
            True if bridge is valid
        """
        xa, ya = self.graph.get_node_coordinates(node_a)
        xb, yb = self.graph.get_node_coordinates(node_b)

        bridge_seg = ((xa, ya), (xb, yb))

        # Check against all boundary edges
        for edge in self.graph.edges:
            from_id, to_id = edge["from_id"], edge["to_id"]
            x1, y1 = self.graph.get_node_coordinates(from_id)
            x2, y2 = self.graph.get_node_coordinates(to_id)

            # Skip if endpoints match
            if (from_id == node_a or from_id == node_b or
                to_id == node_a or to_id == node_b):
                continue

            if self.segment_intersect(
                (xa, ya), (xb, yb), (x1, y1), (x2, y2), self.intersection_buffer
            ):
                return False

        # Check against prior bridges
        for bridge_from, bridge_to in self.bridges:
            xb1, yb1 = self.graph.get_node_coordinates(bridge_from)
            xb2, yb2 = self.graph.get_node_coordinates(bridge_to)
            if self.segment_intersect(
                (xa, ya), (xb, yb), (xb1, yb1), (xb2, yb2)
            ):
                return False

        return True

    def add_bridge(self, from_id: str, to_id: str) -> None:
        """Record a bridge segment."""
        self.bridges.append((from_id, to_id))


class PatrolRouter:
    """High-level routing orchestration."""

    def __init__(self, boundary_graph: Dict, main_graph: Optional[Dict] = None):
        """
        Initialize router.

        Args:
            boundary_graph: Parsed boundary graph JSON
            main_graph: Optional parsed main graph JSON (for entrance_exit nodes)
        """
        self.graph = RoutingGraph(boundary_graph)
        self.walker = LeftHandWalker(self.graph)
        self.validator = BridgeValidator(self.graph)
        self.main_graph = main_graph
        self.main_route: List[str] = []
        self.unconnected: List[List[str]] = []

    def find_start_node(self, entrance_side: str = "perp_side_1") -> str:
        """
        Find a suitable start node: preferably a boundary child of entrance_exit.

        If no entrance_exit nodes exist, pick an arbitrary boundary node.

        Args:
            entrance_side: Preferred side ("perp_side_1" or "perp_side_2")

        Returns:
            Node ID to start from
        """
        # Try to find entrance_exit nodes
        entrance_nodes: Set[str] = set()
        if self.main_graph and "nodes" in self.main_graph:
            for node in self.main_graph["nodes"]:
                if node.get("type") == "entrance_exit":
                    entrance_nodes.add(node["id"])

        # Find boundary children of entrance nodes
        for node_id, node in self.graph.nodes.items():
            parent = node.get("parent_node")
            # Handle parent_node as either single value or list
            parents = [parent] if not isinstance(parent, list) else parent
            
            # Check if any parent is an entrance node
            if any(p in entrance_nodes for p in parents) and node.get("side") == entrance_side:
                return node_id

        # Fallback: return a random boundary node
        all_boundary = self.graph.get_all_boundary_nodes()
        return random.choice(list(all_boundary))

    def compute_route(self, entrance_side: str = "perp_side_1") -> Tuple[List[str], List[List[str]]]:
        """
        Compute the patrol route.

        Args:
            entrance_side: Preferred side for start node

        Returns:
            (main_route list, unconnected components list)
        """
        print("[Routing] Starting route computation...")

        start_node = self.find_start_node(entrance_side)
        print(f"[Routing] Start node: {start_node}")

        # Step 2: Left-hand walk from start
        print("[Routing] Performing left-hand walk...")
        self.main_route = self.walker.walk_cycle(start_node)
        visited_nodes = set(self.main_route)
        print(f"[Routing] Main route covers {len(self.main_route)} nodes")

        # Step 3: Find disconnected components
        all_nodes = self.graph.get_all_boundary_nodes()
        unvisited = all_nodes - visited_nodes
        components: List[List[str]] = []

        if unvisited:
            print(f"[Routing] Found {len(unvisited)} unvisited nodes, extracting components...")
            components = self._extract_components(unvisited)
            print(f"[Routing] Extracted {len(components)} component(s)")
        else:
            print("[Routing] All nodes visited in main route")

        # Step 4: Stitch components
        if components:
            print("[Routing] Stitching components into main route...")
            self._stitch_components(components, entrance_side)
        else:
            self.unconnected = []

        print(f"[Routing] Final route: {len(self.main_route)} nodes")
        print(f"[Routing] Unconnected components: {len(self.unconnected)}")

        return self.main_route, self.unconnected

    def _extract_components(self, unvisited: Set[str]) -> List[List[str]]:
        """
        Extract connected components from unvisited nodes.

        Args:
            unvisited: Set of unvisited node IDs

        Returns:
            List of components, each as a list of node IDs
        """
        components = []
        remaining = unvisited.copy()

        while remaining:
            start = sorted(remaining)[0]  # Deterministic ordering
            component = self._bfs_component(start, remaining)
            components.append(component)
            remaining -= set(component)

        return components

    def _bfs_component(self, start: str, remaining: Set[str]) -> List[str]:
        """
        BFS to find all nodes in the connected component containing start.

        Args:
            start: Starting node
            remaining: Set of available nodes

        Returns:
            List of nodes in this component
        """
        visited = set()
        queue = [start]
        component = []

        while queue:
            node = queue.pop(0)
            if node in visited or node not in remaining:
                continue
            visited.add(node)
            component.append(node)

            for neighbor in self.graph.edge_dict.get(node, []):
                if neighbor not in visited and neighbor in remaining:
                    queue.append(neighbor)

        return sorted(component)  # Deterministic ordering

    def _stitch_components(self, components: List[List[str]], entrance_side: str) -> None:
        """
        Stitch components into main route using bridge segments.

        Args:
            components: List of unvisited components
            entrance_side: For component entry direction selection
        """
        for component in components:
            print(f"[Routing] Processing component with {len(component)} nodes...")

            # Find best bridge candidate
            candidates = []
            for node_a in self.main_route:
                for node_b in component:
                    dist = self._euclidean_distance(node_a, node_b)
                    candidates.append((dist, node_a, node_b))

            candidates.sort()

            best_pair = None
            for dist, node_a, node_b in candidates:
                if self.validator.bridge_is_valid(node_a, node_b, self.main_route):
                    best_pair = (node_a, node_b)
                    break

            if not best_pair:
                print(
                    f"[Routing] WARNING: No valid bridge found for component: {component}"
                )
                self.unconnected.append(component)
                continue

            node_a, node_b = best_pair
            print(f"[Routing] Bridge: {node_a} -> {node_b}")

            # Build component tour starting from node_b
            # Compute entry direction: from node_a to node_b
            xa, ya = self.graph.get_node_coordinates(node_a)
            xb, yb = self.graph.get_node_coordinates(node_b)
            entry_dir = (xb - xa, yb - ya)

            if self.graph.is_2regular():
                # Walk component as cycle
                component_tour = self.walker.walk_component(node_b, entry_dir)
            else:
                # Fallback to DFS
                print(
                    f"[Routing] WARNING: Component not 2-regular, using DFS fallback"
                )
                component_tour = self._dfs_tour(node_b, component)

            # Splice into main route
            self._splice_route(node_a, node_b, component_tour)
            self.validator.add_bridge(node_a, node_b)
            self.validator.add_bridge(node_b, node_a)

    def _euclidean_distance(self, node_a: str, node_b: str) -> float:
        """Compute Euclidean distance between two nodes."""
        xa, ya = self.graph.get_node_coordinates(node_a)
        xb, yb = self.graph.get_node_coordinates(node_b)
        return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

    def _dfs_tour(self, start: str, component: List[str]) -> List[str]:
        """
        DFS out-and-back tour as fallback for non-2-regular components.

        Args:
            start: Starting node
            component: List of nodes in component

        Returns:
            Ordered tour
        """
        visited = set()
        tour = []

        def dfs(node):
            visited.add(node)
            tour.append(node)
            for neighbor in self.graph.edge_dict.get(node, []):
                if neighbor in component and neighbor not in visited:
                    dfs(neighbor)
                    tour.append(node)

        dfs(start)
        return tour

    def _splice_route(self, node_a: str, node_b: str, component_tour: List[str]) -> None:
        """
        Splice component tour into main route at node_a.

        Replaces: ... z, a, x ... with ... z, a, b, [component_interior], b, a, x ...

        Args:
            node_a: Connection point in main route
            node_b: Start of component tour
            component_tour: Component tour [b, ..., b] (b is first and last)
        """
        idx = self.main_route.index(node_a)

        # Extract component interior (skip repeated b at end if present)
        component_interior = component_tour[1:-1] if len(component_tour) > 1 else []

        # Build splice: [... a ...] + [b, interior, b] + [... continue ...]
        splice = [node_b] + component_interior + [node_b]

        self.main_route = (
            self.main_route[: idx + 1] + splice + self.main_route[idx + 1 :]
        )

    def build_output(self) -> Dict:
        """
        Build output JSON structure with ordered waypoints.

        Returns:
            Dictionary ready for JSON serialization
        """
        # Close the loop: append start node at end
        final_route = self.main_route + [self.main_route[0]]

        # Build route entries
        route_entries = []
        for order, node_id in enumerate(final_route):
            x, y = self.graph.get_node_coordinates(node_id)
            entry = {
                "order": str(order),
                "id": node_id,
                "x": float(x),
                "y": float(y),
                "type": "start" if order == 0 else "waypoint",
            }
            route_entries.append(entry)

        return {
            "route": route_entries,
            "unconnected": self.unconnected,
        }


def main(output_dir=None, entrance_side="perp_side_1"):
    """Main entry point."""
    if output_dir is None:
        parser = argparse.ArgumentParser(
            description="Compute closed patrol route from boundary graph"
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Output directory (default: <script_dir>/output/Smart_Parking_Park)",
        )
        parser.add_argument(
            "--entrance-side",
            default=entrance_side,
            choices=["perp_side_1", "perp_side_2"],
            help="Preferred entrance side",
        )

        args, _ = parser.parse_known_args()
        output_dir = args.output_dir
        entrance_side = args.entrance_side

    # Determine output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        default_dir = script_dir / "output" / "Smart_Parking_Park"
        if not default_dir.exists():
            default_dir = script_dir / "output" / "Smart Parking Park"
        output_dir = default_dir
    else:
        output_dir = Path(output_dir)

    # Normalize space vs underscore differences if needed
    if not output_dir.exists():
        alt_space = output_dir.parent / output_dir.name.replace("_", " ")
        alt_under = output_dir.parent / output_dir.name.replace(" ", "_")
        if alt_space.exists():
            output_dir = alt_space
        elif alt_under.exists():
            output_dir = alt_under

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Routing] Output directory: {output_dir}")

    # Load boundary graph
    boundary_graph_path = None
    candidates = sorted(output_dir.glob("*_boundary_graph.json"))
    if candidates:
        boundary_graph_path = candidates[0]
    else:
        for fallback_name in [
            f"{output_dir.name}_boundary_graph.json",
            f"{output_dir.name.replace(' ', '_')}_boundary_graph.json",
            "Smart_Parking_Park_boundary_graph.json",
        ]:
            p = output_dir / fallback_name
            if p.exists():
                boundary_graph_path = p
                break

    if not boundary_graph_path or not boundary_graph_path.exists():
        print(f"[Routing] ERROR: Boundary graph not found in {output_dir}")
        return None

    print(f"[Routing] Loading boundary graph from {boundary_graph_path}")
    with open(boundary_graph_path) as f:
        boundary_graph = json.load(f)

    # Load main graph (optional)
    main_graph = None
    auto_graphs = sorted((output_dir / "Automated_Output").glob("*_graph.json"))
    manual_graphs = sorted((output_dir / "Manual_Output").glob("*.json")) + sorted((output_dir / "Manuel_Output").glob("*.json"))
    for main_graph_path in auto_graphs + manual_graphs + [
        output_dir / "Automated_Output" / "Smart_Parking_Park_graph.json",
        output_dir / "Manual_Output" / "Smart_Parking_Park_graph.json",
    ]:
        if main_graph_path.exists():
            print(f"[Routing] Loading main graph from {main_graph_path}")
            with open(main_graph_path) as f:
                main_graph = json.load(f)
            break

    # Compute route
    router = PatrolRouter(boundary_graph, main_graph)
    main_route, unconnected = router.compute_route(entrance_side)

    # Build output
    output_data = router.build_output()

    # Save JSON
    output_json_path = output_dir / "patrol_route.json"
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"[Routing] Saved route to {output_json_path}")

    # Also save with normalized carpark name prefix
    carpark_name = output_dir.name.replace(" ", "_")
    named_route_path = output_dir / f"{carpark_name}_patrol_route.json"
    if named_route_path != output_json_path:
        with open(named_route_path, "w") as f:
            json.dump(output_data, f, indent=2)

    # Validate
    print("[Routing] Validating output...")
    total_route_nodes = len(main_route) + 1  # +1 for closed loop
    print(f"[Routing] Total nodes in closed route: {total_route_nodes}")
    total_boundary_nodes = len(boundary_graph["nodes"])
    print(f"[Routing] Total boundary nodes: {total_boundary_nodes}")

    if unconnected:
        total_unconnected = sum(len(c) for c in unconnected)
        print(f"[Routing] WARNING: {total_unconnected} nodes unconnected")
    else:
        print("[Routing] All nodes connected!")

    print("[Routing] Done!")
    return output_data


if __name__ == "__main__":
    main()
