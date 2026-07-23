import unittest

import numpy as np

from road_graph_extractor import RoadGraphExtractor


class RoadGraphExtractorTests(unittest.TestCase):
    def test_prune_spurs_keeps_simple_path_without_branch_points(self):
        extractor = RoadGraphExtractor(min_lane_width=2.5, pixels_per_meter=70)

        nodes = [
            {"id": "node_0", "x": 0.0, "y": 0.0},
            {"id": "node_1", "x": 50.0, "y": 0.0},
            {"id": "node_2", "x": 100.0, "y": 0.0},
        ]
        edges = [
            {"id": "edge_0", "from_id": "node_0", "to_id": "node_1"},
            {"id": "edge_1", "from_id": "node_1", "to_id": "node_2"},
        ]

        dist_transform = np.ones((101, 101), dtype=np.float32) * 100.0

        final_nodes, final_edges = extractor._prune_spurs(nodes, edges, dist_transform, [])

        self.assertEqual([n["id"] for n in final_nodes], ["node_0", "node_1", "node_2"])
        self.assertEqual(len(final_edges), 2)

    def test_build_topology_connects_adjacent_segments(self):
        extractor = RoadGraphExtractor(min_lane_width=2.5, pixels_per_meter=70)

        raw_lines = [
            [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0}],
            [{"x": 10.0, "y": 0.0}, {"x": 20.0, "y": 0.0}],
        ]
        branch_nodes_mask = np.zeros((25, 25), dtype=np.uint8)
        occ_clean = np.zeros((25, 25), dtype=np.uint8)

        nodes, edges = extractor._build_topology(raw_lines, branch_nodes_mask, occ_clean)

        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(edges), 2)


if __name__ == "__main__":
    unittest.main()
