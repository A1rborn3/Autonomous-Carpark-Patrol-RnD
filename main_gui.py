import sys
import os
# Ensure the directory containing this script is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk, messagebox
from unified_pipeline_gui import UnifiedPipelineGUI
from parking_annotator import ParkingAnnotatorApp
from robot_line_plotter import RobotLinePlotterApp
from json_exporter import JSONExporter
from unitree_exporter import UnitreeGo2Exporter


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Carpark Patrol - Unified GUI")
        self.root.geometry("1500x800")
        
        # Setup Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Map Generation
        self.tab_pipeline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pipeline, text="Step 1: Map Generation")
        
        # Tab 2: Carpark Annotator
        self.tab_annotator = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_annotator, text="Step 2: Carpark Annotator")
        
        # Tab 3: Robot Line Plotter
        self.tab_line_plotter = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_line_plotter, text="Step 3: Navigation Plotter")
        
        # Instantiate Apps into Tabs
        self.pipeline_app = UnifiedPipelineGUI(
            self.tab_pipeline,
            on_pipeline_finished=self.on_pipeline_finished,
            on_map_generated=self.on_map_generated,
            on_graph_extracted=self.on_graph_extracted
        )
        self.annotator_app = ParkingAnnotatorApp(
            self.tab_annotator,
            on_annotations_saved=self.on_annotations_saved
        )
        self.line_plotter_app = RobotLinePlotterApp(
            self.tab_line_plotter,
            on_extract_graph=self.on_extract_graph_clicked,
            on_route_saved=self.on_route_saved,
            on_export_manual_route=self.on_export_manual_route
        )

        
    def on_map_generated(self, ortho_path):
        """Callback triggered when the pipeline finishes generating a map image."""
        self.annotator_app.load_image(ortho_path)
        self.line_plotter_app.load_image(ortho_path)
        # Auto-switch to Carpark Annotator tab
        self.notebook.select(self.tab_annotator)

    def on_pipeline_finished(self, ortho_path):
        """Callback triggered when the pipeline finishes generating a map."""
        self.annotator_app.load_image(ortho_path)
        self.line_plotter_app.load_image(ortho_path)
        self.notebook.select(self.tab_annotator)
        
    def on_annotations_saved(self, ortho_path):
        """Callback triggered when carpark annotations are saved."""
        self.line_plotter_app.load_image(ortho_path)
        # Auto-switch to Robot Line Plotter tab
        self.notebook.select(self.tab_line_plotter)

    def on_route_saved(self, ortho_path):
        """Callback triggered when robot route lines are saved."""
        pass

    def _get_export_context(self):
        """Gets output directory, bounds, resolution, and basename from pipeline or line plotter."""
        img_path = self.line_plotter_app.image_path
        if img_path:
            base_dir = os.path.dirname(img_path)
            basename = os.path.splitext(os.path.basename(img_path))[0]
        else:
            base_dir = self.pipeline_app.output_dir.get()
            basename = self.pipeline_app.file_basename or "road_graph"

        meta = self.line_plotter_app.map_metadata or {}
        bounds = self.pipeline_app.current_bounds or meta.get("bounds", {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100})
        res = self.pipeline_app.resolution.get() if hasattr(self.pipeline_app, 'resolution') else meta.get("resolution", 70)

        return base_dir, bounds, res, basename

    def on_export_manual_route(self, nodes, edges):
        """Callback triggered when user exports manual route waypoints directly."""
        try:
            out_dir, bounds, res, basename = self._get_export_context()

            # Save manual outputs to a dedicated subfolder
            manual_output_dir = os.path.join(out_dir, "Manual Output")
            os.makedirs(manual_output_dir, exist_ok=True)

            # Export Cartesian JSON graph for manual route
            json_exp = JSONExporter(manual_output_dir)
            json_exp.export_graph(nodes, edges, bounds, res, filename=f"{basename}_robot_route.json")

            # Export Unitree Go2 waypoints & patrol runner script
            unitree_exp = UnitreeGo2Exporter(manual_output_dir)
            unitree_exp.export_unitree_waypoints(nodes, edges, bounds, res, filename_prefix=basename)

            messagebox.showinfo(
                "Manual Route Exported",
                f"Exported manual route with {len(nodes)} waypoints!\n\n"
                f"Saved to: Manual Output/\n- {basename}_robot_route.json\n- run_go2_patrol.py"
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export manual route:\n{e}")

    def on_extract_graph_clicked(self):
        """Triggered by the 'Extract Automated Graph' button on Tab 3."""
        if not self.pipeline_app.current_occ_path:
            img_path = self.line_plotter_app.image_path
            if img_path:
                base_dir = os.path.dirname(img_path)
                self.pipeline_app.file_basename = os.path.splitext(os.path.basename(img_path))[0]
                self.pipeline_app.current_occ_path = os.path.join(base_dir, "obstacle_occupancy.png")
                if self.line_plotter_app.map_metadata:
                    self.pipeline_app.current_bounds = self.line_plotter_app.map_metadata.get("bounds")
                    self.pipeline_app.resolution.set(self.line_plotter_app.map_metadata.get("resolution", 70))
        self.pipeline_app.run_graph_extraction()

    def on_graph_extracted(self, nodes, edges):
        """Callback triggered when automated graph extraction finishes."""
        self.line_plotter_app.set_extracted_graph(nodes, edges)
        self.notebook.select(self.tab_line_plotter)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()


