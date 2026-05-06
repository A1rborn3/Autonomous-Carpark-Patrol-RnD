import tkinter as tk
from tkinter import ttk
from pipeline_gui import PipelineGUI
from parking_annotator import ParkingAnnotatorApp
from waypoint_planner import WaypointPlannerApp

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Carpark Patrol - Unified GUI")
        self.root.geometry("1200x800")
        
        # Setup Notebook (Tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Pipeline
        self.tab_pipeline = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_pipeline, text="Pipeline (Map Generation)")
        
        # Tab 2: Annotator
        self.tab_annotator = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_annotator, text="Parking Annotator")
        
        # Tab 3: Waypoint Planner
        self.tab_waypoint = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_waypoint, text="Waypoint Planner")
        
        # Instantiate Apps into Tabs
        self.pipeline_app = PipelineGUI(self.tab_pipeline, on_map_generated=self.on_map_generated)
        self.annotator_app = ParkingAnnotatorApp(self.tab_annotator, on_annotations_saved=self.on_annotations_saved)
        self.waypoint_app = WaypointPlannerApp(self.tab_waypoint)
        
    def on_map_generated(self, ortho_path):
        """Callback triggered when the pipeline finishes generating a map."""
        # Switch to Annotator Tab
        self.notebook.select(self.tab_annotator)
        # Load the generated image
        self.annotator_app.load_image(ortho_path)
        # Load maps into Waypoint tab
        self.waypoint_app.load_maps(ortho_path)
        
    def on_annotations_saved(self, ortho_path):
        """Callback triggered when annotations are saved to update the waypoint planner."""
        self.waypoint_app.load_maps(ortho_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
