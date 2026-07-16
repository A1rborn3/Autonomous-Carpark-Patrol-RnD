import sys
import os
# Ensure the directory containing this script is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk
from unified_pipeline_gui import UnifiedPipelineGUI
from parking_annotator import ParkingAnnotatorApp


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
        self.notebook.add(self.tab_pipeline, text="Pipeline (Map & Graph)")
        
        # Tab 2: Annotator
        self.tab_annotator = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_annotator, text="Parking Annotator")
        
        # Instantiate Apps into Tabs
        self.pipeline_app = UnifiedPipelineGUI(
            self.tab_pipeline,
            on_pipeline_finished=self.on_pipeline_finished,
            on_map_generated=self.on_map_generated
        )
        self.annotator_app = ParkingAnnotatorApp(self.tab_annotator, on_annotations_saved=self.on_annotations_saved)

        
    def on_map_generated(self, ortho_path):
        """Callback triggered when the pipeline finishes generating a map image."""
        self.annotator_app.load_image(ortho_path)

    def on_pipeline_finished(self, ortho_path):
        """Callback triggered when the pipeline finishes generating a map and graph."""
        # Switch to Annotator Tab
        self.notebook.select(self.tab_annotator)
        # Load the generated image if not already loaded (or reload it)
        self.annotator_app.load_image(ortho_path)
        
    def on_annotations_saved(self, ortho_path):
        """Callback triggered when annotations are saved."""
        # If we had a viewer for the final combined data, we could update it here
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
