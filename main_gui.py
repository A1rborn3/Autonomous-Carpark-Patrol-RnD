import tkinter as tk
from tkinter import ttk
from pipeline_gui import PipelineGUI
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
        self.notebook.add(self.tab_pipeline, text="Pipeline (Map Generation)")
        
        # Tab 2: Annotator
        self.tab_annotator = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_annotator, text="Parking Annotator")
        
        # Instantiate Apps into Tabs
        self.pipeline_app = PipelineGUI(self.tab_pipeline, on_map_generated=self.on_map_generated)
        self.annotator_app = ParkingAnnotatorApp(self.tab_annotator)
        
    def on_map_generated(self, ortho_path):
        """Callback triggered when the pipeline finishes generating a map."""
        # Switch to Annotator Tab
        self.notebook.select(self.tab_annotator)
        # Load the generated image
        self.annotator_app.load_image(ortho_path)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
