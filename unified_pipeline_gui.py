import sys
import os
# Ensure the directory containing this script is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import our pipeline modules
from point_cloud_processor import PointCloudProcessor
from map_generator import MapGenerator
from road_graph_extractor import RoadGraphExtractor
from json_exporter import JSONExporter
from unitree_exporter import UnitreeGo2Exporter


class UnifiedPipelineGUI:
    def __init__(self, root, on_pipeline_finished=None, on_map_generated=None):
        self.root = root
        self.on_pipeline_finished = on_pipeline_finished
        self.on_map_generated = on_map_generated
        
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
        self.voxel_size = tk.DoubleVar(value=0.02)
        self.obs_height = tk.DoubleVar(value=0.1)
        self.max_height = tk.DoubleVar(value=2.0)
        self.resolution = tk.DoubleVar(value=70)
        self.min_lane_width = tk.DoubleVar(value=2.5)
        
        self.processing_state = "idle" # idle, mapping, waiting_edit, extracting
        self.current_ortho = None
        self.current_bounds = None
        self.current_occ_path = None
        self.file_basename = ""
        
        self.setup_ui()
        
    def setup_ui(self):
        # Left Panel for Controls
        self.left_panel = tk.Frame(self.root, width=300, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(self.left_panel, text="Pipeline Configuration", font=("Arial", 12, "bold")).pack(pady=(0, 10))
        
        # File Selection
        tk.Label(self.left_panel, text="PLY Input:").pack(anchor=tk.W)
        file_frame = tk.Frame(self.left_panel)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        tk.Entry(file_frame, textvariable=self.input_file).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(file_frame, text="...", command=self.browse_file).pack(side=tk.RIGHT)
        
        # Parameters
        params_frame = tk.LabelFrame(self.left_panel, text="Parameters", padx=5, pady=5)
        params_frame.pack(fill=tk.X, pady=10)
        
        self._add_param(params_frame, "Voxel Size (m):", self.voxel_size, 0)
        self._add_param(params_frame, "Min Obs H (m):", self.obs_height, 1)
        self._add_param(params_frame, "Max Height (m):", self.max_height, 2)
        self._add_param(params_frame, "Resolution (px/m):", self.resolution, 3)
        self._add_param(params_frame, "Min Lane W (m):", self.min_lane_width, 4)
        
        # Buttons
        self.btn_run = tk.Button(self.left_panel, text="Start Pipeline", command=self.run_full_pipeline, bg="green", fg="black", font=("Arial", 11, "bold"))
        self.btn_run.pack(fill=tk.X, pady=5)
        
        self.btn_extract_only = tk.Button(self.left_panel, text="Extract Graph Only", command=self.run_graph_extraction, state=tk.DISABLED)
        self.btn_extract_only.pack(fill=tk.X, pady=5)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.left_panel, text="Status:").pack(anchor=tk.W, pady=(10, 0))
        self.status_label = tk.Label(self.left_panel, textvariable=self.status_var, fg="blue", wraplength=250, justify=tk.LEFT)
        self.status_label.pack(anchor=tk.W)
        
        # Right Panel for Visualization
        self.right_panel = tk.Frame(self.root)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right_panel)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _add_param(self, parent, label, var, row):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W)
        tk.Entry(parent, textvariable=var, width=10).grid(row=row, column=1, sticky=tk.E, padx=5, pady=2)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PLY Files", "*.ply"), ("All Files", "*.*")])
        if file_path:
            self.input_file.set(file_path)
            
    def run_full_pipeline(self):
        input_path = self.input_file.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Select a valid PLY file.")
            return
            
        self.set_state("mapping")
        threading.Thread(target=self._pipeline_thread, args=(input_path,), daemon=True).start()
        
    def set_state(self, state):
        self.processing_state = state
        if state == "idle":
            self.btn_run.config(state=tk.NORMAL, text="Start Pipeline")
            self.btn_extract_only.config(state=tk.DISABLED)
        elif state == "mapping":
            self.btn_run.config(state=tk.DISABLED)
            self.btn_extract_only.config(state=tk.DISABLED)
        elif state == "waiting_edit":
            self.btn_run.config(state=tk.NORMAL, text="Re-run Pipeline", bg="orange")
            self.btn_extract_only.config(state=tk.NORMAL)
            messagebox.showinfo("Step Finished", "Map generated. You can manually edit the occupancy map or proceed to graph extraction.")
            
    def _pipeline_thread(self, input_path):
        try:
            filename = os.path.basename(input_path)
            self.file_basename = os.path.splitext(filename)[0]
            output_dir = os.path.join(self.output_dir.get(), self.file_basename)
            if not os.path.exists(output_dir): os.makedirs(output_dir)
            
            # 1. Processing
            self.root.after(0, lambda: self.status_var.set("Loading & Processing Cloud..."))
            processor = PointCloudProcessor(self.voxel_size.get(), self.obs_height.get(), self.max_height.get())
            pcd = processor.load_ply(input_path)
            pcd = processor.clean_and_downsample(pcd)
            ground_inliers, non_ground, plane_model = processor.segment_ground(pcd)
            pcd = processor.align_to_ground(pcd, plane_model)
            ground_pts, obs_pts = processor.extract_obstacles(pcd)
            
            # 2. Map Generation
            self.root.after(0, lambda: self.status_var.set("Generating 2D Maps..."))
            map_gen = MapGenerator(resolution=self.resolution.get())
            self.current_ortho, occ, self.current_bounds = map_gen.generate_maps(ground_pts, obs_pts)
            
            ortho_path = os.path.join(output_dir, "orthomosaic.png")
            self.current_occ_path = os.path.join(output_dir, "obstacle_occupancy.png")
            map_gen.save_maps(self.current_ortho, occ, self.current_bounds, ortho_path, self.current_occ_path)
            
            self.root.after(0, lambda: self.display_map(self.current_ortho, self.current_bounds))
            self.root.after(0, lambda: self.status_var.set(f"Map ready. Saved to {output_dir}"))
            self.root.after(0, lambda: self.set_state("waiting_edit"))
            if self.on_map_generated:
                self.root.after(0, lambda: self.on_map_generated(ortho_path))
            
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
            self.root.after(0, lambda: self.set_state("idle"))


    def run_graph_extraction(self):
        if not self.current_occ_path: return
        self.status_var.set("Extracting Road Graph...")
        threading.Thread(target=self._extraction_thread, daemon=True).start()
        
    def _extraction_thread(self):
        try:
            output_dir = os.path.join(self.output_dir.get(), self.file_basename)
            occ_path = os.path.join(output_dir, "obstacle_occupancy.png")
            park_json_path = os.path.join(output_dir, "orthomosaic_parking_spaces.json")
            
            occ = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE)
            
            # Merge parking annotations from JSON if they exist
            if os.path.exists(park_json_path):
                self.root.after(0, lambda: self.status_var.set("Applying parking annotations from JSON..."))
                map_gen = MapGenerator(resolution=self.resolution.get())
                occ = map_gen.apply_parking_annotations(occ, park_json_path)
                
                # Save the merged occupancy used for graph extraction
                merged_path = os.path.join(output_dir, "merged_occupancy.png")
                cv2.imwrite(merged_path, occ)
                self.status_var.set(f"Merged occupancy saved to {os.path.basename(merged_path)}")
            
            # Also support the direct image merge as a secondary check/live update
            park_img_path = os.path.join(output_dir, "parking_occupancy.png")
            if os.path.exists(park_img_path) and not os.path.exists(park_json_path):
                park_occ = cv2.imread(park_img_path, cv2.IMREAD_GRAYSCALE)
                if park_occ is not None:
                    occ = cv2.min(occ, park_occ)

            
            # If occ was merged, it might be BGR (color). We need to extract the blue mask
            # from the color version, then convert the map to grayscale for the skeletonizer.
            if len(occ.shape) == 3:
                # BGR: Pure Blue is [255, 0, 0]
                blue_mask = cv2.inRange(occ, np.array([200, 0, 0]), np.array([255, 50, 50]))
                if cv2.countNonZero(blue_mask) == 0:
                    blue_mask = None
                # Convert to grayscale for graph extraction
                occ = cv2.cvtColor(occ, cv2.COLOR_BGR2GRAY)
            else:
                # If grayscale, check the original color file for blue marks
                occ_color = cv2.imread(occ_path, cv2.IMREAD_COLOR)
                blue_mask = cv2.inRange(occ_color, np.array([200, 0, 0]), np.array([255, 50, 50]))
                if cv2.countNonZero(blue_mask) == 0:
                    blue_mask = None


            
            graph_ext = RoadGraphExtractor(min_lane_width=self.min_lane_width.get(), pixels_per_meter=self.resolution.get())
            nodes, edges = graph_ext.extract_graph(occ, output_dir, blue_mask)

            
            # JSON Export
            json_exp = JSONExporter(output_dir)
            json_path = json_exp.export_graph(nodes, edges, self.current_bounds, self.resolution.get(), filename=f"{self.file_basename}_graph.json")
            
            # Unitree Go2 Export
            unitree_exp = UnitreeGo2Exporter(output_dir)
            unitree_exp.export_unitree_waypoints(nodes, edges, self.current_bounds, self.resolution.get(), filename_prefix=self.file_basename)

            self.root.after(0, lambda: self.display_graph(nodes, edges))
            self.root.after(0, lambda: self.status_var.set(f"Graph extracted & Unitree Go2 waypoints saved."))

            
            if self.on_pipeline_finished:
                ortho_path = os.path.join(output_dir, "orthomosaic.png")
                self.root.after(0, lambda: self.on_pipeline_finished(ortho_path))
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Extraction Error", err_msg))


    def display_map(self, ortho_bgr, bounds):
        rgb = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        extent = [bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']]
        self.ax.imshow(rgb, extent=extent)
        self.ax.set_title("Generated Orthomosaic")
        self.ax.set_aspect('equal')
        self.figure.tight_layout()
        self.canvas.draw()

    def display_graph(self, nodes, edges):
        # We don't clear, we draw ON TOP of the map
        res = self.resolution.get()
        bx = self.current_bounds['min_x']
        by = self.current_bounds['max_y']
        
        # Convert pixel to meter for display
        def p2m(px, py):
            return (px / res) + bx, by - (py / res)
            
        for e in edges:
            n1 = next(n for n in nodes if n['id'] == e['from_id'])
            n2 = next(n for n in nodes if n['id'] == e['to_id'])
            m1x, m1y = p2m(n1['x'], n1['y'])
            m2x, m2y = p2m(n2['x'], n2['y'])
            self.ax.plot([m1x, m2x], [m1y, m2y], 'r-', linewidth=1.5, alpha=0.7)
            
        for n in nodes:
            mx, my = p2m(n['x'], n['y'])
            color = 'lime' if n.get('type') == 'entrance_exit' else 'blue'
            self.ax.plot(mx, my, marker='o', markersize=3, color=color, alpha=0.8)
            
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1024x768")
    app = UnifiedPipelineGUI(root)
    root.mainloop()
