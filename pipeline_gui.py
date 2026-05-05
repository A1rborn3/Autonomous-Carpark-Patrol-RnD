import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# Import our pipeline modules
from point_cloud_processor import PointCloudProcessor
from map_generator import MapGenerator

class PipelineGUI:
    def __init__(self, root, on_map_generated=None):
        self.root = root
        self.on_map_generated = on_map_generated
        if hasattr(self.root, 'title'):
            self.root.title("PLY to 2D Map Pipeline")
        if hasattr(self.root, 'geometry'):
            self.root.geometry("1024x768")
        
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar(value=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"))
        self.voxel_size = tk.DoubleVar(value=0.02)
        self.obs_height = tk.DoubleVar(value=0.1)
        self.max_height = tk.DoubleVar(value=2.0)
        self.resolution = tk.DoubleVar(value=70)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top Frame for controls
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # File selection
        tk.Label(control_frame, text="PLY File:").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(control_frame, textvariable=self.input_file, width=60).grid(row=0, column=1, padx=5)
        tk.Button(control_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)
        
        # Parameters
        params_frame = tk.LabelFrame(control_frame, text="Parameters", padx=5, pady=5)
        params_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=10)
        
        tk.Label(params_frame, text="Voxel Size (m):").grid(row=0, column=0, sticky=tk.W)
        tk.Entry(params_frame, textvariable=self.voxel_size, width=10).grid(row=0, column=1)
        
        tk.Label(params_frame, text="Min Obs Height (m):").grid(row=0, column=2, sticky=tk.W, padx=(10,0))
        tk.Entry(params_frame, textvariable=self.obs_height, width=10).grid(row=0, column=3)
        
        tk.Label(params_frame, text="Max Height (m):").grid(row=0, column=4, sticky=tk.W, padx=(10,0))
        tk.Entry(params_frame, textvariable=self.max_height, width=10).grid(row=0, column=5)
        
        tk.Label(params_frame, text="Resolution (px/m):").grid(row=0, column=6, sticky=tk.W, padx=(10,0))
        tk.Entry(params_frame, textvariable=self.resolution, width=10).grid(row=0, column=7)
        
        # Run Button
        self.btn_run = tk.Button(control_frame, text="Generate Map", command=self.run_pipeline, bg="green", fg="black", font=("Arial", 12, "bold"))
        self.btn_run.grid(row=2, column=0, columnspan=3, pady=10)
        
        # Status Label
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        tk.Label(control_frame, textvariable=self.status_var, fg="blue").grid(row=3, column=0, columnspan=3)
        
        # Plot Frame
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        
        # Add toolbar for zooming/panning
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select PLY File",
            filetypes=[("PLY Files", "*.ply"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_file.set(file_path)
            
    def run_pipeline(self):
        input_path = self.input_file.get()
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid PLY file.")
            return
            
        self.btn_run.config(state=tk.DISABLED)
        self.status_var.set("Processing... This may take a while.")
        self.ax.clear()
        self.canvas.draw()
        
        # Run in a separate thread so GUI doesn't freeze
        threading.Thread(target=self.process_file, args=(input_path,), daemon=True).start()
        
    def process_file(self, input_path):
        try:
            filename = os.path.basename(input_path)
            file_basename = os.path.splitext(filename)[0]
            
            output_dir = os.path.join(self.output_dir.get(), file_basename)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            processor = PointCloudProcessor(
                voxel_size=self.voxel_size.get(),
                obstacle_height=self.obs_height.get(),
                max_height=self.max_height.get()
            )
            
            self.root.after(0, lambda: self.status_var.set("Loading PLY..."))
            pcd = processor.load_ply(input_path)
            
            self.root.after(0, lambda: self.status_var.set("Cleaning and Downsampling..."))
            pcd = processor.clean_and_downsample(pcd)
            
            self.root.after(0, lambda: self.status_var.set("Segmenting Ground..."))
            ground_inliers, non_ground, plane_model = processor.segment_ground(pcd)
            
            self.root.after(0, lambda: self.status_var.set("Aligning to Ground..."))
            pcd = processor.align_to_ground(pcd, plane_model)
            
            self.root.after(0, lambda: self.status_var.set("Extracting Obstacles..."))
            ground_points, obstacle_points = processor.extract_obstacles(pcd)
            
            self.root.after(0, lambda: self.status_var.set("Generating 2D Map..."))
            map_gen = MapGenerator(resolution=self.resolution.get())
            ortho, occ, bounds = map_gen.generate_maps(ground_points, obstacle_points)
            
            ortho_path = os.path.join(output_dir, "orthomosaic.png")
            occ_path = os.path.join(output_dir, "occupancy.png")
            
            self.root.after(0, lambda: self.status_var.set("Saving Maps..."))
            map_gen.save_maps(ortho, occ, bounds, ortho_path, occ_path)
            
            # Display map
            self.root.after(0, lambda: self.display_map(ortho, bounds))
            self.root.after(0, lambda: self.status_var.set(f"Finished. Saved to {output_dir}"))
            
            if self.on_map_generated:
                self.root.after(0, lambda: self.on_map_generated(ortho_path))
            
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Processing Error", str(e)))
            self.root.after(0, lambda: self.status_var.set("Error occurred."))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state=tk.NORMAL))
            
    def display_map(self, orthomosaic_bgr, bounds):
        import cv2
        orthomosaic_rgb = cv2.cvtColor(orthomosaic_bgr, cv2.COLOR_BGR2RGB)
        
        self.ax.clear()
        
        extent = [bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']]
        
        self.ax.imshow(orthomosaic_rgb, extent=extent)
        self.ax.set_title("Coordinate Graph of Orthomosaic Map")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        
        self.ax.set_aspect('equal', adjustable='box')
        
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()
