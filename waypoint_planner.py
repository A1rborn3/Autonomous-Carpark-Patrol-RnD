import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

class WaypointPlannerApp:
    def __init__(self, root):
        self.root = root
        
        self.ortho_path = None
        self.obs_occ_path = None
        self.park_occ_path = None
        self.map_metadata = None
        
        self.waypoints = [] # List of [x, y] in meters
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top Frame for Buttons
        btn_frame = tk.Frame(self.root, padx=5, pady=5)
        btn_frame.pack(fill=tk.X, side=tk.TOP)
        
        btn_load = tk.Button(btn_frame, text="Load Maps Manually", command=self.load_maps_dialog)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_undo = tk.Button(btn_frame, text="Undo Last Waypoint", command=self.undo_waypoint)
        btn_undo.pack(side=tk.LEFT, padx=5)
        
        btn_clear = tk.Button(btn_frame, text="Clear All Waypoints", command=self.clear_waypoints)
        btn_clear.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(btn_frame, text="Save Waypoints", command=self.save_waypoints)
        btn_save.pack(side=tk.RIGHT, padx=5)
        
        # Instructions
        self.status_var = tk.StringVar()
        self.status_var.set("Load maps to begin. Click to place waypoints.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Plot Frame
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Event binding
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        
    def load_maps_dialog(self):
        file_path = filedialog.askopenfilename(
            title="Select Orthomosaic Image",
            filetypes=[("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_maps(file_path)
            
    def load_maps(self, ortho_path):
        if not os.path.exists(ortho_path):
            return
            
        self.ortho_path = ortho_path
        
        base_dir = os.path.dirname(ortho_path)
        self.obs_occ_path = os.path.join(base_dir, "obstacle_occupancy.png")
        self.park_occ_path = os.path.join(base_dir, "parking_occupancy.png")
        
        # Support fallback if old occupancy name exists
        if not os.path.exists(self.obs_occ_path) and os.path.exists(os.path.join(base_dir, "occupancy.png")):
            self.obs_occ_path = os.path.join(base_dir, "occupancy.png")
            
        # Try to load metadata
        meta_path = os.path.join(os.path.dirname(ortho_path), "map_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.map_metadata = json.load(f)
        else:
            self.map_metadata = None
            
        self.draw_plot()
        self.status_var.set(f"Loaded maps from {os.path.dirname(ortho_path)}")
        
    def draw_plot(self):
        self.ax.clear()
        
        if not self.ortho_path or not self.map_metadata:
            self.canvas.draw()
            return
            
        bounds = self.map_metadata["bounds"]
        extent = [bounds['min_x'], bounds['max_x'], bounds['min_y'], bounds['max_y']]
        
        # Load and plot Orthomosaic
        ortho_bgr = cv2.imread(self.ortho_path)
        if ortho_bgr is not None:
            ortho_rgb = cv2.cvtColor(ortho_bgr, cv2.COLOR_BGR2RGB)
            self.ax.imshow(ortho_rgb, extent=extent)
            
        # Load and plot Occupancy Overlay
        if os.path.exists(self.obs_occ_path):
            obs_occ = cv2.imread(self.obs_occ_path, cv2.IMREAD_GRAYSCALE)
            if obs_occ is not None:
                # Create an RGBA image for the overlay
                rgba = np.zeros((obs_occ.shape[0], obs_occ.shape[1], 4), dtype=np.float32)
                
                # Base Green for walkable (255)
                mask_walk = (obs_occ > 200)
                rgba[mask_walk, 1] = 1.0 # G
                rgba[mask_walk, 3] = 0.4 # Alpha
                
                # Layer Blue for parking spaces
                if os.path.exists(self.park_occ_path):
                    park_occ = cv2.imread(self.park_occ_path, cv2.IMREAD_GRAYSCALE)
                    if park_occ is not None:
                        # Parking is black (0) in parking_occupancy
                        mask_park = (park_occ < 50)
                        rgba[mask_park, :] = [0.0, 0.0, 1.0, 0.4] # B, Alpha overrides green
                
                # Layer Red for obstacles (0)
                mask_obs = (obs_occ < 50)
                rgba[mask_obs, :] = [1.0, 0.0, 0.0, 0.4] # R, Alpha overrides everything else
                
                self.ax.imshow(rgba, extent=extent)
                
        self.ax.set_title("Waypoint Planner")
        self.ax.set_xlabel("X (meters)")
        self.ax.set_ylabel("Y (meters)")
        self.ax.set_aspect('equal', adjustable='box')
        
        # Draw waypoints
        self.redraw_waypoints()
        
    def redraw_waypoints(self):
        # Remove old waypoint artists
        for artist in list(self.ax.lines) + list(self.ax.collections):
            artist.remove()
            
        if self.waypoints:
            xs = [wp[0] for wp in self.waypoints]
            ys = [wp[1] for wp in self.waypoints]
            
            # Draw lines
            self.ax.plot(xs, ys, 'b-', linewidth=2, marker='o', markersize=6, markerfacecolor='yellow')
            
            # Mark Start
            self.ax.plot(xs[0], ys[0], 'g*', markersize=12)
            # Mark End
            if len(self.waypoints) > 1:
                self.ax.plot(xs[-1], ys[-1], 'rX', markersize=10)
                
        self.canvas.draw()
        
    def on_click(self, event):
        # Ignore clicks outside the axes or if using a toolbar tool (zoom/pan)
        if event.inaxes != self.ax or self.toolbar.mode != '':
            return
            
        if event.button == 1: # Left click
            x, y = event.xdata, event.ydata
            self.waypoints.append([round(float(x), 4), round(float(y), 4)])
            self.redraw_waypoints()
            self.status_var.set(f"Added waypoint {len(self.waypoints)} at ({x:.2f}, {y:.2f})")
            
    def undo_waypoint(self):
        if self.waypoints:
            self.waypoints.pop()
            self.redraw_waypoints()
            self.status_var.set(f"Removed last waypoint. {len(self.waypoints)} remaining.")
            
    def clear_waypoints(self):
        if messagebox.askyesno("Confirm", "Clear all waypoints?"):
            self.waypoints = []
            self.redraw_waypoints()
            self.status_var.set("Waypoints cleared.")
            
    def save_waypoints(self):
        if not self.waypoints:
            messagebox.showwarning("Warning", "No waypoints to save.")
            return
            
        if not self.ortho_path:
            return
            
        base_dir = os.path.dirname(self.ortho_path)
        default_name = "waypoints.json"
        
        save_path = filedialog.asksaveasfilename(
            initialdir=base_dir,
            initialfile=default_name,
            title="Save Waypoints",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(self.waypoints, f, indent=4)
                messagebox.showinfo("Success", f"Saved {len(self.waypoints)} waypoints.")
                self.status_var.set(f"Saved waypoints to {os.path.basename(save_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save waypoints: {e}")
