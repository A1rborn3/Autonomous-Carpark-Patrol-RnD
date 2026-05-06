import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import cv2
import numpy as np

class ParkingAnnotatorApp:
    def __init__(self, root, on_annotations_saved=None):
        self.root = root
        self.on_annotations_saved = on_annotations_saved
        if hasattr(self.root, 'title'):
            self.root.title("Parking Space Annotator")
        
        # State variables
        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.map_metadata = None
        
        self.polygons = [] # List of completed polygons. Each is a dict: {'id': int, 'points': [(x,y), ...]}
        self.current_polygon = [] # List of (x,y) points for the polygon currently being drawn
        self.poly_id_counter = 1
        self.zoom_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        
        # UI Setup
        self.setup_ui()
        
    def setup_ui(self):
        # Top Frame for Buttons
        btn_frame = tk.Frame(self.root, padx=5, pady=5)
        btn_frame.pack(fill=tk.X, side=tk.TOP)
        
        btn_load = tk.Button(btn_frame, text="Load Map Image", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=5)
        
        btn_undo = tk.Button(btn_frame, text="Undo Last Point", command=self.undo_point)
        btn_undo.pack(side=tk.LEFT, padx=5)
        
        btn_clear_current = tk.Button(btn_frame, text="Clear Current Polygon", command=self.clear_current)
        btn_clear_current.pack(side=tk.LEFT, padx=5)

        btn_finish_poly = tk.Button(btn_frame, text="Finish Polygon (or Right-Click)", command=self.finish_polygon)
        btn_finish_poly.pack(side=tk.LEFT, padx=5)
        
        btn_clear_all = tk.Button(btn_frame, text="Clear ALL Polygons", command=self.clear_all)
        btn_clear_all.pack(side=tk.LEFT, padx=5)
        
        btn_zoom_in = tk.Button(btn_frame, text="Zoom In", command=self.zoom_in)
        btn_zoom_in.pack(side=tk.LEFT, padx=5)
        
        btn_zoom_out = tk.Button(btn_frame, text="Zoom Out", command=self.zoom_out)
        btn_zoom_out.pack(side=tk.LEFT, padx=5)
        
        btn_save = tk.Button(btn_frame, text="Save Annotations", command=self.save_annotations)
        btn_save.pack(side=tk.RIGHT, padx=5)
        
        # Canvas Frame with Scrollbars
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="gray", cursor="crosshair")
        
        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Mouse Bindings
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-2>", self.on_right_click) # Mac right click / middle click
        self.canvas.bind("<Button-3>", self.on_right_click) # Windows/Linux right click
        
        # Mouse wheel for zooming
        self.canvas.bind("<MouseWheel>", self.on_mousewheel) # Windows / Mac
        self.canvas.bind("<Button-4>", self.on_mousewheel) # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel) # Linux scroll down
        
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        
        # Instructions
        self.status_var = tk.StringVar()
        self.status_var.set("Load an image to begin. Left-click to add points. Right-click to complete a polygon.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def load_image(self, file_path=None):
        if not file_path:
            file_path = filedialog.askopenfilename(
                title="Select Map Image",
                filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
            )
        if not file_path:
            return
            
        self.image_path = file_path
        self.original_image = Image.open(self.image_path)
        self.zoom_factor = 1.0 # Reset zoom
        
        # Try to load existing annotations if they exist
        self.polygons = []
        self.current_polygon = []
        self.poly_id_counter = 1
        
        base_dir = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        json_path = os.path.join(base_dir, f"{base_name}_parking_spaces.json")
        
        # Load map metadata if available
        meta_path = os.path.join(base_dir, "map_metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    self.map_metadata = json.load(f)
            except Exception as e:
                print(f"Failed to load map metadata: {e}")
        else:
            self.map_metadata = None
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    loaded_polygons = data.get("parking_spaces", [])
                    is_meters = data.get("coordinate_system") == "meters"
                    
                    self.polygons = []
                    for poly in loaded_polygons:
                        new_poly = {"id": poly["id"], "points": []}
                        for pt in poly["points"]:
                            if is_meters and self.map_metadata:
                                res = self.map_metadata["resolution"]
                                bounds = self.map_metadata["bounds"]
                                x_px = int(round((pt[0] - bounds["min_x"]) * res))
                                y_px = int(round((bounds["max_y"] - pt[1]) * res))
                                new_poly["points"].append((x_px, y_px))
                            else:
                                new_poly["points"].append((int(pt[0]), int(pt[1])))
                        self.polygons.append(new_poly)
                        
                    if self.polygons:
                        self.poly_id_counter = max([p.get('id', 0) for p in self.polygons]) + 1
                self.status_var.set(f"Loaded existing annotations from {json_path}")
            except Exception as e:
                self.status_var.set(f"Failed to load existing annotations: {e}")
        else:
            self.status_var.set(f"Loaded {os.path.basename(file_path)}. Start drawing!")
            
        self.update_image_display()

    def update_image_display(self):
        if not self.original_image:
            return
        
        new_width = int(self.original_image.width * self.zoom_factor)
        new_height = int(self.original_image.height * self.zoom_factor)
        
        # Resize image
        resized = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        self.redraw()

    def zoom_in(self):
        self.zoom_factor *= 1.2
        self.update_image_display()

    def zoom_out(self):
        self.zoom_factor /= 1.2
        self.update_image_display()

    def on_mousewheel(self, event):
        # Determine scroll direction
        # Linux uses num=4/5, Mac/Windows use delta
        if hasattr(event, "num") and event.num == 4:
            self.zoom_in()
        elif hasattr(event, "num") and event.num == 5:
            self.zoom_out()
        elif hasattr(event, "delta"):
            if event.delta > 0:
                self.zoom_in()
            elif event.delta < 0:
                self.zoom_out()

    def on_left_click(self, event):
        if not self.image_path:
            return
            
        # Get actual canvas coordinates taking scroll into account
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Un-scale the coordinates to original image size
        orig_x = (x - self.img_offset_x) / self.zoom_factor
        orig_y = (y - self.img_offset_y) / self.zoom_factor
        
        # Ensure points are within image bounds
        if 0 <= orig_x <= self.original_image.width and 0 <= orig_y <= self.original_image.height:
            self.current_polygon.append((int(orig_x), int(orig_y)))
            self.redraw()

    def on_right_click(self, event):
        self.finish_polygon()

    def finish_polygon(self):
        if len(self.current_polygon) > 2:
            self.polygons.append({
                "id": self.poly_id_counter,
                "points": list(self.current_polygon)
            })
            self.poly_id_counter += 1
            self.current_polygon = []
            self.status_var.set(f"Polygon {self.poly_id_counter-1} saved. {len(self.polygons)} total.")
            self.redraw()
            self.update_live_occupancy()
        elif len(self.current_polygon) > 0:
            messagebox.showwarning("Warning", "A polygon must have at least 3 points.")

    def undo_point(self):
        if self.current_polygon:
            self.current_polygon.pop()
            self.redraw()
        elif self.polygons:
            # Optionally undo last completed polygon if current is empty
            self.polygons.pop()
            self.redraw()
            self.status_var.set(f"Removed last polygon. {len(self.polygons)} total.")
            self.update_live_occupancy()

    def clear_current(self):
        self.current_polygon = []
        self.redraw()

    def clear_all(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all polygons?"):
            self.polygons = []
            self.current_polygon = []
            self.poly_id_counter = 1
            self.redraw()
            self.status_var.set("All polygons cleared.")
            self.update_live_occupancy()

    def redraw(self):
        self.canvas.delete("all")
        
        if self.tk_image:
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            
            if cw <= 1:
                cw = self.canvas.winfo_reqwidth()
            if ch <= 1:
                ch = self.canvas.winfo_reqheight()
                
            iw = self.tk_image.width()
            ih = self.tk_image.height()
            
            self.img_offset_x = max(0, (cw - iw) // 2)
            self.img_offset_y = max(0, (ch - ih) // 2)
            
            self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor=tk.NW, image=self.tk_image)
            self.canvas.config(scrollregion=(0, 0, max(cw, iw), max(ch, ih)))
            
        # Draw completed polygons
        for poly in self.polygons:
            points = poly["points"]
            if len(points) > 2:
                # Scale points
                scaled_points = [(pt[0]*self.zoom_factor + self.img_offset_x, pt[1]*self.zoom_factor + self.img_offset_y) for pt in points]
                
                # Draw filled polygon with outline
                flat_points = [coord for pt in scaled_points for coord in pt]
                self.canvas.create_polygon(flat_points, outline='blue', fill='', width=2, tags="poly")
                
                # Draw ID in the center
                cx = sum([p[0] for p in scaled_points]) / len(scaled_points)
                cy = sum([p[1] for p in scaled_points]) / len(scaled_points)
                self.canvas.create_text(cx, cy, text=str(poly["id"]), fill="red", font=("Arial", 12, "bold"))

        # Draw current polygon points and lines
        if self.current_polygon:
            scaled_current = [(pt[0]*self.zoom_factor + self.img_offset_x, pt[1]*self.zoom_factor + self.img_offset_y) for pt in self.current_polygon]
            for i, pt in enumerate(scaled_current):
                r = 3
                self.canvas.create_oval(pt[0]-r, pt[1]-r, pt[0]+r, pt[1]+r, fill='red', outline='red')
                
                if i > 0:
                    prev_pt = scaled_current[i-1]
                    self.canvas.create_line(prev_pt[0], prev_pt[1], pt[0], pt[1], fill='red', width=2)
            
            # Draw line back to start to close it visually (dashed)
            if len(scaled_current) > 2:
                first_pt = scaled_current[0]
                last_pt = scaled_current[-1]
                self.canvas.create_line(last_pt[0], last_pt[1], first_pt[0], first_pt[1], fill='red', width=2, dash=(4, 4))

    def save_annotations(self):
        if not self.image_path:
            messagebox.showinfo("Info", "No image loaded.")
            return
            
        if not self.polygons:
            if not messagebox.askyesno("Confirm", "No polygons drawn. Save empty file?"):
                return

        base_dir = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        default_name = f"{base_name}_parking_spaces.json"
        
        save_path = filedialog.asksaveasfilename(
            initialdir=base_dir,
            initialfile=default_name,
            title="Save Annotations",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")]
        )
        
        if save_path:
            converted_polygons = []
            for poly in self.polygons:
                new_poly = {"id": poly["id"], "points": []}
                for pt in poly["points"]:
                    x, y = pt[0], pt[1]
                    if self.map_metadata:
                        res = self.map_metadata["resolution"]
                        bounds = self.map_metadata["bounds"]
                        x_m = (x / res) + bounds["min_x"]
                        y_m = bounds["max_y"] - (y / res)
                        new_poly["points"].append([round(x_m, 6), round(y_m, 6)])
                    else:
                        new_poly["points"].append([x, y])
                converted_polygons.append(new_poly)

            data = {
                "image_file": os.path.basename(self.image_path),
                "coordinate_system": "meters" if self.map_metadata else "pixels",
                "parking_spaces": converted_polygons
            }
            try:
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=4)
                        
                messagebox.showinfo("Success", f"Saved {len(self.polygons)} parking spaces to {os.path.basename(save_path)}")
                self.status_var.set(f"Saved to {os.path.basename(save_path)}.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def update_live_occupancy(self):
        if not self.image_path or not self.original_image:
            return
            
        base_dir = os.path.dirname(self.image_path)
        try:
            w, h = self.original_image.size
            # Create white image (255) for walkable, black (0) for parking spaces
            parking_occ = np.full((h, w), 255, dtype=np.uint8)
            
            expansion_m = 0.75
            if self.map_metadata and "resolution" in self.map_metadata:
                res = self.map_metadata["resolution"]
                expansion_px = int(round(expansion_m * res))
            else:
                expansion_px = int(round(expansion_m * 20)) # fallback
                
            thickness = max(1, expansion_px * 2)
            
            for poly in self.polygons:
                pts = np.array(poly["points"], np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Fill polygon with black
                cv2.fillPoly(parking_occ, [pts], 0)
                # Expand polygon by drawing thick outline
                cv2.polylines(parking_occ, [pts], isClosed=True, color=0, thickness=thickness)
                
            park_occ_path = os.path.join(base_dir, "parking_occupancy.png")
            cv2.imwrite(park_occ_path, parking_occ)
            
            if self.on_annotations_saved:
                self.on_annotations_saved(self.image_path)
        except Exception as img_e:
            print(f"Failed to create live parking occupancy image: {img_e}")

if __name__ == "__main__":
    root = tk.Tk()
    
    # Set a decent default window size
    root.geometry("1024x768")
    
    app = ParkingAnnotatorApp(root)
    root.mainloop()
