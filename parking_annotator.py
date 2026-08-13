import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
from inference import run_inference
from inference_config import MODEL_PATH, CONF, TILE_SIZE, PIXEL_THRESHOLD
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
        
        # Inference variables
        self.model = YOLO(MODEL_PATH)
        self.conf = CONF
        self.tile_size = TILE_SIZE
        self.pixel_threshold = PIXEL_THRESHOLD
        
        # State variables
        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.map_metadata = None
        
        self.polygons = [] # List of completed polygons. Each is a dict: {'id': int, 'points': [(x,y), ...]}
        self.current_polygon = [] # List of (x,y) points for the polygon currently being drawn
        self.selected_polygon_id = None
        self.poly_id_counter = 1
        self.zoom_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.current_poly_type = "parking_space" # "parking_space" or "entrance_exit"
        
        self.carpark_types = {
            "Regular": {"tag": "regular", "canvas_colour": "#0fdb16"},
            "Handicap": {"tag": "handicap", "canvas_colour": "blue"},
            "60 Mins Max": {"tag": "60_mins_max", "canvas_colour": "red"}
        }
        # UI Setup

        self.setup_ui()
        
    def setup_ui(self):
        # Left Panel for Controls / Buttons (scrollable so all controls stay
        # reachable even when the window is short or the control list grows)
        sidebar_outer = tk.Frame(self.root, width=260, relief=tk.RAISED, borderwidth=1)
        sidebar_outer.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_outer.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_outer, width=260, highlightthickness=0)
        sidebar_scrollbar = tk.Scrollbar(sidebar_outer, orient=tk.VERTICAL, command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(sidebar_canvas, padx=10, pady=10)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _on_sidebar_configure(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        sidebar.bind("<Configure>", _on_sidebar_configure)

        def _on_sidebar_canvas_configure(event):
            sidebar_canvas.itemconfig(sidebar_window, width=event.width)
        sidebar_canvas.bind("<Configure>", _on_sidebar_canvas_configure)

        def _on_sidebar_mousewheel(event):
            if hasattr(event, "num") and event.num == 4:
                sidebar_canvas.yview_scroll(-1, "units")
            elif hasattr(event, "num") and event.num == 5:
                sidebar_canvas.yview_scroll(1, "units")
            elif hasattr(event, "delta"):
                sidebar_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        def _bind_sidebar_scroll(event):
            sidebar_canvas.bind_all("<MouseWheel>", _on_sidebar_mousewheel)
            sidebar_canvas.bind_all("<Button-4>", _on_sidebar_mousewheel)
            sidebar_canvas.bind_all("<Button-5>", _on_sidebar_mousewheel)

        def _unbind_sidebar_scroll(event):
            sidebar_canvas.unbind_all("<MouseWheel>")
            sidebar_canvas.unbind_all("<Button-4>")
            sidebar_canvas.unbind_all("<Button-5>")

        sidebar_canvas.bind("<Enter>", _bind_sidebar_scroll)
        sidebar_canvas.bind("<Leave>", _unbind_sidebar_scroll)

        tk.Label(sidebar, text="Annotator Controls", font=("Arial", 11, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # 1. File Actions
        file_frame = tk.LabelFrame(sidebar, text="Map File", padx=5, pady=5)
        file_frame.pack(fill=tk.X, pady=(0, 10))

        btn_load = tk.Button(file_frame, text="Load Map Image", command=self.load_image)
        btn_load.pack(fill=tk.X, pady=2)

        # 2. Tool Selection
        self.type_frame = tk.LabelFrame(sidebar, text="Current Tool", padx=5, pady=5)
        self.type_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(self.type_frame, text="Parking Spot Type:").pack(anchor=tk.W, pady=(0, 2))

        self.style_var = tk.StringVar(value="Regular")
        self.opt_type_park = tk.OptionMenu(
            self.type_frame,
            self.style_var,
            *self.carpark_types.keys(),
            command=lambda _: self.set_tool("parking_space")
        )
        self.opt_type_park.config(bg="lightblue")
        self.opt_type_park.pack(fill=tk.X, pady=(0, 5))

        self.btn_type_ent = tk.Button(self.type_frame, text="Entrance/Exit", command=lambda: self.set_tool("entrance_exit"))
        self.btn_type_ent.pack(fill=tk.X, pady=2)

        # 3. Polygon Editing and Listing
        edit_frame = tk.LabelFrame(sidebar, text="Polygon Editing", padx=5, pady=5)
        edit_frame.pack(fill=tk.X, pady=(0, 10))

        btn_undo = tk.Button(edit_frame, text="Undo Last Point", command=self.undo_point)
        btn_undo.pack(fill=tk.X, pady=2)

        btn_finish_poly = tk.Button(edit_frame, text="Finish Polygon", command=self.finish_polygon)
        btn_finish_poly.pack(fill=tk.X, pady=2)

        btn_clear_current = tk.Button(edit_frame, text="Clear Current Polygon", command=self.clear_current)
        btn_clear_current.pack(fill=tk.X, pady=2)

        btn_clear_all = tk.Button(edit_frame, text="Clear ALL Polygons", command=self.clear_all)
        btn_clear_all.pack(fill=tk.X, pady=2)

        list_frame = tk.LabelFrame(sidebar, text="Annotations (select to remove)", padx=5, pady=5)
        list_frame.pack(fill=tk.X, pady=(0, 10))

        list_container = tk.Frame(list_frame)
        list_container.pack(fill=tk.BOTH, expand=True)

        list_scrollbar = tk.Scrollbar(list_container, orient=tk.VERTICAL)
        list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.polygon_listbox = tk.Listbox(
            list_container,
            height=6,
            exportselection=False,
            yscrollcommand=list_scrollbar.set
        )

        self.polygon_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scrollbar.config(command=self.polygon_listbox.yview)

        self.polygon_listbox.bind("<<ListboxSelect>>", self.on_polygon_select)

        btn_delete_selected = tk.Button(list_frame, text="Delete Selected", command=self.delete_selected_polygon, bg="red", fg="white")
        btn_delete_selected.pack(fill=tk.X, pady=(5, 0))

        # 4. View Controls
        view_frame = tk.LabelFrame(sidebar, text="View Controls", padx=5, pady=5)
        view_frame.pack(fill=tk.X, pady=(0, 10))

        btn_zoom_in = tk.Button(view_frame, text="Zoom In", command=self.zoom_in)
        btn_zoom_in.pack(fill=tk.X, pady=2)

        btn_zoom_out = tk.Button(view_frame, text="Zoom Out", command=self.zoom_out)
        btn_zoom_out.pack(fill=tk.X, pady=2)

        # 5. Annotation Actions
        actions_frame = tk.LabelFrame(sidebar, text="Actions & Save", padx=5, pady=5)
        actions_frame.pack(fill=tk.X, pady=(0, 10))

        btn_auto = tk.Button(actions_frame, text="Automatic Annotations", command=self.automatic_annotations, bg="orange", fg="black")
        btn_auto.pack(fill=tk.X, pady=2)

        btn_save = tk.Button(actions_frame, text="Save Annotations", command=self.save_annotations, bg="green", fg="black")
        btn_save.pack(fill=tk.X, pady=2)

        
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
        self.selected_polygon_id = None
        
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
                        new_poly = {"id": poly["id"], "points": [], "type": poly.get("type", "parking_space")}
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
        self.refresh_polygon_list()

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
                "type": self.current_poly_type,
                "points": list(self.current_polygon)
            })
            self.poly_id_counter += 1
            self.current_polygon = []
            self.status_var.set(f"{self.current_poly_type} {self.poly_id_counter-1} saved. {len(self.polygons)} total.")
            self.redraw()
            self.refresh_polygon_list()
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
            self.refresh_polygon_list()
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
            self.selected_polygon_id = None
            self.redraw()
            self.refresh_polygon_list()
            self.status_var.set("All polygons cleared.")
            self.update_live_occupancy()

    def refresh_polygon_list(self):
        """Repopulate the sidebar listbox with the current polygons (id + type)."""
        if not hasattr(self, "polygon_listbox"):
            return

        self.polygon_listbox.delete(0, tk.END)
        for poly in self.polygons:
            label = f"ID {poly['id']}  -  {poly.get('type', 'parking_space')}  ({len(poly['points'])} pts)"
            self.polygon_listbox.insert(tk.END, label)

        if self.selected_polygon_id is not None:
            for idx, poly in enumerate(self.polygons):
                if poly["id"] == self.selected_polygon_id:
                    self.polygon_listbox.selection_set(idx)
                    break
            else:
                self.selected_polygon_id = None

    def on_polygon_select(self, event):
        selection = self.polygon_listbox.curselection()
        if not selection:
            self.selected_polygon_id = None
        else:
            idx = selection[0]
            if 0 <= idx < len(self.polygons):
                self.selected_polygon_id = self.polygons[idx]["id"]
        self.redraw()

    def delete_selected_polygon(self):
        selection = self.polygon_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Select a polygon from the list first.")
            return

        idx = selection[0]
        if not (0 <= idx < len(self.polygons)):
            return

        poly = self.polygons[idx]
        if not messagebox.askyesno("Confirm", f"Delete polygon ID {poly['id']} ({poly.get('type', 'parking_space')})?"):
            return

        del self.polygons[idx]
        self.selected_polygon_id = None
        self.refresh_polygon_list()
        self.redraw()
        self.status_var.set(f"Deleted polygon ID {poly['id']}. {len(self.polygons)} total.")
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
                poly_type = poly.get('type')
                
                # Default colors
                color = 'cyan'
                fill = ''
                stipple = ''
                text_color = 'white'
                
                if poly_type == 'entrance_exit':
                    color = 'blue'
                    fill = 'blue'
                    stipple = 'gray25'
                    text_color = 'white'
                elif poly_type.startswith('parking_space_'):
                    tag = poly_type.replace('parking_space_', '')
                    # Find colour and apply from dictionary
                    for style_name, style_info in self.carpark_types.items():
                        if style_info["tag"] == tag:
                            color = style_info["canvas_colour"]
                            text_color = 'white'
                            break

                # Highlight the polygon selected in the sidebar list
                is_selected = (poly["id"] == self.selected_polygon_id)
                outline_width = 4 if is_selected else 2
                outline_color = 'yellow' if is_selected else color
                
                self.canvas.create_polygon(flat_points, outline=outline_color, fill=fill, stipple=stipple, width=outline_width, tags="poly")
                
                # Draw ID in the center
                cx = sum([p[0] for p in scaled_points]) / len(scaled_points)
                cy = sum([p[1] for p in scaled_points]) / len(scaled_points)
                self.canvas.create_text(cx, cy, text=str(poly["id"]), fill=text_color, font=("Arial", 10, "bold"))




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

    def automatic_annotations(self):
        if not self.image_path or not self.original_image:
            messagebox.showinfo("Info", "Load an image first.")
            return

        if not messagebox.askyesno("Confirm", "This will run automatic detection and ADD to your current polygons. Continue?"):
            return

        try:
            self.status_var.set("Running automatic detection...")
            self.root.update_idletasks()

            detections, mode = run_inference(
                self.model,
                self.image_path,
                conf=self.conf,
                tile_size=self.tile_size,
                pixel_threshold=self.pixel_threshold,
                base_out_dir="model-output",
            )

            class_names = self.model.names
            added = 0
            for det in detections:
                cls_name = class_names.get(det["cls"], "regular")
                points = [(int(round(p[0])), int(round(p[1]))) for p in det["poly"]]
                poly_type = "entrance_exit" if cls_name == "entrance_exit" else f"parking_space_{cls_name}"

                self.polygons.append({
                    "id": self.poly_id_counter,
                    "type": poly_type,
                    "points": points
                })
                self.poly_id_counter += 1
                added += 1

            self.current_polygon = []
            self.redraw()
            self.refresh_polygon_list()
            self.update_live_occupancy()
            self.status_var.set(f"Automatic annotation ({mode}) added {added} detections. {len(self.polygons)} total.")
            messagebox.showinfo("Done", f"Added {added} automatically detected spaces ({mode} mode).")

        except Exception as e:
            messagebox.showerror("Error", f"Automatic annotation failed:\n{e}")
            self.status_var.set("Automatic annotation failed.")

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
                new_poly = {"id": poly["id"], "type": poly.get("type", "parking_space"), "points": []}
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
                "metadata": self.map_metadata,
                "parking_spaces": converted_polygons
            }

            try:
                with open(save_path, 'w') as f:
                    json.dump(data, f, indent=4)
                        
                messagebox.showinfo("Success", f"Saved {len(self.polygons)} parking spaces to {os.path.basename(save_path)}")
                self.status_var.set(f"Saved to {os.path.basename(save_path)}.")
                
                if self.on_annotations_saved:
                    self.on_annotations_saved(self.image_path)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def set_tool(self, tool_type):
        self.current_poly_type = tool_type
        if tool_type == "parking_space":
            selected_style = self.style_var.get()
            tag = self.carpark_types[selected_style]["tag"]
            
            self.current_poly_type = f"parking_space_{tag}"
            
            self.btn_type_ent.config(relief=tk.RAISED, bg="SystemButtonFace")
            self.status_var.set(f"Tool: Parking Spot ({selected_style}). Left-click to draw.")
        else:
            self.current_poly_type = "entrance_exit"
            self.btn_type_ent.config(relief=tk.SUNKEN, bg="blue", fg="white")
            self.status_var.set("Tool: Entrance/Exit. Left-click to draw. These will be marked as Blue (255,0,0) in occupancy.")


    def update_live_occupancy(self):
        if not self.image_path or not self.original_image:
            return
            
        base_dir = os.path.dirname(self.image_path)
        try:
            w, h = self.original_image.size
            # Create white BGR image (255, 255, 255) for walkable
            # We use BGR to support blue entrances
            parking_occ = np.full((h, w, 3), 255, dtype=np.uint8)
            
            expansion_m = 0.5
            res = self.map_metadata["resolution"] if (self.map_metadata and "resolution" in self.map_metadata) else 20
            expansion_px = int(round(expansion_m * res))
            thickness = max(1, expansion_px * 2)
            
            for poly in self.polygons:
                pts = np.array(poly["points"], np.int32).reshape((-1, 1, 2))
                is_entrance = poly.get("type") == "entrance_exit"
                
                # BGR: Pure Blue is (255, 0, 0), Black is (0, 0, 0)
                color = (255, 0, 0) if is_entrance else (0, 0, 0)
                
                # Fill polygon


                cv2.fillPoly(parking_occ, [pts], color)
                # Expand polygon by drawing thick outline
                cv2.polylines(parking_occ, [pts], isClosed=True, color=color, thickness=thickness)
                
            park_occ_path = os.path.join(base_dir, "parking_occupancy.png")
            cv2.imwrite(park_occ_path, parking_occ)
            
        except Exception as img_e:
            print(f"Failed to create live parking occupancy image: {img_e}")


if __name__ == "__main__":
    root = tk.Tk()
    
    # Set a decent default window size
    root.geometry("1024x768")
    
    app = ParkingAnnotatorApp(root)
    root.mainloop()