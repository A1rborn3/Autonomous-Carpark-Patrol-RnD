import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import os
import cv2
import numpy as np


class RobotLinePlotterApp:
    def __init__(self, root, on_extract_graph=None, on_route_saved=None, on_export_manual_route=None):
        self.root = root
        self.on_extract_graph = on_extract_graph
        self.on_route_saved = on_route_saved
        self.on_export_manual_route = on_export_manual_route

        if hasattr(self.root, 'title'):
            self.root.title("Robot Line Plotter")

        # State variables
        self.image_path = None
        self.original_image = None
        self.tk_image = None
        self.map_metadata = None

        self.lines = []  # List of dicts: {'id': int, 'points': [(x, y), ...]} in pixel coordinates
        self.current_line = []  # Current polyline being drawn: [(x, y), ...]
        self.line_id_counter = 1
        self.zoom_factor = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0

        self.parking_polygons = []  # For faint preview overlay of parking spots
        self.extracted_nodes = []  # Extracted road graph nodes
        self.extracted_edges = []  # Extracted road graph edges

        self.setup_ui()

    def setup_ui(self):
        # Top Control Panel
        btn_frame = tk.Frame(self.root, padx=5, pady=5)
        btn_frame.pack(fill=tk.X, side=tk.TOP)

        btn_load = tk.Button(btn_frame, text="Load Map Image", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=3)

        btn_undo = tk.Button(btn_frame, text="Undo Point", command=self.undo_point)
        btn_undo.pack(side=tk.LEFT, padx=3)

        btn_finish = tk.Button(btn_frame, text="Finish Line (or Right-Click)", command=self.finish_line)
        btn_finish.pack(side=tk.LEFT, padx=3)

        btn_clear_curr = tk.Button(btn_frame, text="Clear Current Line", command=self.clear_current)
        btn_clear_curr.pack(side=tk.LEFT, padx=3)

        btn_clear_all = tk.Button(btn_frame, text="Clear ALL Lines", command=self.clear_all)
        btn_clear_all.pack(side=tk.LEFT, padx=3)

        btn_zoom_out = tk.Button(btn_frame, text="Zoom Out", command=self.zoom_out)
        btn_zoom_out.pack(side=tk.LEFT, padx=3)

        btn_save = tk.Button(btn_frame, text="Save Lines JSON", command=self.save_route_annotations)
        btn_save.pack(side=tk.LEFT, padx=3)

        # FINAL STEP EXPORT BUTTONS: Manual Route Waypoints vs Automated Road Graph
        self.btn_export_manual = tk.Button(
            btn_frame,
            text="Export Manual Route Waypoints",
            command=self.export_manual_route_action,
            bg="#00aaff",
            fg="black",
            font=("Arial", 10, "bold"),
            padx=8
        )
        self.btn_export_manual.pack(side=tk.RIGHT, padx=3)

        self.btn_extract_graph = tk.Button(
            btn_frame,
            text="Extract Automated Graph",
            command=self.extract_graph_action,
            bg="#00aa44",
            fg="black",
            font=("Arial", 10, "bold"),
            padx=8
        )
        self.btn_extract_graph.pack(side=tk.RIGHT, padx=3)


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
        self.canvas.bind("<Button-2>", self.on_right_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        # Mouse Wheel for Zoom
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", self.on_mousewheel)
        self.canvas.bind("<Button-5>", self.on_mousewheel)

        self.canvas.bind("<Configure>", lambda e: self.redraw())

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Step 3: Plot the line that the robot will follow. Left-click to add points. Right-click to complete line. Click 'Extract Graph Only' when ready.")
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
        self.zoom_factor = 1.0

        self.lines = []
        self.current_line = []
        self.line_id_counter = 1
        self.parking_polygons = []
        self.extracted_nodes = []
        self.extracted_edges = []

        base_dir = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]

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

        # Load parking space annotations for overlay preview
        park_json_path = os.path.join(base_dir, f"{base_name}_parking_spaces.json")
        if not os.path.exists(park_json_path):
            park_json_path = os.path.join(base_dir, "orthomosaic_parking_spaces.json")

        if os.path.exists(park_json_path):
            try:
                with open(park_json_path, 'r') as f:
                    data = json.load(f)
                    loaded_polygons = data.get("parking_spaces", [])
                    is_meters = data.get("coordinate_system") == "meters"
                    self.parking_polygons = []
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
                        self.parking_polygons.append(new_poly)
            except Exception as e:
                print(f"Failed to load parking annotations preview: {e}")

        # Load existing robot route lines if available
        route_json_path = os.path.join(base_dir, f"{base_name}_robot_route.json")
        if not os.path.exists(route_json_path):
            route_json_path = os.path.join(base_dir, "orthomosaic_robot_route.json")

        if os.path.exists(route_json_path):
            try:
                with open(route_json_path, 'r') as f:
                    data = json.load(f)
                    loaded_lines = data.get("lines", [])
                    is_meters = data.get("coordinate_system") == "meters"
                    self.lines = []
                    for line_item in loaded_lines:
                        new_line = {"id": line_item["id"], "points": []}
                        for pt in line_item["points"]:
                            if is_meters and self.map_metadata:
                                res = self.map_metadata["resolution"]
                                bounds = self.map_metadata["bounds"]
                                x_px = int(round((pt[0] - bounds["min_x"]) * res))
                                y_px = int(round((bounds["max_y"] - pt[1]) * res))
                                new_line["points"].append((x_px, y_px))
                            else:
                                new_line["points"].append((int(pt[0]), int(pt[1])))
                        self.lines.append(new_line)

                    if self.lines:
                        self.line_id_counter = max([l.get('id', 0) for l in self.lines]) + 1
                self.status_var.set(f"Loaded existing route lines from {os.path.basename(route_json_path)}")
            except Exception as e:
                self.status_var.set(f"Loaded image {os.path.basename(file_path)}. Click to draw robot path.")
        else:
            self.status_var.set(f"Loaded {os.path.basename(file_path)}. Plot the robot patrol route lines!")

        self.update_image_display()

    def update_image_display(self):
        if not self.original_image:
            return

        new_width = int(self.original_image.width * self.zoom_factor)
        new_height = int(self.original_image.height * self.zoom_factor)

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

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)

        orig_x = (x - self.img_offset_x) / self.zoom_factor
        orig_y = (y - self.img_offset_y) / self.zoom_factor

        if 0 <= orig_x <= self.original_image.width and 0 <= orig_y <= self.original_image.height:
            self.current_line.append((int(orig_x), int(orig_y)))
            self.redraw()

    def on_right_click(self, event):
        self.finish_line()

    def finish_line(self):
        if len(self.current_line) >= 2:
            self.lines.append({
                "id": self.line_id_counter,
                "points": list(self.current_line)
            })
            self.line_id_counter += 1
            self.current_line = []
            self.status_var.set(f"Robot Line {self.line_id_counter - 1} finished. Total lines: {len(self.lines)}.")
            self.redraw()
            self.save_route_annotations(quiet=True)
        elif len(self.current_line) == 1:
            messagebox.showwarning("Warning", "A line route must have at least 2 points.")

    def undo_point(self):
        if self.current_line:
            self.current_line.pop()
            self.redraw()
        elif self.lines:
            self.lines.pop()
            self.redraw()
            self.status_var.set(f"Removed last line. {len(self.lines)} lines total.")
            self.save_route_annotations(quiet=True)

    def clear_current(self):
        self.current_line = []
        self.redraw()

    def clear_all(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear all plotted lines?"):
            self.lines = []
            self.current_line = []
            self.line_id_counter = 1
            self.redraw()
            self.status_var.set("All plotted lines cleared.")
            self.save_route_annotations(quiet=True)

    def set_extracted_graph(self, nodes, edges):
        self.extracted_nodes = nodes
        self.extracted_edges = edges
        self.redraw()

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

        # 1. Draw parking space faint preview overlay
        for poly in self.parking_polygons:
            pts = poly["points"]
            if len(pts) > 2:
                scaled_pts = [(p[0] * self.zoom_factor + self.img_offset_x, p[1] * self.zoom_factor + self.img_offset_y) for p in pts]
                flat_pts = [c for p in scaled_pts for c in p]
                poly_type = poly.get('type', '')
                color = 'blue' if poly_type == 'entrance_exit' else '#00cc66'
                self.canvas.create_polygon(flat_pts, outline=color, fill='', width=1, dash=(3, 3))

        # 2. Draw completed robot lines
        for line in self.lines:
            pts = line["points"]
            if len(pts) >= 2:
                scaled_pts = [(p[0] * self.zoom_factor + self.img_offset_x, p[1] * self.zoom_factor + self.img_offset_y) for p in pts]
                for i in range(len(scaled_pts) - 1):
                    p1, p2 = scaled_pts[i], scaled_pts[i + 1]
                    # Draw thick blue line for robot path
                    self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#00ffff", width=3)
                    # Draw small node circle at vertices
                    r = 4
                    self.canvas.create_oval(p1[0] - r, p1[1] - r, p1[0] + r, p1[1] + r, fill="#0088ff", outline="white")
                # Last node circle
                last_p = scaled_pts[-1]
                r = 4
                self.canvas.create_oval(last_p[0] - r, last_p[1] - r, last_p[0] + r, last_p[1] + r, fill="#0088ff", outline="white")

                # Label line ID near first point
                first_p = scaled_pts[0]
                self.canvas.create_text(first_p[0] + 10, first_p[1] - 10, text=f"Route {line['id']}", fill="#00ffff", font=("Arial", 9, "bold"))

        # 3. Draw current active line being plotted
        if self.current_line:
            scaled_current = [(p[0] * self.zoom_factor + self.img_offset_x, p[1] * self.zoom_factor + self.img_offset_y) for p in self.current_line]
            for i, pt in enumerate(scaled_current):
                r = 4
                self.canvas.create_oval(pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r, fill="red", outline="white")
                if i > 0:
                    prev_pt = scaled_current[i - 1]
                    self.canvas.create_line(prev_pt[0], prev_pt[1], pt[0], pt[1], fill="red", width=2, dash=(6, 2))

        # 4. Draw extracted graph overlay if available
        if self.extracted_nodes and self.extracted_edges:
            for e in self.extracted_edges:
                n1 = next((n for n in self.extracted_nodes if n['id'] == e['from_id']), None)
                n2 = next((n for n in self.extracted_nodes if n['id'] == e['to_id']), None)
                if n1 and n2:
                    x1 = n1['x'] * self.zoom_factor + self.img_offset_x
                    y1 = n1['y'] * self.zoom_factor + self.img_offset_y
                    x2 = n2['x'] * self.zoom_factor + self.img_offset_x
                    y2 = n2['y'] * self.zoom_factor + self.img_offset_y
                    self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2)

            for n in self.extracted_nodes:
                nx = n['x'] * self.zoom_factor + self.img_offset_x
                ny = n['y'] * self.zoom_factor + self.img_offset_y
                r = 3
                color = "green" if n.get('type') == 'entrance_exit' else "yellow"
                self.canvas.create_oval(nx - r, ny - r, nx + r, ny + r, fill=color, outline="black")

    def save_route_annotations(self, quiet=False):
        if not self.image_path:
            if not quiet:
                messagebox.showinfo("Info", "No image loaded.")
            return

        base_dir = os.path.dirname(self.image_path)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        save_path = os.path.join(base_dir, f"{base_name}_robot_route.json")

        converted_lines = []
        for line in self.lines:
            new_line = {"id": line["id"], "points": []}
            for pt in line["points"]:
                x, y = pt[0], pt[1]
                if self.map_metadata:
                    res = self.map_metadata["resolution"]
                    bounds = self.map_metadata["bounds"]
                    x_m = (x / res) + bounds["min_x"]
                    y_m = bounds["max_y"] - (y / res)
                    new_line["points"].append([round(x_m, 6), round(y_m, 6)])
                else:
                    new_line["points"].append([x, y])
            converted_lines.append(new_line)

        data = {
            "image_file": os.path.basename(self.image_path),
            "coordinate_system": "meters" if self.map_metadata else "pixels",
            "metadata": self.map_metadata,
            "lines": converted_lines
        }

        try:
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)

            if not quiet:
                messagebox.showinfo("Success", f"Saved {len(self.lines)} robot route lines to {os.path.basename(save_path)}")
            self.status_var.set(f"Saved route lines to {os.path.basename(save_path)}.")
            if self.on_route_saved:
                self.on_route_saved(self.image_path)

        except Exception as e:
            if not quiet:
                messagebox.showerror("Error", f"Failed to save route file:\n{e}")

    def manual_lines_to_graph(self):
        """Converts self.lines into nodes and edges dicts in pixel space."""
        nodes = []
        edges = []
        node_map = {}
        node_counter = 0
        edge_counter = 0

        for line in self.lines:
            pts = line.get("points", [])
            if len(pts) < 2:
                continue
            prev_id = None
            for i, pt in enumerate(pts):
                pt_key = (int(round(pt[0])), int(round(pt[1])))
                if pt_key not in node_map:
                    node_id = f"manual_node_{node_counter}"
                    node_map[pt_key] = node_id
                    ntype = "entrance_exit" if i == 0 else ("turn" if 0 < i < len(pts) - 1 else "waypoint")
                    nodes.append({
                        "id": node_id,
                        "x": float(pt_key[0]),
                        "y": float(pt_key[1]),
                        "type": ntype
                    })
                    node_counter += 1
                else:
                    node_id = node_map[pt_key]

                if prev_id and prev_id != node_id:
                    edge_id = f"manual_edge_{edge_counter}"
                    pair = {prev_id, node_id}
                    if not any({e["from_id"], e["to_id"]} == pair for e in edges):
                        edges.append({
                            "id": edge_id,
                            "from_id": prev_id,
                            "to_id": node_id
                        })
                        edge_counter += 1
                prev_id = node_id

        return nodes, edges

    def export_manual_route_action(self):
        if self.current_line:
            self.finish_line()

        if not self.lines:
            messagebox.showwarning("No Route Lines", "Please plot at least one line route on the canvas first.")
            return

        self.save_route_annotations(quiet=True)
        nodes, edges = self.manual_lines_to_graph()
        self.set_extracted_graph(nodes, edges)

        if self.on_export_manual_route:
            self.on_export_manual_route(nodes, edges)
        else:
            messagebox.showinfo("Export Manual Route", f"Generated {len(nodes)} waypoints and {len(edges)} segments from manual route.")

    def extract_graph_action(self):
        # First save any plotted lines
        if self.lines or self.current_line:
            if self.current_line:
                self.finish_line()
            self.save_route_annotations(quiet=True)

        if self.on_extract_graph:
            self.on_extract_graph()
        else:
            messagebox.showinfo("Extract Graph", "Automated graph extraction triggered.")



if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1024x768")
    app = RobotLinePlotterApp(root)
    root.mainloop()
