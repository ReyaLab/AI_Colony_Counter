# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 15:26:47 2025

@author: mchva


This is the Tkinter GUI for Mari's image analysis program

"""

import tkinter as tk
import os
import sys

def find_directories(path):
    """
    Returns a list of directories found within the given path.
    """
    directories = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
    final = []
    for x in directories:
        file_list = os.listdir(path+x+'/')
        file_list = [x for x in file_list if ((x[-4::]==".tif") or (x[-4::]==".tiff"))]
        if len(file_list) >= 1:
            final.append(path+x+'/')
    return(final)


class HoverText:
    """Chatgpt wrote all the code for hovertext lmao"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip:
            return
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)  # Remove window decorations
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, background="white", relief="solid", borderwidth=1, justify="left", anchor="w")
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

filename = ""
global file_list 
file_list = [""]
root = tk.Tk()
root.title("Reya Lab AI Colony Detector")
root.geometry("900x900")  # Set the window size

try:
    root.iconbitmap("C:/Users/mchva/Desktop/Reya Lab/Data/Mari_image_Analysis_Jan2025/cell_icon.ico")
except:
    pass

file_chosen = tk.StringVar(root) 
# Set the default value of the variable 
file_chosen.set("Select an Option") 

label_size_min = tk.Label(root, text="Colony size minimum cutoff:")
label_size_min.grid(row=0, column=0, padx=10, pady=5, sticky="w")
entry_size_min = tk.Entry(root, width=30)
entry_size_min.grid(row=0, column=1, padx=10, pady=5)
entry_size_min.insert(0, "1000")

label_circularity = tk.Label(root, text="Colony circularity cutoff:")
label_circularity.grid(row=2, column=0, padx=10, pady=5, sticky="w")
entry_circularity = tk.Entry(root, width=30)
entry_circularity.grid(row=2, column=1, padx=10, pady=5)
entry_circularity.insert(0, ".25")

def checkbox_change():
    if (checkbox_var.get()):
        dropdown.config(state="disabled")
        global group_analysis
        group_analysis = tk.IntVar()
        global checkbox_group
        #modifying a global object, root, from a function is bad practice. Oh well!
        checkbox_group= tk.Checkbutton(root, text='Z-stack',variable=group_analysis, onvalue=True, offvalue=False)
        checkbox_group.grid(row=4, column=2, padx=10, pady=5)
        
        global multiple_groups
        multiple_groups = tk.IntVar()
        global checkbox_multiple
        checkbox_multiple= tk.Checkbutton(root, text='Do multiple folders',variable=multiple_groups, onvalue=True, offvalue=False)
        checkbox_multiple.grid(row=4, column=3, padx=10, pady=5)
        
        HoverText(checkbox_group, "Z-stack all tif files in a single folder.")
        HoverText(checkbox_multiple, "Apply selected analysis to every folder in the path.")
    else:
        dropdown.config(state="normal")
        try:
            checkbox_group.grid_forget()
            group_analysis.set(False)
            
            checkbox_multiple.grid_forget()
            multiple_groups.set(False)
        except:
            pass
        
def on_entry_change(*args):
    global file_list
    if (entry_path.get() == "") or (entry_path.get() == "Enter image path"):  # Check if the entry box is filled
        label_hidden.config(text="Enter valid path.")
        file_list = [""]
        try:
            file_chosen.set("")
        except:
            pass
    else:
        try:
            file_list = os.listdir(entry_path.get())
            file_list = [x for x in file_list if ((x[-4::]==".tif") or (x[-4::]==".tiff"))]
            if len(file_list) == 0:
                label_hidden.config(text="There are no .tif files in that folder.")
                try:
                    file_chosen.set("")
                except:
                    pass
                return(0)
            else:
                label_hidden.config(text="")
            label_drop = tk.Label(root, text="Choose file:")
            label_drop.grid(row=4, column=0, padx=10, pady=5, sticky="w")
            frame = tk.Frame(root)
            global dropdown
            dropdown = tk.OptionMenu(frame, file_chosen, *file_list) 
            dropdown.pack(side="left", padx=5)
            global checkbox_var
            checkbox_var = tk.IntVar()
            global checkbox_all
            checkbox_all= tk.Checkbutton(frame, text='Analyze group',variable=checkbox_var, onvalue=True, offvalue=False, command=checkbox_change)
            checkbox_all.pack(side="right", padx=5)
            frame.grid(row=4, column=1, padx=5, pady=5)
            HoverText(checkbox_all, "Analyze multiple files.")
            try:
                checkbox_group.grid_forget()
                group_analysis.set(False)
                
                checkbox_multiple.grid_forget()
                multiple_groups.set(False)
            except:
                pass
        except:
            label_hidden.config(text="Enter valid path.")
            file_list = [""]
            try:
                file_chosen.set("")
            except:
                pass
        
entry_path_txt = tk.StringVar(value="/home/mattc/Documents/ColonyAssaySegformer/")
entry_path = tk.Entry(root, width=30, textvariable=entry_path_txt)
entry_path.grid(row=3, column=1, padx=10, pady=5)
entry_path_txt.trace_add("write", on_entry_change)


label_hidden = tk.Label(root, text="")
label_hidden.grid(row=5, column=0, padx=10, pady=5, sticky="w")

def run_analysis(filn, path, params = [0,0], do_all = False):
    sys.path.append(path)
    label_hidden.config(text="Initiated analysis.")
    if do_all == False:
        import Colony_Analyzer_AI2 as MA
        from PIL import Image, ImageTk
        import cv2
        img = MA.main([path, filn, params[0], params[1]])
        cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        del img
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv_image_rgb)
        del cv_image_rgb
        # Resize image (optional)
        pil_image = pil_image.resize((600, 600), Image.LANCZOS)
        
        # Convert PIL image to PhotoImage
        img = ImageTk.PhotoImage(pil_image)
        del pil_image
        # Display in Tkinter label
        labelimg = tk.Label(root, image=img)
        labelimg.photo = img
        labelimg.grid(row=6, column=1)
    elif (do_all == True) and (group_analysis.get() == True):
        import Colony_Analyzer_AI_zstack2 as MA
        from PIL import Image, ImageTk
        import cv2
        img = MA.main([path, file_list, params[0], params[1]])
        cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        del img
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(cv_image_rgb)
        del cv_image_rgb
        # Resize image (optional)
        pil_image = pil_image.resize((600, 600), Image.LANCZOS)
        
        # Convert PIL image to PhotoImage
        img = ImageTk.PhotoImage(pil_image)
        del pil_image
        # Display in Tkinter label
        labelimg = tk.Label(root, image=img)
        labelimg.photo = img
        labelimg.grid(row=6, column=1)
    else:
        import Colony_Analyzer_AI2 as MA
        from PIL import Image, ImageTk
        import cv2
        for x in file_list:
            img = MA.main([path, x, params[0], params[1]])
            cv_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            del img
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(cv_image_rgb)
            del cv_image_rgb
            # Resize image (optional)
            pil_image = pil_image.resize((600, 600), Image.LANCZOS)
            
            # Convert PIL image to PhotoImage
            img = ImageTk.PhotoImage(pil_image)
            del pil_image
            # Display in Tkinter label
            labelimg = tk.Label(root, image=img)
            labelimg.photo = img
            labelimg.grid(row=6, column=1)
            root.update()
#if (entry_size_min.get() == "") or (entry_size_max.get() == "") or (entry_circularity.get() == ""):
        label_hidden.config(text="Please fix input.")
def click_conf():
    global file_list
    if (not entry_size_min.get().isnumeric()) or (entry_circularity.get()==''):
        label_hidden.config(text="Please fix input.")
    elif ((file_chosen.get() in file_list) or (checkbox_var.get())) and (len(file_list)>0) and (file_list != ['']) and (int(entry_size_min.get())>=0 or float(entry_circularity.get())>=0):
        if multiple_groups.get() == True:
            directories = find_directories(entry_path.get())
            if len(directories) < 1:
                label_hidden.config(text="No folders with tif files detected.")
            else:
                label_hidden.config(text='Detected the following folders:\n'+'\n'.join(directories))
                filename = file_chosen.get()
                for x in directories:
                    file_list = os.listdir(x)
                    file_list = [y for y in file_list if ((y[-4::]==".tif") or (y[-4::]==".tiff"))]
                    run_analysis(filename, x, [int(entry_size_min.get()),float(entry_circularity.get())], do_all = checkbox_var.get())
                label_hidden.config(text="Finished analysis.") 
        else:
            file_list = os.listdir(entry_path.get())
            file_list = [y for y in file_list if ((y[-4::]==".tif") or (y[-4::]==".tiff"))]
            label_hidden.config(text="")
            filename = file_chosen.get()
            run_analysis(filename, entry_path.get(), [int(entry_size_min.get()),float(entry_circularity.get())], do_all = checkbox_var.get())
            label_hidden.config(text="Finished analysis.")       
    else:
        label_hidden.config(text="Please fix input.")

btn_confirm = tk.Button(root, text = "Confirm parameters and analyze" , fg = "black", command=click_conf)
# Set Button Grid
btn_confirm.grid(row=5, column=1, padx=10, pady=5)
on_entry_change()

HoverText(entry_size_min, "Minimum size in pixels to be considered a valid colony.")
HoverText(entry_circularity, "Minimum circularity cutoff of colony, defined as 4*pi*area/perimeter^2. \nCloser to 1 is more circular.")


root.mainloop()