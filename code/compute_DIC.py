import os
import numpy as np
import muDIC as dic
import json # To save parameters


# Define paths
folder = "/home/anais/Alexandre/data/Calibration_top_grey"
pathImages = os.path.join(folder, "snapshots")
pathFields = os.path.join(folder, "muDIC")

# Define parameters
frames_to_keep = np.arange(0, 700, 5)
overwrite_output = True
plot = False

# Advanced parameters
max_iter = 80                  # default: 40 (but 80 often usefull)
interpolation_order = 3        # default: 3
frequency_reference = 5        # default: 15 (but sometimes a bit to large) or 5
manual_limits = True
manual_limits_val = {"Xc1":400, "Xc2":1900, 
                     "Yc1":500, "Yc2":1700, 
                     "n_elx":50, "n_ely":40}


############# CODE #############

# Import images
image_stack = dic.image_stack_from_folder(pathImages, file_type='.tiff')
nbImages = len(image_stack)


# Only keep desired frames
toDrop_bool = np.full(nbImages, True)
toDrop_bool[frames_to_keep] = False
frames_to_skip = np.arange(nbImages)[toDrop_bool].tolist()
image_stack.skip_images(frames_to_skip)


# Mesh the images
mesher = dic.Mesher()
print('Mesher created')
if manual_limits:
    mesh = mesher.mesh(image_stack, GUI=False, **manual_limits_val)
else:
    mesh = mesher.mesh(image_stack)


# Prepare the inputs and run the analysis
ref_frames = np.arange(0, len(frames_to_keep), frequency_reference).tolist() # Change the reference frame
inputs = dic.DICInput(mesh, image_stack, maxit=max_iter, ref_update_frames=ref_frames,
                      interpolation_order=interpolation_order, noconvergence="ignore")
# TODO: Understand what happens to the data when noconvergence is ignored
dic_job = dic.DICAnalysis(inputs)
results = dic_job.run()
print("DICAnalysis ended.")


# Compute the fields
fields = dic.Fields(results)
true_strain = fields.true_strain()
eng_strain = fields.eng_strain()
green_strain = fields.green_strain()
coords = fields.coords()


# Save the fields
written = []
overwritten = []
to_write = {"frames": frames_to_keep, "coords": coords,
            "true_strain": true_strain, "eng_strain": eng_strain, 
            "green_strain": green_strain}


# Create folder to save fields if needed
if not os.path.isdir(pathFields):
    os.makedirs(pathFields)

# Save the fields
for name, content in to_write.items():
    filepath = os.path.join(pathFields, name)

    if not os.path.isfile(filepath):
        written.append(name)
        np.save(filepath, content)
    
    else:
        overwritten.append(name)
        if overwrite_output: 
            np.save(filepath, content)

print(f"New files saved: {written}.")
print(f"Already existing: {overwritten}. Overwritten: {overwrite_output}.")

# Save parameters
parameters = {"max_iter": max_iter, "interpolation_order": interpolation_order,
              "frequency_reference": frequency_reference, "ref_frames": ref_frames, 
              "min_x": mesh.Xc1, "max_x": mesh.Xc2, "min_y": mesh.Yc1, "max_y": mesh.Yc2,
              "nel_x": mesh.n_elx, "nel_y": mesh.n_ely, 
              "saved": written, "existing": overwritten, "overwritten": overwrite_output}
path_params = os.path.join(pathFields, 'params.txt')
with open(path_params, 'w') as f:
    f.write(json.dumps(parameters))


if plot:
    # Create visualizer and show it
    viz = dic.Visualizer(fields, images=image_stack)
    viz.show(field="True strain", component=(0,0), frame=len(frames_to_keep)-1)