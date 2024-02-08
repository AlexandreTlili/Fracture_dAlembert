import os
import numpy as np
import muDIC as dic
import json # To save parameters

import modules.calibration_origin as calib


# Define paths
folder = "data/Experiments/h_1p5cm/tauC_100Pa/1_CCGM_500um_375nm_45uN_100Pa_h_1p5cm_m_137g"
pathImages = os.path.join(folder, "snapshots_preProcessed")
pathFields = os.path.join(folder, "muDIC")

# Define parameters
frames_to_keep = np.arange(0, 900, 5)
#frames_to_keep = np.arange(490, 500, 5)
overwrite_output = True
plot = False

# Advanced parameters
max_iter = 80                  # default: 40 (but 80 often usefull)
interpolation_order = 3        # default: 3
frequency_reference = 5        # default: 15 (but sometimes a bit to large) or 5
automatic_limits = True
automatic_origin = False

automatic_limits_val = {"Xc1":100, "Xc2":1000, 
                        "Yc1":100, "Yc2":1000, 
                        "n_elx":30, "n_ely":30}
automatic_origin_val = {"X0": 1023, 
                        "scale_pix_to_m": 7.6678e-5}


############# CODE #############

# Import images
image_stack = dic.image_stack_from_folder(pathImages, file_type='.tiff')
nbImages = len(image_stack)


# Only keep desired frames
frames_to_keep = frames_to_keep[frames_to_keep < nbImages]
toDrop_bool = np.full(nbImages, True)
toDrop_bool[frames_to_keep] = False
frames_to_skip = np.arange(nbImages)[toDrop_bool].tolist()
image_stack.skip_images(frames_to_skip)

# Set the origin and the scale factor
if automatic_origin:
    xOrigin, scale_pix_to_m = automatic_origin_val['X0'], automatic_origin_val['scale_pix_to_m']
else:
    xOrigin, scale_pix_to_m = calib.find_origin_and_scale(image_stack[0])

# Mesh the images
mesher = dic.Mesher()
print('Mesher created')
if automatic_limits:
    mesh = mesher.mesh(image_stack, GUI=False, **automatic_limits_val)
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
deformation_gradient = fields.F()
true_strain = fields.true_strain()
eng_strain = fields.eng_strain()
green_strain = fields.green_strain()
coords = fields.coords()


# Save the fields
written = []
overwritten = []
to_write = {"frames": frames_to_keep, "coords": coords, "deformation_gradient": deformation_gradient,
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
              "saved": written, "existing": overwritten, "overwritten": overwrite_output, 
              "xOrigin": xOrigin, "scale_pix_to_m": scale_pix_to_m}
path_params = os.path.join(pathFields, 'params.txt')
with open(path_params, 'w') as f:
    f.write(json.dumps(parameters))


if plot:
    # Create visualizer and show it
    viz = dic.Visualizer(fields, images=image_stack)
    viz.show(field="True strain", component=(0,0), frame=len(frames_to_keep)-1)