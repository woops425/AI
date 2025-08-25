import numpy as np
import cv2


def apply_dynamic_transform(ref_init, ref_curr, key_positions):
    matched_labels = list(set(ref_init.keys()) & set(ref_curr.keys()))
    if len(matched_labels) < 3:
        return key_positions

    matched_labels = matched_labels[:3]
    init_pts = np.array([ref_init[k] for k in matched_labels], dtype=np.float32)
    curr_pts = np.array([ref_curr[k] for k in matched_labels], dtype=np.float32)
    M = cv2.getAffineTransform(init_pts, curr_pts)

    corrected_positions = []
    for key in key_positions:
        pt = np.array([[key['center_x'], key['center_y']]], dtype=np.float32)
        new_pt = cv2.transform(np.array([pt]), M)[0][0]
        corrected_positions.append({
            "label": key["label"],
            "x": new_pt[0] - key["width"] / 2,
            "y": new_pt[1] - key["height"] / 2,
            "width": key["width"],
            "height": key["height"]
        })
    return corrected_positions