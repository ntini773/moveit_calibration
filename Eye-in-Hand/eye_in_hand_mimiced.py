import numpy as np
import cv2
import yaml
from scipy.spatial.transform import Rotation

def load_samples_from_yaml(yaml_file):
    """Load calibration samples from YAML file"""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    base_to_eef_transforms = []
    camera_to_target_transforms = []
    
    for sample in data:
        # Load 4x4 transformation matrices (row-major format)
        base_to_eef = np.array(sample['effector_wrt_world']).reshape(4, 4)
        camera_to_target = np.array(sample['object_wrt_sensor']).reshape(4, 4)
        
        base_to_eef_transforms.append(base_to_eef)
        camera_to_target_transforms.append(camera_to_target)
    
    return base_to_eef_transforms, camera_to_target_transforms

def transforms_to_opencv_format(transforms):
    """Convert 4x4 transformation matrices to OpenCV R,t format"""
    R_matrices = []
    t_vectors = []
    
    for T in transforms:
        R = T[:3, :3]  # Rotation part
        t = T[:3, 3].reshape(3, 1)  # Translation part
        R_matrices.append(R)
        t_vectors.append(t)
    
    return R_matrices, t_vectors

def solve_eye_in_hand_calibration(yaml_file):
    """
    Solve eye-in-hand calibration - camera mounted on robot end-effector
    Solves AX = XB where X is the camera-to-end-effector transformation
    """
    # Load samples
    base_to_eef_transforms, camera_to_target_transforms = load_samples_from_yaml(yaml_file)
    
    print(f"Loaded {len(base_to_eef_transforms)} calibration samples for eye-in-hand")
    
    # Convert to OpenCV format
    R_gripper2base, t_gripper2base = transforms_to_opencv_format(base_to_eef_transforms)
    R_target2cam, t_target2cam = transforms_to_opencv_format(camera_to_target_transforms)
    
    # Solve eye-in-hand calibration using OpenCV
    # This solves for the camera-to-end-effector transformation
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI  # You can try DANIILIDIS, PARK, etc.
    )
    
    # Combine into 4x4 transformation matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    print("\nEye-in-Hand Calibration Result:")
    print("Camera to End-Effector Transform (X in AX=XB):")
    print(T_cam2gripper)
    
    # Calculate reprojection error
    rotation_error, translation_error = calculate_reprojection_error_eye_in_hand(
        base_to_eef_transforms, camera_to_target_transforms, T_cam2gripper
    )
    
    print(f"\nReprojection Error:")
    print(f"Average Rotation Error: {rotation_error:.6f} rad")
    print(f"Average Translation Error: {translation_error:.6f} m")
    
    return T_cam2gripper, rotation_error, translation_error

def calculate_reprojection_error_eye_in_hand(gripper_poses, target_poses, X):
    """
    Calculate reprojection error for eye-in-hand setup
    
    For eye-in-hand: AX = XB where:
    - A = relative motion between gripper poses (gripper_i^{-1} * gripper_{i+1})
    - B = relative motion between target poses (target_{i+1} * target_i^{-1})  
    - X = camera-to-gripper transformation (what we solved for)
    
    Args:
        gripper_poses: List of 4x4 base-to-gripper transforms
        target_poses: List of 4x4 camera-to-target transforms
        X: 4x4 camera-to-gripper transformation (calibration result)
    """
    if len(gripper_poses) < 2:
        return 0.0, 0.0
        
    rotation_errors = []
    translation_errors = []
    
    for i in range(len(gripper_poses) - 1):
        # Calculate A: relative motion between consecutive gripper poses
        # A = gripper_i^{-1} * gripper_{i+1}
        A = np.linalg.inv(gripper_poses[i]) @ gripper_poses[i + 1]
        
        # Calculate B: relative motion between consecutive target poses
        # B = target_{i+1} * target_i^{-1}
        B = target_poses[i + 1] @ np.linalg.inv(target_poses[i])
        
        # For eye-in-hand, the equation is AX = XB
        # Calculate AX and XB
        AX = A @ X
        XB = X @ B
        
        # Calculate the difference (should be zero for perfect calibration)
        diff_matrix = AX - XB
        
        # Rotation error (Frobenius norm of rotation part difference)
        R_diff = diff_matrix[:3, :3]
        rot_error = np.linalg.norm(R_diff, 'fro')
        rotation_errors.append(rot_error)
        
        # Translation error (Euclidean norm of translation part difference)
        t_diff = diff_matrix[:3, 3]
        trans_error = np.linalg.norm(t_diff)
        translation_errors.append(trans_error)
    print(f"\nIndividual Rotation Errors: {rotation_errors}")
    print(f"Individual Translation Errors: {translation_errors}")
    avg_rot_error = np.mean(rotation_errors)
    avg_trans_error = np.mean(translation_errors)
    
    return avg_rot_error, avg_trans_error

def get_camera_pose_in_base_frame(base_to_eef, cam_to_eef):
    """
    Calculate camera pose in robot base frame for eye-in-hand setup
    
    Args:
        base_to_eef: 4x4 transform from robot base to end-effector
        cam_to_eef: 4x4 transform from camera to end-effector (calibration result)
    
    Returns:
        4x4 transform from robot base to camera
    """
    # For eye-in-hand: T_base_to_camera = T_base_to_eef * T_eef_to_camera
    # Since we have T_camera_to_eef, we need its inverse
    eef_to_cam = np.linalg.inv(cam_to_eef)
    base_to_cam = base_to_eef @ eef_to_cam
    
    return base_to_cam

def validate_calibration_eye_in_hand(yaml_file, cam_to_eef_transform):
    """
    Validate the calibration by projecting target points using the calibrated transform
    """
    base_to_eef_transforms, camera_to_target_transforms = load_samples_from_yaml(yaml_file)
    
    print("\nValidation - Camera poses in base frame for each sample:")
    for i, (base_to_eef, cam_to_target) in enumerate(zip(base_to_eef_transforms, camera_to_target_transforms)):
        # Get camera pose in base frame using calibrated transform
        base_to_cam = get_camera_pose_in_base_frame(base_to_eef, cam_to_eef_transform)
        
        # Get target pose in base frame
        base_to_target = base_to_cam @ cam_to_target
        
        print(f"Sample {i+1}:")
        print(f"  Camera in base: {base_to_cam[:3, 3]}")
        print(f"  Target in base: {base_to_target[:3, 3]}")

# Usage example
if __name__ == "__main__":
    # Load your calibration samples
    # yaml_file = "eye_in_hand_cam_link_oct30.yaml"
    yaml_file = "eye_in_hand_time_varun.yaml"
    
    
    # Solve eye-in-hand calibration
    cam_to_eef_transform, rot_error, trans_error = solve_eye_in_hand_calibration(yaml_file)
    
    # Validate results
    validate_calibration_eye_in_hand(yaml_file, cam_to_eef_transform)
    
    # Save the result (similar to C++ saveCameraPoseBtnClicked)
    print(f"\nCalibration completed!")
    print(f"Camera-to-End-Effector transformation:")
    print(cam_to_eef_transform)
