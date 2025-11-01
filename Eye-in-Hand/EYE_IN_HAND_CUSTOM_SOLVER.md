# Eye-in-Hand Custom Solver

## Overview

This Python implementation provides a custom eye-in-hand calibration solver inspired by the MoveIt Calibration C++ codebase. It mimics the mathematical approach and data structures used in the original ROS-based calibration system while leveraging OpenCV's optimized hand-eye calibration algorithms.

## Inspiration from C++ Codebase

### Source Files Analyzed
- `moveit_calibration_gui/handeye_calibration_rviz_plugin/src/handeye_control_widget.cpp`
- `moveit_calibration_plugins/handeye_calibration_solver/src/handeye_solver_default.cpp`
- Associated header files and plugin architecture

### Key Insights from C++ Implementation

1. **Transform Collection Strategy**
   ```cpp
   // C++ approach in takeTransformSamples()
   camera_to_object_tf = tf_buffer_->lookupTransform(frame_names_["sensor"], frame_names_["object"], ros::Time(0));
   base_to_eef_tf = tf_buffer_->lookupTransform(frame_names_["base"], frame_names_["eef"], ros::Time(0));
   
   effector_wrt_world_.push_back(tf2::transformToEigen(base_to_eef_tf));
   object_wrt_sensor_.push_back(tf2::transformToEigen(camera_to_object_tf));
   ```

2. **Solver Architecture**
   - Plugin-based solver system with multiple algorithms (Daniilidis1999, ParkBryan1994, TsaiLenz1989)
   - Consistent data format using Eigen::Isometry3d transformations
   - Error calculation methodology for validation

3. **Eye-in-Hand vs Eye-to-Hand Distinction**
   ```cpp
   // Frame configuration based on sensor mount type
   case mhc::EYE_IN_HAND:
       from_frame_tag_ = "eef";  // Camera relative to end-effector
   case mhc::EYE_TO_HAND:
       from_frame_tag_ = "base"; // Camera relative to base
   ```

## Mathematical Foundation

### Problem Formulation
For **eye-in-hand** calibration, we solve the classic hand-eye calibration equation:

**AX = XB**

Where:
- **A**: Relative motion between consecutive robot end-effector poses
- **B**: Relative motion between consecutive target poses (as seen by camera)
- **X**: Camera-to-end-effector transformation (what we're solving for)

### Transform Relationships
```
A = T_eef_i^(-1) * T_eef_(i+1)    # Robot motion
B = T_target_(i+1) * T_target_i^(-1)  # Target motion (inverted)
X = T_camera_to_eef               # Calibration result
```

## Python Implementation Approach

### 1. Data Compatibility
- **YAML Format**: Maintains compatibility with C++ saved sample format
- **Transform Representation**: 4x4 homogeneous transformation matrices
- **Sample Structure**: Preserves `effector_wrt_world` and `object_wrt_sensor` naming

### 2. Algorithm Selection
```python
# OpenCV methods corresponding to C++ solvers
cv2.CALIB_HAND_EYE_TSAI       # TsaiLenz1989
cv2.CALIB_HAND_EYE_PARK       # ParkBryan1994  
cv2.CALIB_HAND_EYE_HORAUD     # Alternative approach
cv2.CALIB_HAND_EYE_ANDREFF    # Alternative approach
cv2.CALIB_HAND_EYE_DANIILIDIS # Daniilidis1999
```

### 3. Error Calculation Methodology
Mimics the C++ `getReprojectionError()` function:

```python
def calculate_reprojection_error_eye_in_hand(gripper_poses, target_poses, X):
    """
    Calculate AX - XB for each pose pair
    Rotation error: Frobenius norm of rotation difference
    Translation error: Euclidean norm of translation difference
    """
    for i in range(len(gripper_poses) - 1):
        A = np.linalg.inv(gripper_poses[i]) @ gripper_poses[i + 1]
        B = target_poses[i + 1] @ np.linalg.inv(target_poses[i])
        
        AX = A @ X
        XB = X @ B
        
        # Calculate ||AX - XB|| errors
```

## Key Features

### 1. Complete Workflow Replication
- **Sample Loading**: From YAML files saved by C++ implementation
- **Calibration Solving**: Using OpenCV's optimized algorithms
- **Error Validation**: Same mathematical approach as C++ version
- **Result Interpretation**: Camera pose calculations and transformations

### 2. Eye-in-Hand Specific Implementation
- **Correct equation setup**: AX = XB formulation
- **Proper transform ordering**: Accounts for camera-moves-with-robot scenario
- **Base frame calculations**: Converts end-effector-relative results to world frame

### 3. Validation and Debugging
- **Reprojection error metrics**: Quantitative calibration quality assessment
- **Pose validation**: Cross-check results across multiple samples
- **Transform verification**: Ensure mathematical consistency

## Usage Workflow

### 1. Data Preparation
```python
# Load samples from C++ compatible YAML format
base_to_eef_transforms, camera_to_target_transforms = load_samples_from_yaml("samples.yaml")
```

### 2. Calibration Execution
```python
# Solve for camera-to-end-effector transformation
cam_to_eef_transform, rot_error, trans_error = solve_eye_in_hand_calibration("samples.yaml")
```

### 3. Result Validation
```python
# Validate calibration quality
validate_calibration_eye_in_hand("samples.yaml", cam_to_eef_transform)
```

## Advantages Over Direct C++ Usage

### 1. **Simplified Dependencies**
- No ROS/MoveIt installation required
- Pure Python + OpenCV + NumPy stack
- Cross-platform compatibility

### 2. **Enhanced Flexibility**  
- Easy algorithm comparison (multiple OpenCV methods)
- Straightforward result visualization and analysis
- Simple integration with existing Python workflows

### 3. **Debugging Capabilities**
- Direct access to intermediate calculations
- Easy modification of error metrics
- Comprehensive logging and validation

## Mathematical Validation

### Transform Chain Verification
For eye-in-hand setup, the complete transform chain:

```
T_base_to_target = T_base_to_eef * T_eef_to_camera * T_camera_to_target
```

Where `T_eef_to_camera = inv(T_camera_to_eef)` (our calibration result)

### Error Metrics
- **Rotation Error**: `||R_AX - R_XB||_F` (Frobenius norm)
- **Translation Error**: `||t_AX - t_XB||_2` (Euclidean norm)
- **Typical Good Results**: Rotation < 0.01 rad, Translation < 0.005 m

## Integration with Original Workflow

This Python solver can be used as:

1. **Standalone calibration tool**: Independent of ROS ecosystem
2. **Validation reference**: Cross-check C++ calibration results  
3. **Research platform**: Easy experimentation with different algorithms
4. **Educational tool**: Clear mathematical implementation for learning

## File Structure
```
Eye_in_hand/
├── eye_on_hand_mimiced.py          # Main solver implementation
├── EYE_IN_HAND_CUSTOM_SOLVER.md    # This documentation
└── samples.yaml                     # Calibration data (C++ compatible)
```

## Conclusion

This implementation successfully bridges the gap between the robust ROS-based MoveIt Calibration system and modern Python computer vision workflows. It preserves the mathematical rigor and validation methodology of the original C++ implementation while providing the flexibility and accessibility of Python-based development.

The solver maintains full compatibility with existing calibration data while offering enhanced debugging capabilities and simplified deployment scenarios.