import numpy as np
from scipy.interpolate import splprep, splev

def fit_plane_to_point_cloud(x):
    M = x.sum(axis=0) / x.shape[0] # Find the mean coordinate.
    u, s, vh = np.linalg.svd(x - M) # Run SVD.
    plane_normal = vh[2] # Unit normal vector.
    return plane_normal

def project_vector_onto_plane(v, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal) # Normalize the plane normal vector.
    v = v - np.dot(v, plane_normal) * plane_normal # Project the vector onto the plane.
    
    # Compute orthogonal basis vectors (u, w) for the plane
    u = np.cross(plane_normal, np.array([1, 0, 0]))
    if np.linalg.norm(u) == 0:
        u = np.cross(plane_normal, np.array([0, 1, 0]))
    u = u / np.linalg.norm(u)
    w = np.cross(plane_normal, u)
    
    # Project vectors onto the new basis
    v = np.array([np.dot(v, u), np.dot(v, w)])
    
    return v

def get_mean_tangent(x):
    '''
    M = x.sum(axis=0) / x.shape[0] # Find the mean coordinate.
    u, s, vh = np.linalg.svd(x - M) # Run SVD.
    mean_tangent = vh[0] # Unit normal vector.
    '''
    tangents = np.diff(x, axis=0) # Compute the difference vectors between consecutive points.
    norms = np.linalg.norm(tangents, axis=1)
    unit_tangents = tangents / norms[:, np.newaxis] # Normalize the difference vectors to get unit vectors.
    mean_tangent = np.mean(unit_tangents, axis=0) # Compute the mean of the unit vectors.
    mean_tangent = mean_tangent / np.linalg.norm(mean_tangent) # Normalize the mean vector to get the mean tangent.
    mean_tangent *= -1
    
    return mean_tangent
    
def get_angle_diff(a0, a1):
    # a0 is angle at time t, and a1 is angle at time t+1.
    # Returns the (smallest) angle difference [rad] with sign.
    # The returned angle is always < np.pi in absolute value.
    
    da = np.maximum(a0,a1) - np.minimum(a0,a1) # Always positive.
    
    if(a1 > a0 and da > np.pi):
        da = -((2*np.pi) - da)
    elif(a1 < a0):
        if(da < np.pi):
            da = -da
        else:
            da = (2*np.pi) - da
            
    return da
    
def get_principal_plane_rotation(midline, prev_normal):
    
    # TODO: check that v0 and v1 are assigned the correct values (and are different)
    v0 = prev_normal.copy()
    v1 = fit_plane_to_point_cloud(midline)
    
    # Project normal vectors onto the plane defined by the normal vector corresponding to the main body axis
    mean_body_tangent = get_mean_tangent(midline)
    v0p = project_vector_onto_plane(v0, mean_body_tangent)
    v1p = project_vector_onto_plane(v1, mean_body_tangent)
    
    # Flip the plane normal if the angle difference from the previous normal is larger than 90
    dn = np.math.atan2(np.linalg.det([v0p,v1p]),np.dot(v0p,v1p)) % (2*np.pi)
    # dn = np.math.atan2(np.linalg.det([v0[[0,2]],v1[[0,2]]]),np.dot(v0[[0,2]],v1[[0,2]])) % (2*np.pi)
    if(dn > np.pi/2):
        v1 *= -1 # Flip the normal.
        v1p *= -1
    
    pp_a1 = np.math.atan2(v0p[1], v0p[0]) % (2*np.pi)
    pp_a2 = np.math.atan2(v1p[1], v1p[0]) % (2*np.pi)
    principal_plane_angle_diff = get_angle_diff(pp_a1, pp_a2) # The output is an angle within [-π,+π].
    
    if(abs(principal_plane_angle_diff) > (np.pi/2)):
        principal_plane_angle_diff = - np.sign(principal_plane_angle_diff) * (np.pi - abs(principal_plane_angle_diff)) # Take the smaller angle and flip the sign.
        v1 *= -1 # Flip the normal.
        # print(round(pp_a1 * 180 / np.pi,0), round(pp_a2 * 180 / np.pi,0), round(principal_plane_angle_diff * 180 / np.pi,0))
    
    return principal_plane_angle_diff, v1
    
def interpolate_line(xyz, n_eval):
    tck, u = splprep(xyz, s=0) # Fit a B-spline to the data.
    new_u = np.linspace(0, 1, n_eval) # len(x)
    xyz = splev(new_u, tck) # Evaluate the B-spline at new parameter values.
    return xyz