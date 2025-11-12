"""Projective Homography and Panorama Solution."""
import numpy as np

from typing import Tuple
from random import sample
from collections import namedtuple


from numpy.linalg import svd
from scipy.interpolate import griddata


PadStruct = namedtuple('PadStruct',
                       ['pad_up', 'pad_down', 'pad_right', 'pad_left'])


class Solution:
    """Implement Projective Homography and Panorama Solution."""
    def __init__(self):
        pass

    @staticmethod
    def compute_homography_naive(match_p_src: np.ndarray,
                                 match_p_dst: np.ndarray) -> np.ndarray:
        """Compute a Homography in the Naive approach, using SVD decomposition.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.

        Returns:
            Homography from source to destination, 3x3 numpy array.
        """
        #return homography
        """INSERT YOUR CODE HERE"""
        N = match_p_src.shape[1]
        A = np.zeros((2 * N, 9)) 
        for i in range(N):
            u, v = match_p_src[0, i], match_p_src[1, i]           
            up, vp = match_p_dst[0, i], match_p_dst[1, i] # Get the destination point (u', v')
            A[2 * i] = [u, v, 1, 0, 0, 0, -up * u, -up * v, -up]  # Fill the 2i-th row of A (from the u' equation)
            A[2 * i + 1] = [0, 0, 0, u, v, 1, -vp * u, -vp * v, -vp] # Fill the (2i+1)-th row of A (from the v' equation)        
        try:
            U, S, Vh = np.linalg.svd(A)
        except np.linalg.LinAlgError as e:
            print(f"Error computing SVD: {e}")
            return np.eye(3) # Return identity matrix as a fallback
        h = Vh[-1, :]
        
        homography = h.reshape((3, 3)) # Reshape the 9x1 vector into the 3x3 homography matrix H
        return homography
      

    @staticmethod
    def compute_forward_homography_slow(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in the Naive approach, using loops.

        Iterate over the rows and columns of the source image, and compute
        the corresponding point in the destination image using the
        projective homography. Place each pixel value from the source image
        to its corresponding location in the destination image.
        Don't forget to round the pixel locations computed using the
        homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""
        
        src_h, src_w, _ = src_image.shape # Get source image dimensions
        dst_h, dst_w, _ = dst_image_shape # Get destination image dimensions
        forward_map = np.zeros(dst_image_shape, dtype=src_image.dtype) # Create an empty destination image
        
        # Iterate over every pixel (y, x) in the SOURCE image
        for y in range(src_h):
            for x in range(src_w): 
                p_src = np.array([x, y, 1])
                p_dst_prime = homography @ p_src
                w_prime = p_dst_prime[2]
                if w_prime == 0:
                    continue
                # Normalize
                x_prime = p_dst_prime[0] / w_prime
                y_prime = p_dst_prime[1] / w_prime
                
                #Round to the nearest integer pixel location
                x_dst = int(np.round(x_prime))
                y_dst = int(np.round(y_prime))
                
                # Check if the destination pixel is within the bounds
                if (0 <= x_dst < dst_w) and (0 <= y_dst < dst_h):
                    forward_map[y_dst, x_dst] = src_image[y, x]
                    
        return forward_map

    @staticmethod
    def compute_forward_homography_fast(
            homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute a Forward-Homography in a fast approach, WITHOUT loops.

        (1) Create a meshgrid of columns and rows.
        (2) Generate a matrix of size 3x(H*W) which stores the pixel locations
        in homogeneous coordinates.
        (3) Transform the source homogeneous coordinates to the target
        homogeneous coordinates with a simple matrix multiplication and
        apply the normalization you've seen in class.
        (4) Convert the coordinates into integer values and clip them
        according to the destination image size.
        (5) Plant the pixels from the source image to the target image according
        to the coordinates you found.

        Args:
            homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination.
            image height, width and color dimensions.

        Returns:
            The forward homography of the source image to its destination.
        """
        # return new_image
        """INSERT YOUR CODE HERE"""

        # Get source and destination dimensions
        src_h, src_w, _ = src_image.shape
        dst_h, dst_w, _ = dst_image_shape
        
        # Create an empty destination image
        forward_map = np.zeros(dst_image_shape, dtype=src_image.dtype)

        # (1) Create a meshgrid for the source image
        xx, yy = np.meshgrid(np.arange(src_w), np.arange(src_h))

        # (2) Generate 3x(H*W) matrix of homogeneous coordinates
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        
        ones_flat = np.ones(src_h * src_w)
        p_src_all = np.vstack((x_flat, y_flat, ones_flat))

        # (3) Apply homography and normalize
        p_dst_prime = homography @ p_src_all
        w_prime = p_dst_prime[2, :]
        w_prime[w_prime == 0] = 1e-9 # Avoid division by zero. Replace 0 with a tiny number
        
        # Normalize
        x_prime_norm = p_dst_prime[0, :] / w_prime
        y_prime_norm = p_dst_prime[1, :] / w_prime

        # (4) Convert coordinates to integers
        x_dst = np.round(x_prime_norm).astype(int)
        y_dst = np.round(y_prime_norm).astype(int)

        # (5) Plant the pixels
        mask = (x_dst >= 0) & (x_dst < dst_w) & \
               (y_dst >= 0) & (y_dst < dst_h)
        
        y_dst_valid = y_dst[mask]
        x_dst_valid = x_dst[mask]
        y_src_valid = y_flat[mask].astype(int)
        x_src_valid = x_flat[mask].astype(int)
        forward_map[y_dst_valid, x_dst_valid] = src_image[y_src_valid, x_src_valid]

        return forward_map

    @staticmethod
    def test_homography(homography: np.ndarray,
                        match_p_src: np.ndarray,
                        match_p_dst: np.ndarray,
                        max_err: float) -> Tuple[float, float]:
        """Calculate the quality of the projective transformation model.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.

        Returns:
            A tuple containing the following metrics to quantify the
            homography performance:
            fit_percent: The probability (between 0 and 1) validly mapped src
            points (inliers).
            dist_mse: Mean square error of the distances between validly
            mapped src points, to their corresponding dst points (only for
            inliers). In edge case where the number of inliers is zero,
            return dist_mse = 10 ** 9.
        """
        # return fit_percent, dist_mse
        """INSERT YOUR CODE HERE"""
        # stack ones to match_p_src
        ones = np.ones((1, match_p_src.shape[1]))
        src_h = np.vstack((match_p_src, ones))

        # apply the homography to src points and normalize
        proj = homography @ src_h
        proj /= proj[2, :]  # divide each column by w
        projected_pts = proj[:2, :]

        # compute squared distances
        diffs = projected_pts - match_p_dst
        dists_sq = np.sum(diffs ** 2, axis=0)

        # determine inliers
        mask = dists_sq <= max_err ** 2
        num_inliers = np.sum(mask)
        total_pts = match_p_src.shape[1]
        fit_percent = num_inliers / total_pts

        # calculate dist_mse between mapped points to dst points for inliers only
        if num_inliers > 0:
            dist_mse = np.mean(dists_sq[mask])
        else:
            dist_mse = 10 ** 9

        return fit_percent, dist_mse

    @staticmethod
    def meet_the_model_points(homography: np.ndarray,
                              match_p_src: np.ndarray,
                              match_p_dst: np.ndarray,
                              max_err: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return which matching points meet the homography.

        Loop through the matching points, and return the matching points from
        both images that are inliers for the given homography.

        Args:
            homography: 3x3 Projective Homography matrix.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            A tuple containing two numpy nd-arrays, containing the matching
            points which meet the model (the homography). The first entry in
            the tuple is the matching points from the source image. That is a
            nd-array of size 2xD (D=the number of points which meet the model).
            The second entry is the matching points form the destination
            image (shape 2xD; D as above).
        """
        # return mp_src_meets_model, mp_dst_meets_model
        """INSERT YOUR CODE HERE"""

        # stack ones
        ones = np.ones((1, match_p_src.shape[1]), dtype=match_p_src.dtype)
        src_h = np.vstack((match_p_src, ones))  # [x; y; 1]

        # apply homography
        proj = homography @ src_h  # (3, N)
        w = proj[2, :]
        # avoid division by zero
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        pred = proj[:2, :] / w  # normalize to get (2, N)

        # compute squared distances
        diffs = pred - match_p_dst
        d2 = np.sum(diffs ** 2, axis=0)

        # use mask to filter points
        mask = d2 <= (max_err ** 2)
        mp_src_meets_model = match_p_src[:, mask]
        mp_dst_meets_model = match_p_dst[:, mask]

        return mp_src_meets_model, mp_dst_meets_model
    
    def compute_homography(self,
                           match_p_src: np.ndarray,
                           match_p_dst: np.ndarray,
                           inliers_percent: float,
                           max_err: float) -> np.ndarray:
        """Compute homography coefficients using RANSAC to overcome outliers.

        Args:
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in
            pixels) between the mapped src point to its corresponding dst
            point, in order to be considered as valid inlier.
        Returns:
            homography: Projective transformation matrix from src to dst.
        """
        # # use class notations:
        # w = inliers_percent
        # # t = max_err
        # # p = parameter determining the probability of the algorithm to
        # # succeed
        # p = 0.99
        # # the minimal probability of points which meets with the model
        # d = 0.5
        # # number of points sufficient to compute the model
        # n = 4
        # # number of RANSAC iterations (+1 to avoid the case where w=1)
        # k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n))) + 1
        # return homography
        """INSERT YOUR CODE HERE"""
        N = match_p_src.shape[1]
        n = 4
        p = 0.99
        w = float(inliers_percent)
        t = max_err
        d = 0.5
        # Guard extremes to avoid log(0)
        w = np.clip(w, 1e-6, 1 - 1e-6)

        k = int(np.ceil(np.log(1 - p) / np.log(1 - w ** n)))

        min_inliers = max(n, int(np.ceil(d * N)))  # need at least d*N (and at least n)

        best_inliers_src = None
        best_inliers_dst = None
        best_inlier_count = -1
        best_mse = np.inf
        best_homography = None

        idx_all = np.arange(N)

        # k iterations
        for _ in range(k):

            if N < n:
                # not enough points
                break

            # sample n matches
            sample_idx = np.random.choice(idx_all, size=n, replace=False)
            Ps4 = match_p_src[:, sample_idx]  # shape (2, 4)
            Pd4 = match_p_dst[:, sample_idx]  # shape (2, 4)

            # compute the model using these points
            try:
                H_t = self.compute_homography_naive(Ps4, Pd4)  # 3x3
            except Exception:
                # degenerate sample, skip this iteration
                continue

            # inliers for this model
            Ps_in, Pd_in = self.meet_the_model_points(H_t, match_p_src, match_p_dst, t)
            inlier_count = Ps_in.shape[1]
            if inlier_count < min_inliers:
                continue

            # re-fit the model using all inliers
            try:
                H_refit = self.compute_homography_naive(Ps_in, Pd_in)
            except Exception:
                continue

            # if err is better save the model
            _, mse = self.test_homography(H_refit, Ps_in, Pd_in, t)
            if (best_inlier_count == -1) or (mse < best_mse):
                best_inlier_count = inlier_count
                best_mse = mse
                best_inliers_src = Ps_in
                best_inliers_dst = Pd_in


        if best_inlier_count > 0 and best_inliers_src is not None:
            H = self.compute_homography_naive(best_inliers_src, best_inliers_dst)
        else:
            # fallback if no acceptable model found
            H = self.compute_homography_naive(match_p_src, match_p_dst)

        # normalize scale for stability
        if abs(H[2, 2]) > 1e-12:
            H = H / H[2, 2]

        return H

    @staticmethod
    def compute_backward_mapping(
            backward_projective_homography: np.ndarray,
            src_image: np.ndarray,
            dst_image_shape: tuple = (1088, 1452, 3)) -> np.ndarray:
        """Compute backward mapping.

        (1) Create a mesh-grid of columns and rows of the destination image.
        (2) Create a set of homogenous coordinates for the destination image
        using the mesh-grid from (1).
        (3) Compute the corresponding coordinates in the source image using
        the backward projective homography.
        (4) Create the mesh-grid of source image coordinates.
        (5) For each color channel (RGB): Use scipy's interpolation.griddata
        with an appropriate configuration to compute the bi-cubic
        interpolation of the projected coordinates.

        Args:
            backward_projective_homography: 3x3 Projective Homography matrix.
            src_image: HxWx3 source image.
            dst_image_shape: tuple of length 3 indicating the destination shape.

        Returns:
            The source image backward warped to the destination coordinates.
        """

        # return backward_warp
        """INSERT YOUR CODE HERE"""
        dst_h, dst_w, _ = dst_image_shape
        src_h, src_w, _ = src_image.shape

        # (1) Create meshgrid for destination image
        xx_dst, yy_dst = np.meshgrid(np.arange(dst_w), np.arange(dst_h))
        
        # (2) Create homogeneous coordinates for destination
        x_flat = xx_dst.flatten()
        y_flat = yy_dst.flatten()
        ones_flat = np.ones_like(x_flat)
        p_dst_all = np.vstack((x_flat, y_flat, ones_flat)) # 3 x (H_dst * W_dst)

        # (3) Apply backward homography: p_src = H_inv * p_dst
        p_src_prime = backward_projective_homography @ p_dst_all
        
        # Normalize
        w_prime = p_src_prime[2, :]
        w_prime[w_prime == 0] = 1e-9 
        x_src_norm = p_src_prime[0, :] / w_prime
        y_src_norm = p_src_prime[1, :] / w_prime
        
        # (4) Create meshgrid of source image coords
        xx_src, yy_src = np.meshgrid(np.arange(src_w), np.arange(src_h))
        points_src = np.vstack((xx_src.flatten(), yy_src.flatten())).T
        
        # (5) Interpolate each channel
        backward_map = np.zeros(dst_image_shape, dtype=src_image.dtype)
        query_points = np.vstack((x_src_norm, y_src_norm)).T
        
        for channel in range(3):
            # Get the values for the current channel from the source
            values_src = src_image[:, :, channel].flatten()
            
            # Interpolate
            interpolated_channel = griddata(points_src,       
                                            values_src,      
                                            query_points,    
                                            method='cubic',  
                                            fill_value=0)     
            
            # Reshape back to destination image shape
            backward_map[:, :, channel] = interpolated_channel.reshape((dst_h, dst_w))

        return backward_map

    @staticmethod
    def find_panorama_shape(src_image: np.ndarray,
                            dst_image: np.ndarray,
                            homography: np.ndarray
                            ) -> Tuple[int, int, PadStruct]:
        """Compute the panorama shape and the padding in each axes.

        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            homography: 3x3 Projective Homography matrix.

        For each image we define a struct containing it's corners.
        For the source image we compute the projective transformation of the
        coordinates. If some of the transformed image corners yield negative
        indices - the resulting panorama should be padded with at least
        this absolute amount of pixels.
        The panorama's shape should be:
        dst shape + |the largest negative index in the transformed src index|.

        Returns:
            The panorama shape and a struct holding the padding in each axes (
            row, col).
            panorama_rows_num: The number of rows in the panorama of src to dst.
            panorama_cols_num: The number of columns in the panorama of src to
            dst.
            padStruct = a struct with the padding measures along each axes
            (row,col).
        """
        src_rows_num, src_cols_num, _ = src_image.shape
        dst_rows_num, dst_cols_num, _ = dst_image.shape
        src_edges = {}
        src_edges['upper left corner'] = np.array([1, 1, 1])
        src_edges['upper right corner'] = np.array([src_cols_num, 1, 1])
        src_edges['lower left corner'] = np.array([1, src_rows_num, 1])
        src_edges['lower right corner'] = \
            np.array([src_cols_num, src_rows_num, 1])
        transformed_edges = {}
        for corner_name, corner_location in src_edges.items():
            transformed_edges[corner_name] = homography @ corner_location
            transformed_edges[corner_name] /= transformed_edges[corner_name][-1]
        pad_up = pad_down = pad_right = pad_left = 0
        for corner_name, corner_location in transformed_edges.items():
            if corner_location[1] < 1:
                # pad up
                pad_up = max([pad_up, abs(corner_location[1])])
            if corner_location[0] > dst_cols_num:
                # pad right
                pad_right = max([pad_right,
                                 corner_location[0] - dst_cols_num])
            if corner_location[0] < 1:
                # pad left
                pad_left = max([pad_left, abs(corner_location[0])])
            if corner_location[1] > dst_rows_num:
                # pad down
                pad_down = max([pad_down,
                                corner_location[1] - dst_rows_num])
        panorama_cols_num = int(dst_cols_num + pad_right + pad_left)
        panorama_rows_num = int(dst_rows_num + pad_up + pad_down)
        pad_struct = PadStruct(pad_up=int(pad_up),
                               pad_down=int(pad_down),
                               pad_left=int(pad_left),
                               pad_right=int(pad_right))
        return panorama_rows_num, panorama_cols_num, pad_struct

    @staticmethod
    def add_translation_to_backward_homography(backward_homography: np.ndarray,
                                               pad_left: int,
                                               pad_up: int) -> np.ndarray:
        """Create a new homography which takes translation into account.

        Args:
            backward_homography: 3x3 Projective Homography matrix.
            pad_left: number of pixels that pad the destination image with
            zeros from left.
            pad_up: number of pixels that pad the destination image with
            zeros from the top.

        (1) Build the translation matrix from the pads.
        (2) Compose the backward homography and the translation matrix together.
        (3) Scale the homography as learnt in class.

        Returns:
            A new homography which includes the backward homography and the
            translation.
        """
        # return final_homography
        """INSERT YOUR CODE HERE"""
        # (1) Build the translation matrix
        translation_matrix = np.array([
            [1, 0, -pad_left],
            [0, 1, -pad_up],
            [0, 0, 1]
        ])
        
        # (2) Compose the homographies
        final_homography = backward_homography @ translation_matrix
        
        # (3) Scale (normalize)
        if abs(final_homography[2, 2]) > 1e-12:
            final_homography /= final_homography[2, 2]
            
        return final_homography

    def panorama(self,
                 src_image: np.ndarray,
                 dst_image: np.ndarray,
                 match_p_src: np.ndarray,
                 match_p_dst: np.ndarray,
                 inliers_percent: float,
                 max_err: float) -> np.ndarray:
        """Produces a panorama image from two images, and two lists of
        matching points, that deal with outliers using RANSAC.

        (1) Compute the forward homography and the panorama shape.
        (2) Compute the backward homography.
        (3) Add the appropriate translation to the homography so that the
        source image will plant in place.
        (4) Compute the backward warping with the appropriate translation.
        (5) Create the an empty panorama image and plant there the
        destination image.
        (6) place the backward warped image in the indices where the panorama
        image is zero.
        (7) Don't forget to clip the values of the image to [0, 255].


        Args:
            src_image: Source image expected to undergo projective
            transformation.
            dst_image: Destination image to which the source image is being
            mapped to.
            match_p_src: 2xN points from the source image.
            match_p_dst: 2xN points from the destination image.
            inliers_percent: The expected probability (between 0 and 1) of
            correct match points from the entire list of match points.
            max_err: A scalar that represents the maximum distance (in pixels)
            between the mapped src point to its corresponding dst point,
            in order to be considered as valid inlier.

        Returns:
            A panorama image.

        """
        # return np.clip(img_panorama, 0, 255).astype(np.uint8)
        """INSERT YOUR CODE HERE"""

        # (1) Compute the FORWARD homography
        H_forward = self.compute_homography(match_p_src, match_p_dst, 
                                            inliers_percent, max_err)
        
        # Find the shape of the combined image
        pano_h, pano_w, pads = self.find_panorama_shape(src_image, dst_image, H_forward)
        
        # (2) Compute the BACKWARD homography
        H_backward = self.compute_homography(match_p_dst, match_p_src,
                                             inliers_percent, max_err)

        # (3) Add translation to the backward homography
        H_backward_translated = self.add_translation_to_backward_homography(
            H_backward, pads.pad_left, pads.pad_up)

        # (4) Compute the backward warping
        panorama_shape = (pano_h, pano_w, 3)
        src_warped = self.compute_backward_mapping(
            H_backward_translated, src_image, panorama_shape)
        
        # (5) Create the empty panorama and plant the destination image
        img_panorama = np.zeros(panorama_shape, dtype=dst_image.dtype)
        img_panorama[pads.pad_up: pads.pad_up + dst_image.shape[0],
                     pads.pad_left: pads.pad_left + dst_image.shape[1]] = dst_image

        # (6) Place the backward warped image
        mask = (img_panorama == 0)
        img_panorama[mask] = src_warped[mask]

        # (7) Clip values
        return np.clip(img_panorama, 0, 255).astype(np.uint8)
