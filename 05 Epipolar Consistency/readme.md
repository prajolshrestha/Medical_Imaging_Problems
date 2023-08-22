# Implementation of the epipolar consistency measure as described in "Efficient Epipolar Consistency" by Aichert et al. (2016)

Medical Image Processing

# Steps

- Load Projection
- Scale image (0.05) ie, Downscale
- Compute Radon Derivatives (Derivative of sinogram)


- Load Geometry
- Compute projection center
- get pluecker coordinates
- compute mapping circle to plane
   - Compute pluecker base moment
   - compute pluecker base direction
- compute mapping per projection


- Compute Consistency
