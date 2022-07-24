# asprsteach
Code for making visuals for 2022 ASPRS photogrammetry-teaching competition

## Narration:

For hundreds of years, artists have used the principles of 
vanishing points to create realistic 3D perspective. 
Today we are going in the opposite direction; 
using vanishing points observable in photographs
to *discover* the perspective used by the camera.

This is a house. We can assume it's perpendicular edges
are aligned with the directions *East*, *North*, and *Up*. 
(There are also some diagonal lines.)

Although the edges are truly parallel in the real world, 
when that 3D scene is viewed in a 2D perspective, 
they appear to converge. The points at which all the lines for each
direction converge are called *Vanishing Points*, and the closer
lines are to appearing parallel, the further away the Vanishing Points
are, often far beyond the edges of the image.

This image has three vanishing points, and any three points form a triangle.
If we construct a perpendicular line from each side across to the opposite
vertex, all three lines intersect at the *center* of the triangle.
That center happens to be the *Principal Point* of the camera.

(Of all the light rays that fan into the camera, the *Principal Ray* 
is the unique ray that hits the film perpendicularly, and the 
Principal Point is where that ray hits. For a well-constructed camera, 
it's generally very near the center, but for a cropped image,
it could be anywhere.)

The three lines that intersect at the Principal Point, divide
the triangle into three smaller triangles. The sum of the 3 angles 
around the Principal Point is 360 degrees; and the
tips of the triangles are 120 degrees each.
But if we step away from the film along a perpendicular line above the
Principal Point, those three triangles elevate to 
become three faces of a pyramid,
and the sum of the three angles decreases. 

At just the right the pyramid faces have 90 degrees each.
This right pyramid is called the *Perspective Pyramid*, and its
height is the camera's *Focal Length*.

A shorter Focal Length means the camera has a wide field of view,
parallel lines are more convergent, Vanishing Points are closer in,
so the base of the Perspective Pyramid is smaller (to go with that
shorter Focal Length). This is also known as *zooming out*.

Increasing the Focal Length, or *zooming in*, narrows the field of view,
and makes ground-parallel lines appear closer to parallel in perspective,
which pushes the Vanishing Points out and makes the Perspective Pyramid bigger.

So far, we have been looking at the house in *3 point perspective*, 
which is called that because all three ground directions converge to 
Vanishing Points, because our Principal Ray is not parallel or 
perpendicular to any of them; we are looking NorthEast, 
and diagonally downward.

But as the camera shifts, say, towards due North, which is 
perpendicular to East, the Eastbound ground lines start to lose 
their convergence. Their vanishing point moves arbitrarily 
far away from the scene, and the Perspective Pyramid degenerates
into an infinitely long A-frame, or a *Perspective Tent*. 



 
