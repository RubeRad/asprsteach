# asprsteach
Code for making visuals for 2022 ASPRS photogrammetry-teaching competition

## Narration:

For hundreds of years, artists have used the principles of 
vanishing points to create realistic 3D perspective. 
Today we are going in the opposite direction; 
we will start with a perspective image, and discover
the vanishing points, in order to understand 
the perspective used by the camera.

This is a house. We can assume it's perpendicular edges
are aligned with the directions *East*, *North*, and *Up*. 
(There are also some diagonal lines.)

Although the edges are truly parallel in the real world, 
when that 3D scene is viewed in a 2D perspective, 
they appear to converge. The points at which all the lines for each
direction converge are called *Vanishing Points*, and the closer
lines are to appearing parallel, the further away the Vanishing Points
are, often far beyond the edges of the image.

This image has three vanishing points
(which is what we call 'Three Point Perspective'), 
and any three points form a triangle.
If we construct a perpendicular line from each side across to the opposite
vertex, all three lines intersect, and their intersection is 
known as the *center* of the triangle.
(This works for any triangle, try it!)
That center happens to be the *Principal Point* of the camera.

(Of all the light rays that fan into the camera, the *Principal Ray* 
is the unique ray that hits the film perpendicularly, and the 
Principal Point is where that ray hits. For a well-constructed camera, 
it's generally very near the center, but for a cropped image,
it could be anywhere.)

The three lines that intersect at the Principal Point appear to be in the
same flat image plane as the Vanishing Point Triangle, separated by 120
degrees each (because all the way around is 360). 
But that's only because we are viewing the focal plane from
directly above. If we can tilt our view to see it from the side, 
we can see that these lines
actually define a pyramid, which is known as the *Perspective Pyramid*.

The lines meet at an apex directly above the Principal Point. How far above?
Not so far that the angles at the apex are small and pointy; not so close
to the focal plane that the angles are near 120 degrees
There is a just-right height at which all three faces are right triangles. 
That height is the Focal Length.

Now that we have the Focal Length, that is the last piece of the puzzle, 
and we can assemble a mathematical model of the camera that
makes projection of 3D ground coordinates into 2D pixels as
easy as matrix multiplication.

But that math is not the story I want to tell here; I want to
help give a concrete understanding of Focal Length.

A shorter Focal Length means the camera has a wide field of view.
This is also known as *zooming out*.

When shortening the focal length, parallel lines become
*more* convergent, so Vanishing Points are closer in,
and the base of the Perspective Pyramid is smaller (to go with that
shorter Focal Length). 

Increasing the Focal Length, or *zooming in*, narrows the field of view,
and makes ground-parallel lines appear closer to parallel in perspective,
which pushes the Vanishing Points out and makes the Perspective Pyramid 
wider and taller.

Here's another view of zooming in, just looking at the Picture.
In addition to the apparent size of the house increasing, we can see the
lines getting more parallel. Imagine how those Vanishing 
Points are being pushed way further out for a larger Pyramid
(with a taller focal length).

And then zooming out, the object appears smaller, and lines that are
parallel in reality, appear so convergent that the house
looks quite distorted.

If you are zoomed out with a short focal length and the 
scene looks really small, you could make the house look 
bigger by zooming in, but another way is just to move the camera closer.
However, you can see that close range, combined with the short
focal length, results in a *really* distorted perspective.

We can zoom in a little bit, to make it look bigger, and then
move out a little bit, to make it smaller, and repeat in
alternating steps:
* zoom in; move out
* zoom in; move out
* zoom in; move out
* zoom in; move out

Or, starting from long range and long focal length, we could
carefully decrease both focal length and range simultaneously.
This is the camera
effect that Alfred Hitchcock invented to express 
dizziness in his masterpiece *Vertigo*.

And this is also why humanity has invented the 'selfie stick',
because we quickly discovered that smartphone photos with close range
and short focal length, distort the size of your nose!

I hope you enjoyed this explanation. If you want to see the python
code that made the animation, visit my github page.

 
