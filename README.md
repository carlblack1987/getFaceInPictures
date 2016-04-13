# getFaceInPictures

This is a cpp based facial and head angle analysis program using opencv.(Not completed)

It first use function GetSkin to get the face area by implementing skin detection rule.

Then extract face area as an isolated image to process.

The next step is to convert the face image from RGB to binary format. In binary format, the facial features such as eyes and mouth will be black and other skin area will be in white color.

Then in the function findFacialFeatures the face area will be divided into three parts: eye parts, mouth parts.

In the eyes part, the rough eye part will be obtained. Then the gradient method will be implemented to get the pupil's position.

In the mouth part, I will just get a rough position center of mouth.

The nose parts will be extracted based on the positions of eyes and mouth.

After getting facial features positions, the analysis will start.
Firstly, a center node in the middle of the line A linking two eyes will be defined. Then a line X between this node and the the mouth will be drawn, the slope of this line can be used to calculate the angle of the head.

Secondly, a ratio will be obtained by comparing the distance from the nose to the line X and the half length of line A. This ratio will help us to calculate the angle of face.

Well, it is still under experimenting.
