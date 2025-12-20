import cv2
import cairosvg
import numpy as np

# Convert SVG to PNG (in memory)
svg_code = "G:/Pract_Prog/Pract_Prog_F/test/02.svg"
png_data = cairosvg.svg2png(bytestring=svg_code)

# Convert PNG data to a NumPy array for cv2
nparr = np.frombuffer(png_data, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

# Now you can work with the image using cv2
cv2.imshow('SVG Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()